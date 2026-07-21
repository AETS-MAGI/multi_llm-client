use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::process::Command;

const DEFAULT_CONFIG: &str = r#"{
  "model_name": "tinyllama:latest",
  "endpoint": "http://127.0.0.1:11434/api/generate",
  "use_local_model": true,
  "local_framework": "ollama",
  "openai_compatible": false,
  "max_tokens": null,
  "api_key": null,
  "stream": true,
  "inline_stream": true,
  "temperature": 0.0,
  "num_ctx": null,
  "num_batch": null,
  "num_thread": null,
  "keep_alive": "10m",
  "request_timeout_secs": 300,
  "connect_timeout_secs": 5,
  "preset": "gfx900_safe",
  "python_command": "python3",
  "log_dir": "logs"
}"#;

#[derive(Debug, Deserialize, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
enum LocalFramework {
    Ollama,
    Python,
}

impl Default for LocalFramework {
    fn default() -> Self {
        Self::Ollama
    }
}

#[derive(Debug, Deserialize, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
enum Preset {
    Default,
    Gfx900Safe,
    Gfx900Balanced,
    Gfx900Longctx,
    Gfx900Tinybench,
    Gfx900AnchorBaseline,
    Gfx900AnchorSide1024,
}

impl Default for Preset {
    fn default() -> Self {
        Self::Default
    }
}

impl Preset {
    fn from_cli(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "default" => Some(Self::Default),
            "gfx900_safe" => Some(Self::Gfx900Safe),
            "gfx900_balanced" => Some(Self::Gfx900Balanced),
            "gfx900_longctx" => Some(Self::Gfx900Longctx),
            "gfx900_tinybench" => Some(Self::Gfx900Tinybench),
            "gfx900_anchor_baseline" => Some(Self::Gfx900AnchorBaseline),
            "gfx900_anchor_side1024" => Some(Self::Gfx900AnchorSide1024),
            _ => None,
        }
    }

    fn all_names() -> &'static [&'static str] {
        &[
            "default",
            "gfx900_safe",
            "gfx900_balanced",
            "gfx900_longctx",
            "gfx900_tinybench",
            "gfx900_anchor_baseline",
            "gfx900_anchor_side1024",
        ]
    }
}

#[derive(Debug, Deserialize, Clone)]
struct Config {
    model_name: String,
    endpoint: Option<String>,
    use_local_model: bool,
    local_framework: Option<LocalFramework>,
    openai_compatible: bool,
    max_tokens: Option<u32>,
    api_key: Option<String>,

    stream: Option<bool>,
    inline_stream: Option<bool>,
    temperature: Option<f32>,
    num_ctx: Option<u32>,
    num_batch: Option<u32>,
    num_thread: Option<u32>,
    keep_alive: Option<String>,
    request_timeout_secs: Option<u64>,
    connect_timeout_secs: Option<u64>,

    // Preferred modern preset key.
    preset: Option<Preset>,
    // Backward-compat key from previous revision.
    gfx900_preset: Option<bool>,

    python_command: Option<String>,
    log_dir: Option<String>,
}

impl Config {
    fn stream_enabled(&self) -> bool {
        self.stream.unwrap_or(true)
    }

    fn render_stream_inline(&self) -> bool {
        self.inline_stream.unwrap_or(true)
    }

    fn endpoint_for_ollama(&self) -> String {
        self.endpoint
            .clone()
            .unwrap_or_else(|| "http://127.0.0.1:11434/api/generate".to_string())
    }

    fn framework(&self) -> LocalFramework {
        self.local_framework.clone().unwrap_or_default()
    }

    fn preset(&self) -> Preset {
        if let Some(preset) = &self.preset {
            return preset.clone();
        }

        if self.gfx900_preset.unwrap_or(false) {
            Preset::Gfx900Safe
        } else {
            Preset::Default
        }
    }

    fn python_command(&self) -> &str {
        self.python_command.as_deref().unwrap_or("python3")
    }

    fn log_dir(&self) -> &str {
        self.log_dir.as_deref().unwrap_or("logs")
    }

    fn effective_max_tokens(&self) -> u32 {
        if let Some(v) = self.max_tokens {
            return v;
        }

        match self.preset() {
            Preset::Default => 256,
            Preset::Gfx900Safe => 128,
            Preset::Gfx900Balanced => 192,
            Preset::Gfx900Longctx => 128,
            Preset::Gfx900Tinybench => 32,
            Preset::Gfx900AnchorBaseline => 128,
            Preset::Gfx900AnchorSide1024 => 128,
        }
    }

    fn effective_num_ctx(&self) -> Option<u32> {
        if self.num_ctx.is_some() {
            return self.num_ctx;
        }

        match self.preset() {
            Preset::Default => None,
            Preset::Gfx900Safe => Some(4096),
            Preset::Gfx900Balanced => Some(4096),
            Preset::Gfx900Longctx => Some(8192),
            Preset::Gfx900Tinybench => Some(2048),
            Preset::Gfx900AnchorBaseline => Some(8192),
            Preset::Gfx900AnchorSide1024 => Some(8192),
        }
    }

    fn effective_num_batch(&self) -> Option<u32> {
        if self.num_batch.is_some() {
            return self.num_batch;
        }

        match self.preset() {
            Preset::Default => None,
            Preset::Gfx900Safe => Some(256),
            Preset::Gfx900Balanced => Some(512),
            Preset::Gfx900Longctx => Some(128),
            Preset::Gfx900Tinybench => Some(64),
            Preset::Gfx900AnchorBaseline => Some(512),
            Preset::Gfx900AnchorSide1024 => Some(1024),
        }
    }

    fn effective_num_thread(&self) -> Option<u32> {
        self.num_thread
    }

    fn connect_timeout(&self) -> Duration {
        Duration::from_secs(self.connect_timeout_secs.unwrap_or(5))
    }

    fn request_timeout(&self) -> Duration {
        Duration::from_secs(self.request_timeout_secs.unwrap_or(300))
    }

    fn should_stream_inline(&self) -> bool {
        self.stream_enabled()
            && self.render_stream_inline()
            && !self.openai_compatible
            && matches!(self.framework(), LocalFramework::Ollama)
    }
}

fn generate_default_config(path: &str) {
    let mut file = fs::File::create(path).expect("設定ファイルの作成に失敗しました");
    file.write_all(DEFAULT_CONFIG.as_bytes())
        .expect("設定ファイルの書き込みに失敗しました");
}

fn load_config(path: &str) -> Config {
    if !Path::new(path).exists() {
        println!("設定ファイルが見つかりません。デフォルト設定を作成します...");
        generate_default_config(path);
    }

    let config_data = fs::read_to_string(path).expect("設定ファイルの読み込みに失敗しました");
    serde_json::from_str(&config_data).expect("JSONのパースに失敗しました")
}

#[derive(Debug, Clone, Serialize)]
struct EffectiveConfig {
    model_name: String,
    endpoint: Option<String>,
    use_local_model: bool,
    framework: LocalFramework,
    openai_compatible: bool,
    stream: bool,
    inline_stream: bool,
    temperature: Option<f32>,
    keep_alive: Option<String>,
    preset: Preset,
    max_tokens: u32,
    num_ctx: Option<u32>,
    num_batch: Option<u32>,
    num_thread: Option<u32>,
    connect_timeout_secs: u64,
    request_timeout_secs: u64,
}

impl EffectiveConfig {
    fn from_config(config: &Config) -> Self {
        Self {
            model_name: config.model_name.clone(),
            endpoint: config.endpoint.clone(),
            use_local_model: config.use_local_model,
            framework: config.framework(),
            openai_compatible: config.openai_compatible,
            stream: config.stream_enabled(),
            inline_stream: config.render_stream_inline(),
            temperature: config.temperature,
            keep_alive: config.keep_alive.clone(),
            preset: config.preset(),
            max_tokens: config.effective_max_tokens(),
            num_ctx: config.effective_num_ctx(),
            num_batch: config.effective_num_batch(),
            num_thread: config.effective_num_thread(),
            connect_timeout_secs: config.connect_timeout().as_secs(),
            request_timeout_secs: config.request_timeout().as_secs(),
        }
    }
}

struct App {
    config: Config,
    effective: EffectiveConfig,
    client: reqwest::Client,
}

#[derive(Debug, Default, Clone)]
struct OllamaChunk {
    token: String,
    done: bool,
    prompt_eval_count: Option<u64>,
    prompt_eval_duration: Option<u64>,
    eval_count: Option<u64>,
    eval_duration: Option<u64>,
    total_duration: Option<u64>,
    load_duration: Option<u64>,
}

#[derive(Debug, Default, Clone, Serialize)]
struct OllamaFinalMetrics {
    prompt_eval_count: Option<u64>,
    prompt_eval_duration: Option<u64>,
    eval_count: Option<u64>,
    eval_duration: Option<u64>,
    total_duration: Option<u64>,
    load_duration: Option<u64>,
}

#[derive(Debug)]
struct InferenceStats {
    streaming_response: bool,
    started_at: Instant,
    first_token_at: Option<Instant>,
    finished_at: Option<Instant>,
    output_chars: usize,
    final_metrics: OllamaFinalMetrics,
}

impl InferenceStats {
    fn new(streaming_response: bool) -> Self {
        Self {
            streaming_response,
            started_at: Instant::now(),
            first_token_at: None,
            finished_at: None,
            output_chars: 0,
            final_metrics: OllamaFinalMetrics::default(),
        }
    }

    fn absorb_chunk(&mut self, chunk: &OllamaChunk) {
        if chunk.done && self.final_metrics.total_duration.is_none() {
            // done=true arrives before/with final metrics depending on backend behavior.
        }

        if !chunk.token.is_empty() {
            if self.first_token_at.is_none() {
                // In non-stream mode, a single aggregated response token often
                // arrives only after generation completes. Measuring first token
                // from that point would overestimate TTFT, so we only take
                // wall-clock first-token timing from streamed responses.
                if self.streaming_response {
                    self.first_token_at = Some(Instant::now());
                }
            }
            self.output_chars += chunk.token.chars().count();
        }

        if chunk.prompt_eval_count.is_some() {
            self.final_metrics.prompt_eval_count = chunk.prompt_eval_count;
        }
        if chunk.prompt_eval_duration.is_some() {
            self.final_metrics.prompt_eval_duration = chunk.prompt_eval_duration;
        }
        if chunk.eval_count.is_some() {
            self.final_metrics.eval_count = chunk.eval_count;
        }
        if chunk.eval_duration.is_some() {
            self.final_metrics.eval_duration = chunk.eval_duration;
        }
        if chunk.total_duration.is_some() {
            self.final_metrics.total_duration = chunk.total_duration;
        }
        if chunk.load_duration.is_some() {
            self.final_metrics.load_duration = chunk.load_duration;
        }
    }

    fn finish(&mut self) {
        self.finished_at = Some(Instant::now());
    }

    fn total_ms(&self) -> u128 {
        let wall_ms = self
            .finished_at
            .map(|t| t.duration_since(self.started_at).as_millis())
            .unwrap_or(0);
        let backend_ms = self
            .final_metrics
            .total_duration
            .map(|ns| (ns as u128) / 1_000_000);

        match backend_ms {
            Some(v) if wall_ms == 0 => v,
            Some(v) => wall_ms.max(v),
            None => wall_ms,
        }
    }

    fn ttft_ms(&self) -> Option<u128> {
        if let Some(v) = self
            .first_token_at
            .map(|t| t.duration_since(self.started_at).as_millis())
        {
            return Some(v);
        }

        // Fallback for non-stream mode where first-token timing cannot be
        // observed directly from chunk arrival.
        let prompt_ns = self.final_metrics.prompt_eval_duration?;
        let load_ns = self.final_metrics.load_duration.unwrap_or(0);
        Some(((prompt_ns as u128) + (load_ns as u128)) / 1_000_000)
    }

    fn approx_tok_per_sec(&self) -> Option<f64> {
        match (
            self.final_metrics.eval_count,
            self.final_metrics.eval_duration,
        ) {
            (Some(count), Some(duration_ns)) if duration_ns > 0 => {
                Some(count as f64 / (duration_ns as f64 / 1_000_000_000.0))
            }
            _ => None,
        }
    }

    fn print_summary(&self) {
        match (self.ttft_ms(), self.approx_tok_per_sec()) {
            (Some(ttft), Some(tok_s)) => println!(
                "[stats] ttft={}ms total={}ms output_chars={} eval_count={:?} tok/s={:.2}",
                ttft,
                self.total_ms(),
                self.output_chars,
                self.final_metrics.eval_count,
                tok_s
            ),
            (Some(ttft), None) => println!(
                "[stats] ttft={}ms total={}ms output_chars={} eval_count={:?}",
                ttft,
                self.total_ms(),
                self.output_chars,
                self.final_metrics.eval_count
            ),
            (None, Some(tok_s)) => println!(
                "[stats] ttft=n/a total={}ms output_chars={} eval_count={:?} tok/s={:.2}",
                self.total_ms(),
                self.output_chars,
                self.final_metrics.eval_count,
                tok_s
            ),
            (None, None) => println!(
                "[stats] ttft=n/a total={}ms output_chars={} eval_count={:?}",
                self.total_ms(),
                self.output_chars,
                self.final_metrics.eval_count
            ),
        }
    }
}

#[derive(Debug, Serialize)]
struct InferenceLogRecord {
    ts_unix_secs: u64,
    prompt_chars: usize,
    response_chars: usize,
    ttft_ms: Option<u128>,
    total_ms: u128,
    approx_tok_per_sec: Option<f64>,
    keep_alive_observability_min_ok: Option<bool>,
    error: Option<String>,
    effective: EffectiveConfig,
    ollama_metrics: OllamaFinalMetrics,
}

impl App {
    fn new(config: Config) -> Result<Self, String> {
        let effective = EffectiveConfig::from_config(&config);
        let client = reqwest::Client::builder()
            .connect_timeout(config.connect_timeout())
            .timeout(config.request_timeout())
            .tcp_nodelay(true)
            .pool_max_idle_per_host(8)
            .pool_idle_timeout(Duration::from_secs(90))
            .build()
            .map_err(|e| format!("HTTPクライアント生成エラー: {e}"))?;

        Ok(Self {
            config,
            effective,
            client,
        })
    }

    async fn infer(&self, prompt: &str) -> String {
        if self.config.use_local_model {
            match self.config.framework() {
                LocalFramework::Python => self.python_inference(prompt).await,
                LocalFramework::Ollama => self.ollama_inference(prompt).await,
            }
        } else if self.config.openai_compatible {
            self.openai_compatible_inference(prompt).await
        } else {
            self.ollama_inference(prompt).await
        }
    }

    async fn python_inference(&self, prompt: &str) -> String {
        let script_path = if Path::new("./src/llm_interface.py").exists() {
            "./src/llm_interface.py"
        } else if Path::new("./llm_interface.py").exists() {
            "./llm_interface.py"
        } else {
            return "Pythonスクリプトが見つかりません（./src/llm_interface.py または ./llm_interface.py）"
                .to_string();
        };

        let primary_cmd = self.config.python_command();
        let output = match Command::new(primary_cmd)
            .arg(script_path)
            .arg(prompt)
            .output()
            .await
        {
            Ok(out) => Ok(out),
            Err(e) if e.kind() == io::ErrorKind::NotFound && primary_cmd != "python" => {
                Command::new("python")
                    .arg(script_path)
                    .arg(prompt)
                    .output()
                    .await
            }
            Err(e) => Err(e),
        };

        match output {
            Ok(output) => {
                if output.status.success() {
                    String::from_utf8_lossy(&output.stdout).trim().to_string()
                } else {
                    format!(
                        "Pythonスクリプトエラー: {}",
                        String::from_utf8_lossy(&output.stderr)
                    )
                }
            }
            Err(e) => format!("Pythonスクリプト呼び出しエラー: {e}"),
        }
    }

    fn build_ollama_request(&self, prompt: &str) -> Value {
        let mut options = serde_json::Map::new();
        options.insert("num_predict".to_string(), json!(self.effective.max_tokens));

        if let Some(temp) = self.effective.temperature {
            options.insert("temperature".to_string(), json!(temp));
        }
        if let Some(num_ctx) = self.effective.num_ctx {
            options.insert("num_ctx".to_string(), json!(num_ctx));
        }
        if let Some(num_batch) = self.effective.num_batch {
            options.insert("num_batch".to_string(), json!(num_batch));
        }
        if let Some(num_thread) = self.effective.num_thread {
            options.insert("num_thread".to_string(), json!(num_thread));
        }

        let mut root = serde_json::Map::new();
        root.insert("model".to_string(), json!(self.effective.model_name));
        root.insert("prompt".to_string(), json!(prompt));
        root.insert("stream".to_string(), json!(self.effective.stream));
        root.insert("options".to_string(), Value::Object(options));

        if let Some(keep_alive) = &self.effective.keep_alive {
            root.insert("keep_alive".to_string(), json!(keep_alive));
        }

        Value::Object(root)
    }

    async fn ollama_inference(&self, prompt: &str) -> String {
        let endpoint = self.config.endpoint_for_ollama();
        let request_body = self.build_ollama_request(prompt);
        let mut req = self.client.post(&endpoint).json(&request_body);
        let mut stats = InferenceStats::new(self.effective.stream);

        if let Some(api_key) = &self.config.api_key {
            req = req.bearer_auth(api_key);
        }

        let res = match req.send().await {
            Ok(response) => response,
            Err(e) => {
                let err = format!("[error] Ollama推論エラー: {e}");
                stats.finish();
                let _ = self.write_log(prompt, "", Some(&stats), Some(err.clone()));
                return err;
            }
        };

        if !res.status().is_success() {
            let status = res.status();
            let body = res
                .text()
                .await
                .unwrap_or_else(|_| "<body read failed>".to_string());
            let err = format!("[error] Ollama HTTPエラー: status={} body={}", status, body);
            stats.finish();
            let _ = self.write_log(prompt, "", Some(&stats), Some(err.clone()));
            return err;
        }

        let result = if self.effective.stream {
            self.consume_ollama_stream(res, self.config.should_stream_inline(), &mut stats)
                .await
        } else {
            self.consume_ollama_non_stream(res, &mut stats).await
        };

        stats.finish();
        stats.print_summary();

        let error_for_log = if result.starts_with("[error]") {
            Some(result.clone())
        } else {
            None
        };
        let _ = self.write_log(prompt, &result, Some(&stats), error_for_log);

        result
    }

    async fn consume_ollama_non_stream(
        &self,
        response: reqwest::Response,
        stats: &mut InferenceStats,
    ) -> String {
        let text = match response.text().await {
            Ok(text) => text,
            Err(e) => return format!("[error] レスポンス本文の取得に失敗: {e}"),
        };

        let mut collected = String::new();

        // stream=false は通常「単一JSON」なので、まず全体を一括パースする。
        if let Ok(val) = serde_json::from_str::<Value>(&text) {
            match parse_ollama_value(&val) {
                Ok(chunk) => {
                    stats.absorb_chunk(&chunk);
                    if !chunk.token.is_empty() {
                        collected.push_str(&chunk.token);
                    }
                }
                Err(e) => return format!("[error] {e}"),
            }
        } else {
            // 互換性のため、NDJSON 形式で返る実装にもフォールバック対応する。
            for line in text.lines() {
                if line.trim().is_empty() {
                    continue;
                }
                match parse_ollama_line(line) {
                    Ok(chunk) => {
                        stats.absorb_chunk(&chunk);
                        if !chunk.token.is_empty() {
                            collected.push_str(&chunk.token);
                        }
                    }
                    Err(e) => return format!("[error] {e}"),
                }
            }
        }

        if collected.is_empty() {
            "[error] Ollama推論エラー（空レスポンス）".to_string()
        } else {
            collected
        }
    }

    async fn consume_ollama_stream(
        &self,
        response: reqwest::Response,
        render_stdout: bool,
        stats: &mut InferenceStats,
    ) -> String {
        let mut stream = response.bytes_stream();
        let mut pending = String::new();
        let mut collected = String::new();

        while let Some(chunk_res) = stream.next().await {
            let chunk = match chunk_res {
                Ok(c) => c,
                Err(e) => return format!("[error] ストリーム受信エラー: {e}"),
            };

            pending.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(idx) = pending.find('\n') {
                let line = pending[..idx].trim().to_string();
                pending.drain(..=idx);
                if line.is_empty() {
                    continue;
                }

                match parse_ollama_line(&line) {
                    Ok(parsed) => {
                        stats.absorb_chunk(&parsed);
                        if !parsed.token.is_empty() {
                            if render_stdout {
                                print!("{}", parsed.token);
                                let _ = io::stdout().flush();
                            }
                            collected.push_str(&parsed.token);
                        }
                    }
                    Err(e) => return format!("[error] {e}"),
                }
            }
        }

        if !pending.trim().is_empty() {
            match parse_ollama_line(pending.trim()) {
                Ok(parsed) => {
                    stats.absorb_chunk(&parsed);
                    if !parsed.token.is_empty() {
                        if render_stdout {
                            print!("{}", parsed.token);
                            let _ = io::stdout().flush();
                        }
                        collected.push_str(&parsed.token);
                    }
                }
                Err(e) => return format!("[error] {e}"),
            }
        }

        if collected.is_empty() {
            "[error] Ollama推論エラー（空レスポンス）".to_string()
        } else {
            collected
        }
    }

    async fn openai_compatible_inference(&self, prompt: &str) -> String {
        let endpoint = match &self.config.endpoint {
            Some(e) => e.as_str(),
            None => return "[error] openai_compatible=true では endpoint が必須です".to_string(),
        };

        let request_body = json!({
            "model": self.config.model_name,
            "prompt": prompt,
            "max_tokens": self.config.effective_max_tokens(),
            "temperature": self.config.temperature.unwrap_or(0.7)
        });

        let mut req = self.client.post(endpoint).json(&request_body);
        if let Some(api_key) = &self.config.api_key {
            req = req.bearer_auth(api_key);
        }

        let res = match req.send().await {
            Ok(r) => r,
            Err(e) => return format!("[error] OpenAI互換推論エラー: {e}"),
        };

        if !res.status().is_success() {
            let status = res.status();
            let body = res
                .text()
                .await
                .unwrap_or_else(|_| "<body read failed>".to_string());
            return format!(
                "[error] OpenAI互換HTTPエラー: status={} body={}",
                status, body
            );
        }

        let json_val: Value = match res.json().await {
            Ok(v) => v,
            Err(e) => return format!("[error] OpenAI互換レスポンスJSON解析エラー: {e}"),
        };

        if let Some(text) = json_val
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("text"))
            .and_then(|v| v.as_str())
        {
            return text.to_string();
        }
        if let Some(text) = json_val
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|v| v.as_str())
        {
            return text.to_string();
        }

        "[error] OpenAI互換レスポンスが不正です".to_string()
    }

    fn write_log(
        &self,
        prompt: &str,
        response: &str,
        stats: Option<&InferenceStats>,
        error: Option<String>,
    ) -> Result<(), String> {
        fs::create_dir_all(self.config.log_dir())
            .map_err(|e| format!("ログディレクトリ作成失敗: {e}"))?;

        let ts = current_unix_secs();
        let path = format!("{}/infer-{}.jsonl", self.config.log_dir(), ts_to_ymd(ts));

        let record = InferenceLogRecord {
            ts_unix_secs: ts,
            prompt_chars: prompt.chars().count(),
            response_chars: response.chars().count(),
            ttft_ms: stats.and_then(|s| s.ttft_ms()),
            total_ms: stats.map(|s| s.total_ms()).unwrap_or(0),
            approx_tok_per_sec: stats.and_then(|s| s.approx_tok_per_sec()),
            keep_alive_observability_min_ok: keep_alive_observability_min_ok(
                self.effective.keep_alive.as_deref(),
            ),
            error,
            effective: self.effective.clone(),
            ollama_metrics: stats.map(|s| s.final_metrics.clone()).unwrap_or_default(),
        };

        let line = serde_json::to_string(&record).map_err(|e| format!("ログJSON化失敗: {e}"))?;

        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| format!("ログファイルオープン失敗: {e}"))?;

        writeln!(file, "{line}").map_err(|e| format!("ログ書き込み失敗: {e}"))?;
        Ok(())
    }
}

fn parse_ollama_line(line: &str) -> Result<OllamaChunk, String> {
    let val: Value =
        serde_json::from_str(line).map_err(|e| format!("Ollamaレスポンス行のJSON解析失敗: {e}"))?;
    parse_ollama_value(&val)
}

fn parse_ollama_value(val: &Value) -> Result<OllamaChunk, String> {
    if let Some(err) = val.get("error").and_then(|e| e.as_str()) {
        return Err(format!("Ollamaエラー: {err}"));
    }

    let chunk = OllamaChunk {
        token: val
            .get("response")
            .and_then(|r| r.as_str())
            .or_else(|| val.get("thinking").and_then(|t| t.as_str()))
            .unwrap_or("")
            .to_string(),
        done: val.get("done").and_then(|d| d.as_bool()).unwrap_or(false),
        prompt_eval_count: val.get("prompt_eval_count").and_then(as_u64_flexible),
        prompt_eval_duration: val.get("prompt_eval_duration").and_then(as_u64_flexible),
        eval_count: val.get("eval_count").and_then(as_u64_flexible),
        eval_duration: val.get("eval_duration").and_then(as_u64_flexible),
        total_duration: val.get("total_duration").and_then(as_u64_flexible),
        load_duration: val.get("load_duration").and_then(as_u64_flexible),
    };

    Ok(chunk)
}

fn as_u64_flexible(v: &Value) -> Option<u64> {
    if let Some(n) = v.as_u64() {
        return Some(n);
    }
    if let Some(s) = v.as_str() {
        return s.parse::<u64>().ok();
    }
    None
}

fn current_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn ts_to_ymd(ts: u64) -> String {
    let days = ts / 86_400;
    let z = days as i64 + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = mp + if mp < 10 { 3 } else { -9 };
    let year = y + if m <= 2 { 1 } else { 0 };

    format!("{:04}-{:02}-{:02}", year, m, d)
}

fn print_effective_config(config: &Config, effective: &EffectiveConfig) {
    println!("モデル: {}", effective.model_name);
    println!(
        "モード: {} / framework={:?}",
        if config.use_local_model {
            "local"
        } else {
            "online"
        },
        effective.framework
    );
    println!("preset: {:?}", effective.preset);
    println!(
        "stream: {} / inline_stream: {} / openai_compatible: {}",
        if effective.stream { "on" } else { "off" },
        if effective.inline_stream { "on" } else { "off" },
        if effective.openai_compatible {
            "on"
        } else {
            "off"
        }
    );
    println!("effective_max_tokens: {}", effective.max_tokens);
    println!("effective_num_ctx: {:?}", effective.num_ctx);
    println!("effective_num_batch: {:?}", effective.num_batch);
    println!("effective_num_thread: {:?}", effective.num_thread);
    println!(
        "timeout: connect={}s request={}s",
        effective.connect_timeout_secs, effective.request_timeout_secs
    );
    println!("keep_alive: {:?}", effective.keep_alive);
    println!("endpoint: {:?}", effective.endpoint);
    println!("python_command: {}", config.python_command());
    println!("log_dir: {}", config.log_dir());
}

#[derive(Debug, Clone, Copy)]
enum BenchMode {
    PresetSweep,
    ThreadSweep,
    KeepaliveSweep,
    PredictSweep,
    All,
}

impl BenchMode {
    fn from_cli(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "preset-sweep" => Some(Self::PresetSweep),
            "thread-sweep" => Some(Self::ThreadSweep),
            "keepalive-sweep" => Some(Self::KeepaliveSweep),
            "predict-sweep" => Some(Self::PredictSweep),
            "all" => Some(Self::All),
            _ => None,
        }
    }

    fn all_names() -> &'static [&'static str] {
        &[
            "preset-sweep",
            "thread-sweep",
            "keepalive-sweep",
            "predict-sweep",
            "all",
        ]
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::PresetSweep => "preset-sweep",
            Self::ThreadSweep => "thread-sweep",
            Self::KeepaliveSweep => "keepalive-sweep",
            Self::PredictSweep => "predict-sweep",
            Self::All => "all",
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum BenchReportFormat {
    Tsv,
    Markdown,
    Json,
    All,
}

impl BenchReportFormat {
    fn from_cli(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "tsv" => Some(Self::Tsv),
            "markdown" | "md" => Some(Self::Markdown),
            "json" => Some(Self::Json),
            "all" => Some(Self::All),
            _ => None,
        }
    }

    fn all_names() -> &'static [&'static str] {
        &["tsv", "markdown", "json", "all"]
    }
}

fn preset_name(preset: &Preset) -> &'static str {
    match preset {
        Preset::Default => "default",
        Preset::Gfx900Safe => "gfx900_safe",
        Preset::Gfx900Balanced => "gfx900_balanced",
        Preset::Gfx900Longctx => "gfx900_longctx",
        Preset::Gfx900Tinybench => "gfx900_tinybench",
        Preset::Gfx900AnchorBaseline => "gfx900_anchor_baseline",
        Preset::Gfx900AnchorSide1024 => "gfx900_anchor_side1024",
    }
}

#[derive(Debug)]
struct CliArgs {
    config_path: String,
    prompt: Option<String>,
    repeat: u32,
    preset: Option<Preset>,
    model_name: Option<String>,
    keep_alive: Option<String>,
    clear_keep_alive: bool,
    num_thread: Option<u32>,
    clear_num_thread: bool,
    stream: Option<bool>,
    inline_stream: Option<bool>,
    bench_mode: Option<BenchMode>,
    bench_out: Option<String>,
    bench_threads_csv: Option<String>,
    bench_keep_alive_csv: Option<String>,
    bench_predict_values_csv: Option<String>,
    bench_report_input: Option<String>,
    bench_report_out: Option<String>,
    bench_report_format: BenchReportFormat,
    bench_compare_baseline: Option<String>,
    bench_compare_side: Option<String>,
    bench_compare_out: Option<String>,
    quiet: bool,
}

impl Default for CliArgs {
    fn default() -> Self {
        Self {
            config_path: "config.json".to_string(),
            prompt: None,
            repeat: 1,
            preset: None,
            model_name: None,
            keep_alive: None,
            clear_keep_alive: false,
            num_thread: None,
            clear_num_thread: false,
            stream: None,
            inline_stream: None,
            bench_mode: None,
            bench_out: None,
            bench_threads_csv: None,
            bench_keep_alive_csv: None,
            bench_predict_values_csv: None,
            bench_report_input: None,
            bench_report_out: None,
            bench_report_format: BenchReportFormat::Tsv,
            bench_compare_baseline: None,
            bench_compare_side: None,
            bench_compare_out: None,
            quiet: false,
        }
    }
}

fn print_usage() {
    println!(
        "Usage: multi_llm_client [options]\n\
         \n\
         Options:\n\
         \t--config <path>             Config file path (default: config.json)\n\
         \t--prompt <text>             Run one-shot inference (non-interactive)\n\
         \t--repeat <n>                Repeat count for one-shot mode (default: 1)\n\
         \t--preset <name>             Override preset (one of: {})\n\
         \t--model <model_name>        Override model_name\n\
         \t--keep-alive <value|none>   Override keep_alive (none clears config value)\n\
         \t--num-thread <n|none>       Override num_thread option (none clears config value)\n\
         \t--stream <true|false>       Override stream mode\n\
         \t--inline-stream <true|false> Override inline_stream mode\n\
         \t--bench <mode>              Run built-in benchmark mode (one of: {})\n\
         \t--out <path>                TSV output path for --bench mode\n\
         \t--threads <csv>             Thread set for --bench thread-sweep (default: 2,4,6)\n\
         \t--keep-alive-values <csv>   keep_alive set for --bench keepalive-sweep (default: 10s,30s,5m)\n\
         \t--predict-values <csv>      max_tokens set for --bench predict-sweep (default: 64,128,256,512,1024)\n\
         \t--bench-report <path>       Build grouped mode summary TSV from an existing bench TSV\n\
         \t--report-out <path>         Output path for --bench-report (default: <input>_mode_summary.tsv)\n\
         \t--report-format <fmt>       Output format for --bench-report (tsv|markdown|json|all; default: tsv)\n\
         \t--bench-compare <path>      Compare two phase summary TSVs (baseline file path)\n\
         \t--compare-side <path>       Side phase summary TSV for --bench-compare\n\
         \t--compare-out <path>        Output path for --bench-compare (default: <baseline>_vs_<side>.tsv)\n\
         \t--quiet                     Suppress banner and response echo in one-shot mode\n\
         \t-h, --help                  Show this help\n",
        Preset::all_names().join(", "),
        BenchMode::all_names().join(", ")
    );
}

fn parse_bool_arg(value: &str) -> Result<bool, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Ok(true),
        "false" | "0" | "no" | "off" => Ok(false),
        _ => Err(format!("invalid bool value: {value}")),
    }
}

fn parse_keep_alive_seconds(value: &str) -> Option<u64> {
    let v = value.trim().to_ascii_lowercase();
    if v.is_empty() {
        return None;
    }
    if v == "none" {
        return None;
    }

    let (num_str, unit) = if let Some(stripped) = v.strip_suffix('s') {
        (stripped, "s")
    } else if let Some(stripped) = v.strip_suffix('m') {
        (stripped, "m")
    } else if let Some(stripped) = v.strip_suffix('h') {
        (stripped, "h")
    } else {
        (v.as_str(), "s")
    };

    let n = num_str.parse::<u64>().ok()?;
    match unit {
        "s" => Some(n),
        "m" => n.checked_mul(60),
        "h" => n.checked_mul(3600),
        _ => None,
    }
}

fn keep_alive_observability_min_ok(keep_alive: Option<&str>) -> Option<bool> {
    keep_alive
        .and_then(parse_keep_alive_seconds)
        .map(|secs| secs >= 10)
}

fn parse_cli_args() -> Result<CliArgs, String> {
    let mut cli = CliArgs::default();
    let mut args = env::args().skip(1).peekable();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            "--config" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--config requires a path".to_string())?;
                cli.config_path = value;
            }
            "--prompt" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--prompt requires a value".to_string())?;
                cli.prompt = Some(value);
            }
            "--repeat" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--repeat requires a value".to_string())?;
                let parsed = value
                    .parse::<u32>()
                    .map_err(|_| format!("invalid --repeat value: {value}"))?;
                if parsed == 0 {
                    return Err("--repeat must be >= 1".to_string());
                }
                cli.repeat = parsed;
            }
            "--preset" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--preset requires a value".to_string())?;
                let parsed = Preset::from_cli(&value).ok_or_else(|| {
                    format!(
                        "invalid --preset value: {value} (expected one of: {})",
                        Preset::all_names().join(", ")
                    )
                })?;
                cli.preset = Some(parsed);
            }
            "--model" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--model requires a value".to_string())?;
                cli.model_name = Some(value);
            }
            "--keep-alive" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--keep-alive requires a value".to_string())?;
                if value.eq_ignore_ascii_case("none") {
                    cli.keep_alive = None;
                    cli.clear_keep_alive = true;
                } else {
                    cli.keep_alive = Some(value);
                    cli.clear_keep_alive = false;
                }
            }
            "--num-thread" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--num-thread requires a value".to_string())?;
                if value.eq_ignore_ascii_case("none") {
                    cli.num_thread = None;
                    cli.clear_num_thread = true;
                } else {
                    let parsed = value
                        .parse::<u32>()
                        .map_err(|_| format!("invalid --num-thread value: {value}"))?;
                    if parsed == 0 {
                        return Err("--num-thread must be >= 1".to_string());
                    }
                    cli.num_thread = Some(parsed);
                    cli.clear_num_thread = false;
                }
            }
            "--stream" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--stream requires a value".to_string())?;
                cli.stream = Some(parse_bool_arg(&value)?);
            }
            "--inline-stream" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--inline-stream requires a value".to_string())?;
                cli.inline_stream = Some(parse_bool_arg(&value)?);
            }
            "--bench" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--bench requires a value".to_string())?;
                let parsed = BenchMode::from_cli(&value).ok_or_else(|| {
                    format!(
                        "invalid --bench value: {value} (expected one of: {})",
                        BenchMode::all_names().join(", ")
                    )
                })?;
                cli.bench_mode = Some(parsed);
            }
            "--out" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--out requires a value".to_string())?;
                cli.bench_out = Some(value);
            }
            "--threads" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--threads requires a value".to_string())?;
                cli.bench_threads_csv = Some(value);
            }
            "--keep-alive-values" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--keep-alive-values requires a value".to_string())?;
                cli.bench_keep_alive_csv = Some(value);
            }
            "--predict-values" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--predict-values requires a value".to_string())?;
                cli.bench_predict_values_csv = Some(value);
            }
            "--bench-report" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--bench-report requires a value".to_string())?;
                cli.bench_report_input = Some(value);
            }
            "--report-out" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--report-out requires a value".to_string())?;
                cli.bench_report_out = Some(value);
            }
            "--report-format" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--report-format requires a value".to_string())?;
                let parsed = BenchReportFormat::from_cli(&value).ok_or_else(|| {
                    format!(
                        "invalid --report-format value: {value} (expected one of: {})",
                        BenchReportFormat::all_names().join(", ")
                    )
                })?;
                cli.bench_report_format = parsed;
            }
            "--bench-compare" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--bench-compare requires a value".to_string())?;
                cli.bench_compare_baseline = Some(value);
            }
            "--compare-side" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--compare-side requires a value".to_string())?;
                cli.bench_compare_side = Some(value);
            }
            "--compare-out" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--compare-out requires a value".to_string())?;
                cli.bench_compare_out = Some(value);
            }
            "--quiet" => {
                cli.quiet = true;
            }
            _ => {
                return Err(format!("unknown argument: {arg}"));
            }
        }
    }

    Ok(cli)
}

fn apply_cli_overrides(config: &mut Config, cli: &CliArgs) {
    if let Some(preset) = &cli.preset {
        config.preset = Some(preset.clone());
        config.gfx900_preset = None;
    }
    if let Some(model_name) = &cli.model_name {
        config.model_name = model_name.clone();
    }
    if let Some(stream) = cli.stream {
        config.stream = Some(stream);
    }
    if let Some(inline_stream) = cli.inline_stream {
        config.inline_stream = Some(inline_stream);
    }
    if cli.clear_keep_alive {
        config.keep_alive = None;
    } else if let Some(keep_alive) = &cli.keep_alive {
        config.keep_alive = Some(keep_alive.clone());
    }
    if cli.clear_num_thread {
        config.num_thread = None;
    } else if let Some(num_thread) = cli.num_thread {
        config.num_thread = Some(num_thread);
    }
}

fn parse_csv_tokens(csv: &str) -> Vec<String> {
    csv.split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn parse_threads_csv(csv: &str) -> Result<Vec<u32>, String> {
    let tokens = parse_csv_tokens(csv);
    if tokens.is_empty() {
        return Err("--threads must not be empty".to_string());
    }
    let mut out = Vec::new();
    for t in tokens {
        let parsed = t
            .parse::<u32>()
            .map_err(|_| format!("invalid --threads value: {t}"))?;
        if parsed == 0 {
            return Err("--threads must contain values >= 1".to_string());
        }
        out.push(parsed);
    }
    Ok(out)
}

fn parse_predict_values_csv(csv: &str) -> Result<Vec<u32>, String> {
    let tokens = parse_csv_tokens(csv);
    if tokens.is_empty() {
        return Err("--predict-values must not be empty".to_string());
    }
    let mut out = Vec::new();
    for t in tokens {
        let parsed = t
            .parse::<u32>()
            .map_err(|_| format!("invalid --predict-values value: {t}"))?;
        if parsed == 0 {
            return Err("--predict-values must contain values >= 1".to_string());
        }
        out.push(parsed);
    }
    Ok(out)
}

fn latest_log_record(log_dir: &str) -> Result<Value, String> {
    let day_path = format!("{}/infer-{}.jsonl", log_dir, ts_to_ymd(current_unix_secs()));
    let chosen_path = if Path::new(&day_path).exists() {
        day_path
    } else {
        let mut candidates = Vec::new();
        let dir = fs::read_dir(log_dir)
            .map_err(|e| format!("ログディレクトリ読み取り失敗 ({log_dir}): {e}"))?;
        for ent in dir {
            let ent = ent.map_err(|e| format!("ログディレクトリエントリ読み取り失敗: {e}"))?;
            let name = ent.file_name().to_string_lossy().to_string();
            if name.starts_with("infer-") && name.ends_with(".jsonl") {
                candidates.push(name);
            }
        }
        candidates.sort();
        let latest = candidates
            .last()
            .ok_or_else(|| format!("推論ログが見つかりません: {log_dir}"))?;
        format!("{log_dir}/{latest}")
    };

    let content = fs::read_to_string(&chosen_path)
        .map_err(|e| format!("推論ログ読み取り失敗 ({chosen_path}): {e}"))?;
    let line = content
        .lines()
        .rev()
        .find(|l| !l.trim().is_empty())
        .ok_or_else(|| format!("推論ログが空です: {chosen_path}"))?;

    serde_json::from_str::<Value>(line)
        .map_err(|e| format!("推論ログJSONパース失敗 ({chosen_path}): {e}"))
}

fn safe_tsv(s: &str) -> String {
    s.replace(['\t', '\n', '\r'], " ")
}

fn write_tsv_row(out_file: &mut fs::File, cols: &[String]) -> Result<(), String> {
    writeln!(out_file, "{}", cols.join("\t")).map_err(|e| format!("ベンチ行書き込み失敗: {e}"))
}

fn phase_signature(prompt_eval_ns: Option<u64>, eval_ns: Option<u64>) -> String {
    match (prompt_eval_ns.unwrap_or(0), eval_ns.unwrap_or(0)) {
        (p, e) if p > 0 && e > 0 => "prefill+decode".to_string(),
        (0, e) if e > 0 => "decode-only".to_string(),
        (p, 0) if p > 0 => "prefill-only".to_string(),
        _ => "unavailable".to_string(),
    }
}

#[derive(Default, Clone)]
struct BenchGroupStats {
    rows: u64,
    ok_rows: u64,
    phase_prefill_decode_rows: u64,
    phase_decode_only_rows: u64,
    phase_prefill_only_rows: u64,
    phase_unavailable_rows: u64,
    ttft_sum: f64,
    ttft_n: u64,
    total_sum: f64,
    total_n: u64,
    tok_sum: f64,
    tok_n: u64,
    prompt_eval_sum: f64,
    prompt_eval_n: u64,
    eval_sum: f64,
    eval_n: u64,
    decode_tok_sum: f64,
    decode_tok_n: u64,
    ratio_sum: f64,
    ratio_n: u64,
}

fn parse_f64_col(cols: &[&str], idx: usize) -> Option<f64> {
    cols.get(idx).and_then(|v| v.parse::<f64>().ok())
}

fn parse_u64_col(cols: &[&str], idx: usize) -> Option<u64> {
    cols.get(idx).and_then(|v| v.parse::<u64>().ok())
}

fn write_bench_phase_summary(out_path: &str) -> Result<String, String> {
    let content = fs::read_to_string(out_path)
        .map_err(|e| format!("ベンチ結果読み取り失敗 ({out_path}): {e}"))?;

    type GroupKey = (String, String, String, String, String, String, String);
    let mut groups: BTreeMap<GroupKey, BenchGroupStats> = BTreeMap::new();

    for line in content.lines().skip(1) {
        if line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 23 {
            continue;
        }

        let key = (
            cols[1].to_string(),  // mode
            cols[3].to_string(),  // preset_effective
            cols[4].to_string(),  // requested_preset
            cols[5].to_string(),  // num_thread
            cols[6].to_string(),  // keep_alive
            cols[15].to_string(), // max_tokens
            cols[22].to_string(), // phase_signature
        );

        let g = groups.entry(key).or_default();
        g.rows += 1;
        if cols[13] == "0" {
            g.ok_rows += 1;
        }
        match cols[22] {
            "prefill+decode" => g.phase_prefill_decode_rows += 1,
            "decode-only" => g.phase_decode_only_rows += 1,
            "prefill-only" => g.phase_prefill_only_rows += 1,
            _ => g.phase_unavailable_rows += 1,
        }

        if let Some(v) = parse_f64_col(&cols, 8) {
            g.ttft_sum += v;
            g.ttft_n += 1;
        }
        if let Some(v) = parse_f64_col(&cols, 9) {
            g.total_sum += v;
            g.total_n += 1;
        }
        if let Some(v) = parse_f64_col(&cols, 10) {
            g.tok_sum += v;
            g.tok_n += 1;
        }
        if let Some(v) = parse_f64_col(&cols, 17) {
            g.prompt_eval_sum += v;
            g.prompt_eval_n += 1;
        }
        if let Some(v) = parse_f64_col(&cols, 19) {
            g.eval_sum += v;
            g.eval_n += 1;
        }
        if let Some(v) = parse_f64_col(&cols, 20) {
            g.decode_tok_sum += v;
            g.decode_tok_n += 1;
        }
        if let Some(v) = parse_f64_col(&cols, 21) {
            g.ratio_sum += v;
            g.ratio_n += 1;
        }
    }

    let out = Path::new(out_path);
    let parent = out.parent().unwrap_or_else(|| Path::new("."));
    let stem = out
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| format!("ベンチ出力名解析失敗: {out_path}"))?;
    let summary_path = parent.join(format!("{stem}_phase_summary.tsv"));

    let mut file = fs::File::create(&summary_path).map_err(|e| {
        format!(
            "phase summary TSV作成失敗 ({}): {e}",
            summary_path.display()
        )
    })?;

    writeln!(
        file,
        "mode\tpreset_effective\trequested_preset\tnum_thread\tkeep_alive\tmax_tokens\tphase_signature\trows\tok_rows\tavg_ttft_ms\tavg_total_ms\tavg_tok_s\tavg_prompt_eval_ms\tavg_eval_ms\tavg_decode_tok_s_proxy\tavg_prefill_decode_ratio"
    )
    .map_err(|e| format!("phase summary ヘッダ書き込み失敗: {e}"))?;

    let avg = |sum: f64, n: u64| -> String {
        if n == 0 {
            String::new()
        } else {
            format!("{:.4}", sum / (n as f64))
        }
    };

    for (
        (mode, preset_effective, requested_preset, num_thread, keep_alive, max_tokens, phase),
        g,
    ) in groups
    {
        writeln!(
            file,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            mode,
            preset_effective,
            requested_preset,
            num_thread,
            keep_alive,
            max_tokens,
            phase,
            g.rows,
            g.ok_rows,
            avg(g.ttft_sum, g.ttft_n),
            avg(g.total_sum, g.total_n),
            avg(g.tok_sum, g.tok_n),
            avg(g.prompt_eval_sum, g.prompt_eval_n),
            avg(g.eval_sum, g.eval_n),
            avg(g.decode_tok_sum, g.decode_tok_n),
            avg(g.ratio_sum, g.ratio_n)
        )
        .map_err(|e| format!("phase summary 行書き込み失敗: {e}"))?;
    }

    Ok(summary_path.display().to_string())
}

#[derive(Debug, Clone, Serialize)]
struct BenchModeSummaryRow {
    mode: String,
    preset_effective: String,
    requested_preset: String,
    num_thread: String,
    keep_alive: String,
    max_tokens: String,
    rows: u64,
    ok_rows: u64,
    prefill_decode_rows: u64,
    decode_only_rows: u64,
    prefill_only_rows: u64,
    unavailable_rows: u64,
    avg_ttft_ms: Option<f64>,
    avg_total_ms: Option<f64>,
    avg_tok_s: Option<f64>,
    avg_prompt_eval_ms: Option<f64>,
    avg_eval_ms: Option<f64>,
    avg_decode_tok_s_proxy: Option<f64>,
    avg_prefill_decode_ratio: Option<f64>,
}

#[derive(Debug, Clone, Copy)]
enum ReportOutputKind {
    Tsv,
    Markdown,
    Json,
}

fn avg_opt(sum: f64, n: u64) -> Option<f64> {
    if n == 0 {
        None
    } else {
        Some(sum / n as f64)
    }
}

fn fmt_opt4_or_blank(v: Option<f64>) -> String {
    v.map(|x| format!("{x:.4}")).unwrap_or_default()
}

fn fmt_opt4_or_dash(v: Option<f64>) -> String {
    v.map(|x| format!("{x:.4}"))
        .unwrap_or_else(|| "-".to_string())
}

fn path_with_ext(path: &str, ext: &str) -> String {
    Path::new(path).with_extension(ext).display().to_string()
}

fn write_mode_summary_tsv(path: &str, rows: &[BenchModeSummaryRow]) -> Result<(), String> {
    let mut file = fs::File::create(path).map_err(|e| format!("bench report TSV作成失敗 ({path}): {e}"))?;
    writeln!(
        file,
        "mode\tpreset_effective\trequested_preset\tnum_thread\tkeep_alive\tmax_tokens\trows\tok_rows\tprefill_decode_rows\tdecode_only_rows\tprefill_only_rows\tunavailable_rows\tavg_ttft_ms\tavg_total_ms\tavg_tok_s\tavg_prompt_eval_ms\tavg_eval_ms\tavg_decode_tok_s_proxy\tavg_prefill_decode_ratio"
    )
    .map_err(|e| format!("bench report TSVヘッダ書き込み失敗: {e}"))?;

    for r in rows {
        writeln!(
            file,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            r.mode,
            r.preset_effective,
            r.requested_preset,
            r.num_thread,
            r.keep_alive,
            r.max_tokens,
            r.rows,
            r.ok_rows,
            r.prefill_decode_rows,
            r.decode_only_rows,
            r.prefill_only_rows,
            r.unavailable_rows,
            fmt_opt4_or_blank(r.avg_ttft_ms),
            fmt_opt4_or_blank(r.avg_total_ms),
            fmt_opt4_or_blank(r.avg_tok_s),
            fmt_opt4_or_blank(r.avg_prompt_eval_ms),
            fmt_opt4_or_blank(r.avg_eval_ms),
            fmt_opt4_or_blank(r.avg_decode_tok_s_proxy),
            fmt_opt4_or_blank(r.avg_prefill_decode_ratio),
        )
        .map_err(|e| format!("bench report TSV行書き込み失敗: {e}"))?;
    }

    Ok(())
}

fn write_mode_summary_markdown(path: &str, rows: &[BenchModeSummaryRow]) -> Result<(), String> {
    let mut file = fs::File::create(path).map_err(|e| format!("bench report Markdown作成失敗 ({path}): {e}"))?;
    writeln!(file, "# Bench Mode Summary").map_err(|e| format!("bench report Markdown書き込み失敗: {e}"))?;
    writeln!(file).map_err(|e| format!("bench report Markdown書き込み失敗: {e}"))?;
    writeln!(
        file,
        "| mode | preset_effective | requested_preset | num_thread | keep_alive | max_tokens | rows | ok_rows | phase(prefill/decode/prefill-only/unavail) | avg_ttft_ms | avg_total_ms | avg_tok_s | avg_decode_tok_s_proxy | avg_prefill_decode_ratio |"
    )
    .map_err(|e| format!("bench report Markdownヘッダ書き込み失敗: {e}"))?;
    writeln!(
        file,
        "|---|---|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|"
    )
    .map_err(|e| format!("bench report Markdownヘッダ書き込み失敗: {e}"))?;

    for r in rows {
        let phase = format!(
            "{}/{}/{}/{}",
            r.prefill_decode_rows, r.decode_only_rows, r.prefill_only_rows, r.unavailable_rows
        );
        writeln!(
            file,
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            r.mode,
            r.preset_effective,
            r.requested_preset,
            r.num_thread,
            r.keep_alive,
            r.max_tokens,
            r.rows,
            r.ok_rows,
            phase,
            fmt_opt4_or_dash(r.avg_ttft_ms),
            fmt_opt4_or_dash(r.avg_total_ms),
            fmt_opt4_or_dash(r.avg_tok_s),
            fmt_opt4_or_dash(r.avg_decode_tok_s_proxy),
            fmt_opt4_or_dash(r.avg_prefill_decode_ratio),
        )
        .map_err(|e| format!("bench report Markdown行書き込み失敗: {e}"))?;
    }

    Ok(())
}

fn write_mode_summary_json(path: &str, rows: &[BenchModeSummaryRow]) -> Result<(), String> {
    let file = fs::File::create(path).map_err(|e| format!("bench report JSON作成失敗 ({path}): {e}"))?;
    serde_json::to_writer_pretty(file, rows)
        .map_err(|e| format!("bench report JSON書き込み失敗 ({path}): {e}"))?;
    Ok(())
}

fn write_bench_mode_summary(
    input_path: &str,
    report_out: Option<&str>,
    report_format: BenchReportFormat,
) -> Result<Vec<String>, String> {
    let content = fs::read_to_string(input_path)
        .map_err(|e| format!("ベンチ結果読み取り失敗 ({input_path}): {e}"))?;

    type GroupKey = (String, String, String, String, String, String);
    let mut groups: BTreeMap<GroupKey, BenchGroupStats> = BTreeMap::new();

    for line in content.lines().skip(1) {
        if line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 23 {
            continue;
        }

        let key = (
            cols[1].to_string(),  // mode
            cols[3].to_string(),  // preset_effective
            cols[4].to_string(),  // requested_preset
            cols[5].to_string(),  // num_thread
            cols[6].to_string(),  // keep_alive
            cols[15].to_string(), // max_tokens
        );
        let g = groups.entry(key).or_default();

        g.rows += 1;
        if cols[13] == "0" {
            g.ok_rows += 1;
        }
        match cols[22] {
            "prefill+decode" => g.phase_prefill_decode_rows += 1,
            "decode-only" => g.phase_decode_only_rows += 1,
            "prefill-only" => g.phase_prefill_only_rows += 1,
            _ => g.phase_unavailable_rows += 1,
        }

        if let Some(v) = parse_f64_col(&cols, 8) {
            g.ttft_sum += v;
            g.ttft_n += 1;
        }
        if let Some(v) = parse_f64_col(&cols, 9) {
            g.total_sum += v;
            g.total_n += 1;
        }
        if let Some(v) = parse_f64_col(&cols, 10) {
            g.tok_sum += v;
            g.tok_n += 1;
        }
        if let Some(v) = parse_f64_col(&cols, 17) {
            g.prompt_eval_sum += v;
            g.prompt_eval_n += 1;
        }
        if let Some(v) = parse_f64_col(&cols, 19) {
            g.eval_sum += v;
            g.eval_n += 1;
        }
        if let Some(v) = parse_f64_col(&cols, 20) {
            g.decode_tok_sum += v;
            g.decode_tok_n += 1;
        }
        if let Some(v) = parse_f64_col(&cols, 21) {
            g.ratio_sum += v;
            g.ratio_n += 1;
        }
    }

    let rows: Vec<BenchModeSummaryRow> = groups
        .into_iter()
        .map(|((mode, preset_effective, requested_preset, num_thread, keep_alive, max_tokens), g)| {
            BenchModeSummaryRow {
                mode,
                preset_effective,
                requested_preset,
                num_thread,
                keep_alive,
                max_tokens,
                rows: g.rows,
                ok_rows: g.ok_rows,
                prefill_decode_rows: g.phase_prefill_decode_rows,
                decode_only_rows: g.phase_decode_only_rows,
                prefill_only_rows: g.phase_prefill_only_rows,
                unavailable_rows: g.phase_unavailable_rows,
                avg_ttft_ms: avg_opt(g.ttft_sum, g.ttft_n),
                avg_total_ms: avg_opt(g.total_sum, g.total_n),
                avg_tok_s: avg_opt(g.tok_sum, g.tok_n),
                avg_prompt_eval_ms: avg_opt(g.prompt_eval_sum, g.prompt_eval_n),
                avg_eval_ms: avg_opt(g.eval_sum, g.eval_n),
                avg_decode_tok_s_proxy: avg_opt(g.decode_tok_sum, g.decode_tok_n),
                avg_prefill_decode_ratio: avg_opt(g.ratio_sum, g.ratio_n),
            }
        })
        .collect();

    let input = Path::new(input_path);
    let parent = input.parent().unwrap_or_else(|| Path::new("."));
    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| format!("bench report入力名解析失敗: {input_path}"))?;
    let default_tsv = parent
        .join(format!("{stem}_mode_summary.tsv"))
        .display()
        .to_string();

    let outputs: Vec<(ReportOutputKind, String)> = match report_format {
        BenchReportFormat::Tsv => {
            vec![(ReportOutputKind::Tsv, report_out.unwrap_or(&default_tsv).to_string())]
        }
        BenchReportFormat::Markdown => {
            let path = if let Some(v) = report_out {
                v.to_string()
            } else {
                parent
                    .join(format!("{stem}_mode_summary.md"))
                    .display()
                    .to_string()
            };
            vec![(ReportOutputKind::Markdown, path)]
        }
        BenchReportFormat::Json => {
            let path = if let Some(v) = report_out {
                v.to_string()
            } else {
                parent
                    .join(format!("{stem}_mode_summary.json"))
                    .display()
                    .to_string()
            };
            vec![(ReportOutputKind::Json, path)]
        }
        BenchReportFormat::All => {
            let tsv = report_out.unwrap_or(&default_tsv).to_string();
            let md = path_with_ext(&tsv, "md");
            let json = path_with_ext(&tsv, "json");
            vec![
                (ReportOutputKind::Tsv, tsv),
                (ReportOutputKind::Markdown, md),
                (ReportOutputKind::Json, json),
            ]
        }
    };

    let mut generated_paths = Vec::new();
    for (kind, path) in outputs {
        let out_parent = Path::new(&path).parent().unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(out_parent).map_err(|e| {
            format!(
                "bench report出力先ディレクトリ作成失敗 ({:?}): {e}",
                out_parent
            )
        })?;

        match kind {
            ReportOutputKind::Tsv => write_mode_summary_tsv(&path, &rows)?,
            ReportOutputKind::Markdown => write_mode_summary_markdown(&path, &rows)?,
            ReportOutputKind::Json => write_mode_summary_json(&path, &rows)?,
        }
        generated_paths.push(path);
    }

    Ok(generated_paths)
}

#[derive(Default, Clone)]
struct WeightedMetric {
    sum: f64,
    weight: u64,
}

impl WeightedMetric {
    fn add(&mut self, value: Option<f64>, row_weight: u64) {
        if let Some(v) = value {
            self.sum += v * row_weight as f64;
            self.weight += row_weight;
        }
    }

    fn avg(&self) -> Option<f64> {
        if self.weight == 0 {
            None
        } else {
            Some(self.sum / self.weight as f64)
        }
    }
}

#[derive(Default, Clone)]
struct PhaseSummaryAgg {
    preset_effective_set: BTreeSet<String>,
    requested_preset_set: BTreeSet<String>,
    rows: u64,
    ok_rows: u64,
    ttft: WeightedMetric,
    total: WeightedMetric,
    tok_s: WeightedMetric,
    prompt_eval_ms: WeightedMetric,
    eval_ms: WeightedMetric,
    decode_tok_s: WeightedMetric,
    prefill_decode_ratio: WeightedMetric,
}

type CompareKey = (String, String, String, String, String);

fn parse_phase_summary_agg(path: &str) -> Result<BTreeMap<CompareKey, PhaseSummaryAgg>, String> {
    let content = fs::read_to_string(path)
        .map_err(|e| format!("phase summary 読み取り失敗 ({path}): {e}"))?;

    let mut map: BTreeMap<CompareKey, PhaseSummaryAgg> = BTreeMap::new();

    for line in content.lines().skip(1) {
        if line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 16 {
            continue;
        }

        let key: CompareKey = (
            cols[0].to_string(), // mode
            cols[3].to_string(), // num_thread
            cols[4].to_string(), // keep_alive
            cols[5].to_string(), // max_tokens
            cols[6].to_string(), // phase_signature
        );
        let entry = map.entry(key).or_default();
        entry.preset_effective_set.insert(cols[1].to_string());
        entry.requested_preset_set.insert(cols[2].to_string());

        let row_weight = parse_u64_col(&cols, 7).unwrap_or(1);
        entry.rows += row_weight;
        entry.ok_rows += parse_u64_col(&cols, 8).unwrap_or(0);

        entry.ttft.add(parse_f64_col(&cols, 9), row_weight);
        entry.total.add(parse_f64_col(&cols, 10), row_weight);
        entry.tok_s.add(parse_f64_col(&cols, 11), row_weight);
        entry
            .prompt_eval_ms
            .add(parse_f64_col(&cols, 12), row_weight);
        entry.eval_ms.add(parse_f64_col(&cols, 13), row_weight);
        entry.decode_tok_s.add(parse_f64_col(&cols, 14), row_weight);
        entry
            .prefill_decode_ratio
            .add(parse_f64_col(&cols, 15), row_weight);
    }

    Ok(map)
}

fn fmt_opt4(v: Option<f64>) -> String {
    v.map(|x| format!("{x:.4}")).unwrap_or_default()
}

fn delta_opt(side: Option<f64>, base: Option<f64>) -> Option<f64> {
    Some(side? - base?)
}

fn ratio_opt(side: Option<f64>, base: Option<f64>) -> Option<f64> {
    let b = base?;
    if b == 0.0 {
        None
    } else {
        Some(side? / b)
    }
}

fn set_to_csv(set: &BTreeSet<String>) -> String {
    if set.is_empty() {
        String::new()
    } else {
        set.iter().cloned().collect::<Vec<_>>().join(",")
    }
}

fn write_bench_compare_summary(
    baseline_path: &str,
    side_path: &str,
    out_path: Option<&str>,
) -> Result<String, String> {
    let baseline = parse_phase_summary_agg(baseline_path)?;
    let side = parse_phase_summary_agg(side_path)?;

    let final_out = if let Some(v) = out_path {
        v.to_string()
    } else {
        let b = Path::new(baseline_path);
        let s = Path::new(side_path);
        let parent = b.parent().unwrap_or_else(|| Path::new("."));
        let b_stem = b
            .file_stem()
            .and_then(|v| v.to_str())
            .ok_or_else(|| format!("baseline名解析失敗: {baseline_path}"))?;
        let s_stem = s
            .file_stem()
            .and_then(|v| v.to_str())
            .ok_or_else(|| format!("side名解析失敗: {side_path}"))?;
        parent
            .join(format!("{b_stem}_vs_{s_stem}.tsv"))
            .display()
            .to_string()
    };

    let out_parent = Path::new(&final_out)
        .parent()
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(out_parent).map_err(|e| {
        format!(
            "bench compare出力先ディレクトリ作成失敗 ({:?}): {e}",
            out_parent
        )
    })?;

    let mut file = fs::File::create(&final_out)
        .map_err(|e| format!("bench compare出力失敗 ({final_out}): {e}"))?;

    writeln!(
        file,
        "mode\tnum_thread\tkeep_alive\tmax_tokens\tphase_signature\tstatus\tbaseline_preset_effective\tside_preset_effective\tbaseline_rows\tside_rows\tbaseline_ok_rows\tside_ok_rows\tbaseline_avg_ttft_ms\tside_avg_ttft_ms\tdelta_ttft_ms\tratio_ttft\tbaseline_avg_total_ms\tside_avg_total_ms\tdelta_total_ms\tratio_total\tbaseline_avg_tok_s\tside_avg_tok_s\tdelta_tok_s\tratio_tok_s\tbaseline_avg_decode_tok_s\tside_avg_decode_tok_s\tdelta_decode_tok_s\tratio_decode_tok_s\tbaseline_avg_prefill_decode_ratio\tside_avg_prefill_decode_ratio\tdelta_prefill_decode_ratio"
    )
    .map_err(|e| format!("bench compareヘッダ書き込み失敗: {e}"))?;

    let mut keys: BTreeSet<CompareKey> = BTreeSet::new();
    keys.extend(baseline.keys().cloned());
    keys.extend(side.keys().cloned());

    for key in keys {
        let (mode, num_thread, keep_alive, max_tokens, phase_sig) = (
            key.0.clone(),
            key.1.clone(),
            key.2.clone(),
            key.3.clone(),
            key.4.clone(),
        );

        let base = baseline.get(&key);
        let side_v = side.get(&key);

        let status = match (base, side_v) {
            (Some(_), Some(_)) => "matched",
            (Some(_), None) => "baseline_only",
            (None, Some(_)) => "side_only",
            _ => "unavailable",
        };

        let (
            base_rows,
            base_ok,
            base_preset,
            base_ttft,
            base_total,
            base_tok,
            base_decode,
            base_ratio,
        ) = if let Some(v) = base {
            (
                v.rows.to_string(),
                v.ok_rows.to_string(),
                set_to_csv(&v.preset_effective_set),
                v.ttft.avg(),
                v.total.avg(),
                v.tok_s.avg(),
                v.decode_tok_s.avg(),
                v.prefill_decode_ratio.avg(),
            )
        } else {
            (
                String::new(),
                String::new(),
                String::new(),
                None,
                None,
                None,
                None,
                None,
            )
        };

        let (
            side_rows,
            side_ok,
            side_preset,
            side_ttft,
            side_total,
            side_tok,
            side_decode,
            side_ratio,
        ) = if let Some(v) = side_v {
            (
                v.rows.to_string(),
                v.ok_rows.to_string(),
                set_to_csv(&v.preset_effective_set),
                v.ttft.avg(),
                v.total.avg(),
                v.tok_s.avg(),
                v.decode_tok_s.avg(),
                v.prefill_decode_ratio.avg(),
            )
        } else {
            (
                String::new(),
                String::new(),
                String::new(),
                None,
                None,
                None,
                None,
                None,
            )
        };

        let cols = vec![
            mode,
            num_thread,
            keep_alive,
            max_tokens,
            phase_sig,
            status.to_string(),
            base_preset,
            side_preset,
            base_rows,
            side_rows,
            base_ok,
            side_ok,
            fmt_opt4(base_ttft),
            fmt_opt4(side_ttft),
            fmt_opt4(delta_opt(side_ttft, base_ttft)),
            fmt_opt4(ratio_opt(side_ttft, base_ttft)),
            fmt_opt4(base_total),
            fmt_opt4(side_total),
            fmt_opt4(delta_opt(side_total, base_total)),
            fmt_opt4(ratio_opt(side_total, base_total)),
            fmt_opt4(base_tok),
            fmt_opt4(side_tok),
            fmt_opt4(delta_opt(side_tok, base_tok)),
            fmt_opt4(ratio_opt(side_tok, base_tok)),
            fmt_opt4(base_decode),
            fmt_opt4(side_decode),
            fmt_opt4(delta_opt(side_decode, base_decode)),
            fmt_opt4(ratio_opt(side_decode, base_decode)),
            fmt_opt4(base_ratio),
            fmt_opt4(side_ratio),
            fmt_opt4(delta_opt(side_ratio, base_ratio)),
        ];
        write_tsv_row(&mut file, &cols)?;
    }

    Ok(final_out)
}

fn append_bench_worklog_summary(
    out_path: &str,
    bench_mode: BenchMode,
    repeat: u32,
    phase_summary_path: Option<&str>,
) -> Result<(), String> {
    let content = fs::read_to_string(out_path)
        .map_err(|e| format!("ベンチ結果読み取り失敗 ({out_path}): {e}"))?;

    let mut rows = 0u64;
    let mut ok_rows = 0u64;

    let mut ttft_sum = 0f64;
    let mut ttft_n = 0u64;
    let mut total_sum = 0f64;
    let mut total_n = 0u64;
    let mut tok_sum = 0f64;
    let mut tok_n = 0u64;
    let mut prefill_sum = 0f64;
    let mut prefill_n = 0u64;
    let mut decode_sum = 0f64;
    let mut decode_n = 0u64;
    let mut phase_counts: BTreeMap<String, u64> = BTreeMap::new();
    let mut preset_counts: BTreeMap<String, u64> = BTreeMap::new();

    for line in content.lines().skip(1) {
        if line.trim().is_empty() {
            continue;
        }
        rows += 1;
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 15 {
            continue;
        }

        if cols[13] == "0" {
            ok_rows += 1;
        }
        if let Some(v) = parse_f64_col(&cols, 8) {
            ttft_sum += v;
            ttft_n += 1;
        }
        if let Some(v) = parse_f64_col(&cols, 9) {
            total_sum += v;
            total_n += 1;
        }
        if let Some(v) = parse_f64_col(&cols, 10) {
            tok_sum += v;
            tok_n += 1;
        }
        if let Some(v) = parse_f64_col(&cols, 17) {
            prefill_sum += v;
            prefill_n += 1;
        }
        if let Some(v) = parse_f64_col(&cols, 19) {
            decode_sum += v;
            decode_n += 1;
        }
        if cols.len() > 22 && !cols[22].is_empty() {
            *phase_counts.entry(cols[22].to_string()).or_insert(0) += 1;
        }
        if !cols[4].is_empty() {
            *preset_counts.entry(cols[4].to_string()).or_insert(0) += 1;
        }
    }

    if rows == 0 {
        return Ok(());
    }

    let avg = |sum: f64, n: u64| -> String {
        if n == 0 {
            "n/a".to_string()
        } else {
            format!("{:.2}", sum / (n as f64))
        }
    };

    let dominant_phase = phase_counts
        .iter()
        .max_by_key(|(_, count)| *count)
        .map(|(k, _)| k.clone())
        .unwrap_or_else(|| "n/a".to_string());

    let preset_digest = if preset_counts.is_empty() {
        "n/a".to_string()
    } else {
        preset_counts
            .iter()
            .map(|(k, v)| format!("{k}:{v}"))
            .collect::<Vec<_>>()
            .join(",")
    };

    let summary_path = format!(
        "worklog/bench_auto_summary_{}.md",
        ts_to_ymd(current_unix_secs())
    );
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&summary_path)
        .map_err(|e| format!("worklog追記失敗 ({summary_path}): {e}"))?;

    if file
        .metadata()
        .map_err(|e| format!("worklogメタデータ取得失敗: {e}"))?
        .len()
        == 0
    {
        writeln!(file, "# Bench Auto Summary")
            .map_err(|e| format!("worklogヘッダ書き込み失敗: {e}"))?;
        writeln!(file).map_err(|e| format!("worklog改行書き込み失敗: {e}"))?;
    }

    writeln!(
        file,
        "- ts={} mode={} repeat={} rows={} ok={} avg_ttft_ms={} avg_total_ms={} avg_tok_s={} avg_prefill_ms={} avg_decode_eval_ms={} dominant_phase={} presets={} out={} phase_summary={}",
        current_unix_secs(),
        bench_mode.as_str(),
        repeat,
        rows,
        ok_rows,
        avg(ttft_sum, ttft_n),
        avg(total_sum, total_n),
        avg(tok_sum, tok_n),
        avg(prefill_sum, prefill_n),
        avg(decode_sum, decode_n),
        dominant_phase,
        preset_digest,
        out_path,
        phase_summary_path.unwrap_or("n/a")
    )
    .map_err(|e| format!("worklog追記失敗: {e}"))?;

    Ok(())
}

async fn run_bench_case(
    base_config: &Config,
    mode: &str,
    prompt: &str,
    repeat: u32,
    requested_preset: Option<Preset>,
    requested_preset_name: &str,
    num_thread_override: Option<u32>,
    keep_alive_override: Option<&str>,
    max_tokens_override: Option<u32>,
    out_file: &mut fs::File,
) -> Result<(), String> {
    for rep_idx in 1..=repeat {
        let mut run_config = base_config.clone();
        run_config.stream = Some(false);
        run_config.inline_stream = Some(false);

        if let Some(preset) = requested_preset.clone() {
            run_config.preset = Some(preset);
            run_config.gfx900_preset = None;
        }
        if let Some(t) = num_thread_override {
            run_config.num_thread = Some(t);
        }
        if let Some(keep_alive) = keep_alive_override {
            run_config.keep_alive = Some(keep_alive.to_string());
        }
        if let Some(max_tokens) = max_tokens_override {
            run_config.max_tokens = Some(max_tokens);
        }

        let app = match App::new(run_config.clone()) {
            Ok(v) => v,
            Err(e) => {
                let effective = EffectiveConfig::from_config(&run_config);
                let row = vec![
                    "0".to_string(),
                    mode.to_string(),
                    run_config.model_name.clone(),
                    String::new(),
                    requested_preset_name.to_string(),
                    run_config
                        .num_thread
                        .map(|v| v.to_string())
                        .unwrap_or_else(|| "none".to_string()),
                    run_config
                        .keep_alive
                        .clone()
                        .unwrap_or_else(|| "none".to_string()),
                    rep_idx.to_string(),
                    String::new(),
                    String::new(),
                    String::new(),
                    String::new(),
                    String::new(),
                    "1".to_string(),
                    safe_tsv(&e),
                    effective.max_tokens.to_string(),
                    String::new(),
                    String::new(),
                    String::new(),
                    String::new(),
                    String::new(),
                    String::new(),
                    "unavailable".to_string(),
                ];
                write_tsv_row(out_file, &row)?;
                continue;
            }
        };

        let response = app.infer(prompt).await;
        let record = latest_log_record(run_config.log_dir());

        let mut ts_unix = "0".to_string();
        let mut preset_effective = String::new();
        let mut ttft_ms = String::new();
        let mut total_ms = String::new();
        let mut tok_s = String::new();
        let mut response_chars = String::new();
        let mut keep_alive_obs_ok = String::new();
        let mut error_text = String::new();
        let mut rc = 0;
        let mut max_tokens = app.effective.max_tokens.to_string();
        let mut prompt_eval_count = String::new();
        let mut prompt_eval_ms = String::new();
        let mut eval_count = String::new();
        let mut eval_ms = String::new();
        let mut decode_tok_s_proxy = String::new();
        let mut prefill_decode_ratio = String::new();
        let mut phase_sig = "unavailable".to_string();

        match record {
            Ok(log) => {
                if let Some(v) = log.get("ts_unix_secs").and_then(as_u64_flexible) {
                    ts_unix = v.to_string();
                }
                preset_effective = log
                    .get("effective")
                    .and_then(|e| e.get("preset"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                if let Some(v) = log
                    .get("effective")
                    .and_then(|e| e.get("max_tokens"))
                    .and_then(as_u64_flexible)
                {
                    max_tokens = v.to_string();
                }
                ttft_ms = log
                    .get("ttft_ms")
                    .and_then(as_u64_flexible)
                    .map(|v| v.to_string())
                    .unwrap_or_default();
                total_ms = log
                    .get("total_ms")
                    .and_then(as_u64_flexible)
                    .map(|v| v.to_string())
                    .unwrap_or_default();
                tok_s = log
                    .get("approx_tok_per_sec")
                    .and_then(|v| v.as_f64())
                    .map(|v| format!("{v:.4}"))
                    .unwrap_or_default();
                response_chars = log
                    .get("response_chars")
                    .and_then(as_u64_flexible)
                    .map(|v| v.to_string())
                    .unwrap_or_default();

                let prompt_eval_ns = log
                    .get("ollama_metrics")
                    .and_then(|m| m.get("prompt_eval_duration"))
                    .and_then(as_u64_flexible);
                let eval_ns = log
                    .get("ollama_metrics")
                    .and_then(|m| m.get("eval_duration"))
                    .and_then(as_u64_flexible);
                let prompt_count = log
                    .get("ollama_metrics")
                    .and_then(|m| m.get("prompt_eval_count"))
                    .and_then(as_u64_flexible);
                let eval_count_num = log
                    .get("ollama_metrics")
                    .and_then(|m| m.get("eval_count"))
                    .and_then(as_u64_flexible);

                if let Some(v) = prompt_count {
                    prompt_eval_count = v.to_string();
                }
                if let Some(v) = eval_count_num {
                    eval_count = v.to_string();
                }
                if let Some(v) = prompt_eval_ns {
                    prompt_eval_ms = format!("{:.3}", (v as f64) / 1_000_000.0);
                }
                if let Some(v) = eval_ns {
                    eval_ms = format!("{:.3}", (v as f64) / 1_000_000.0);
                }
                if let (Some(count), Some(duration_ns)) = (eval_count_num, eval_ns) {
                    if duration_ns > 0 {
                        decode_tok_s_proxy = format!(
                            "{:.4}",
                            count as f64 / (duration_ns as f64 / 1_000_000_000.0)
                        );
                    }
                }
                if let (Some(p), Some(e)) = (prompt_eval_ns, eval_ns) {
                    if e > 0 {
                        prefill_decode_ratio = format!("{:.4}", p as f64 / e as f64);
                    }
                }
                phase_sig = phase_signature(prompt_eval_ns, eval_ns);

                keep_alive_obs_ok = match log
                    .get("keep_alive_observability_min_ok")
                    .and_then(|v| v.as_bool())
                {
                    Some(true) => "true".to_string(),
                    Some(false) => "false".to_string(),
                    None => String::new(),
                };

                if let Some(err) = log.get("error").and_then(|v| v.as_str()) {
                    if !err.is_empty() {
                        rc = 1;
                        error_text = err.to_string();
                    }
                }
            }
            Err(e) => {
                rc = 1;
                error_text = e;
            }
        }

        if response.starts_with("[error]") {
            rc = 1;
            error_text = response;
        }

        let row = vec![
            ts_unix,
            mode.to_string(),
            run_config.model_name.clone(),
            preset_effective,
            requested_preset_name.to_string(),
            run_config
                .num_thread
                .map(|v| v.to_string())
                .unwrap_or_else(|| "none".to_string()),
            run_config
                .keep_alive
                .clone()
                .unwrap_or_else(|| "none".to_string()),
            rep_idx.to_string(),
            ttft_ms,
            total_ms,
            tok_s,
            response_chars,
            keep_alive_obs_ok,
            rc.to_string(),
            safe_tsv(&error_text),
            max_tokens,
            prompt_eval_count,
            prompt_eval_ms,
            eval_count,
            eval_ms,
            decode_tok_s_proxy,
            prefill_decode_ratio,
            phase_sig,
        ];
        write_tsv_row(out_file, &row)?;
    }

    Ok(())
}

async fn run_benchmark(base_config: &Config, cli: &CliArgs) -> Result<(), String> {
    let bench_mode = cli
        .bench_mode
        .ok_or_else(|| "internal error: bench_mode is not set".to_string())?;
    let prompt = cli
        .prompt
        .clone()
        .unwrap_or_else(|| "short test".to_string());
    let repeat = cli.repeat;

    let out_path = cli.bench_out.clone().unwrap_or_else(|| {
        format!(
            "worklog/bench_{}_{}.tsv",
            bench_mode.as_str(),
            current_unix_secs()
        )
    });
    let out_parent = Path::new(&out_path)
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| Path::new(".").to_path_buf());
    fs::create_dir_all(&out_parent)
        .map_err(|e| format!("ベンチ出力先ディレクトリ作成失敗 ({:?}): {e}", out_parent))?;

    let mut out_file = fs::File::create(&out_path)
        .map_err(|e| format!("ベンチ出力ファイル作成失敗 ({out_path}): {e}"))?;

    writeln!(
        out_file,
        "ts_unix\tmode\tmodel\tpreset_effective\trequested_preset\tnum_thread\tkeep_alive\trepeat_idx\tttft_ms\ttotal_ms\ttok_s\tresponse_chars\tkeep_alive_observability_min_ok\trc\terror\tmax_tokens\tprompt_eval_count\tprompt_eval_ms\teval_count\teval_ms\tdecode_tok_s_proxy\tprefill_decode_ratio\tphase_signature"
    )
    .map_err(|e| format!("ベンチヘッダ書き込み失敗: {e}"))?;

    let base_preset = cli.preset.clone().unwrap_or_else(|| base_config.preset());
    let threads_csv = cli.bench_threads_csv.as_deref().unwrap_or("2,4,6");
    let thread_set = parse_threads_csv(threads_csv)?;
    let keep_alive_csv = cli.bench_keep_alive_csv.as_deref().unwrap_or("10s,30s,5m");
    let keep_alive_set = parse_csv_tokens(keep_alive_csv);
    if keep_alive_set.is_empty() {
        return Err("--keep-alive-values must not be empty".to_string());
    }
    let predict_values_csv = cli
        .bench_predict_values_csv
        .as_deref()
        .unwrap_or("64,128,256,512,1024");
    let predict_values_set = parse_predict_values_csv(predict_values_csv)?;

    let preset_cases = vec![
        Preset::Gfx900Safe,
        Preset::Gfx900Balanced,
        Preset::Gfx900Longctx,
        Preset::Gfx900Tinybench,
        Preset::Gfx900AnchorBaseline,
        Preset::Gfx900AnchorSide1024,
    ];

    if matches!(bench_mode, BenchMode::PresetSweep | BenchMode::All) {
        for preset in &preset_cases {
            run_bench_case(
                base_config,
                "preset-sweep",
                &prompt,
                repeat,
                Some(preset.clone()),
                preset_name(preset),
                None,
                None,
                None,
                &mut out_file,
            )
            .await?;
        }
    }

    if matches!(bench_mode, BenchMode::ThreadSweep | BenchMode::All) {
        for t in &thread_set {
            run_bench_case(
                base_config,
                "thread-sweep",
                &prompt,
                repeat,
                Some(base_preset.clone()),
                preset_name(&base_preset),
                Some(*t),
                None,
                None,
                &mut out_file,
            )
            .await?;
        }
    }

    if matches!(bench_mode, BenchMode::KeepaliveSweep | BenchMode::All) {
        for keep_alive in &keep_alive_set {
            run_bench_case(
                base_config,
                "keepalive-sweep",
                &prompt,
                repeat,
                Some(base_preset.clone()),
                preset_name(&base_preset),
                None,
                Some(keep_alive),
                None,
                &mut out_file,
            )
            .await?;
        }
    }

    if matches!(bench_mode, BenchMode::PredictSweep | BenchMode::All) {
        for predict in &predict_values_set {
            run_bench_case(
                base_config,
                "predict-sweep",
                &prompt,
                repeat,
                Some(base_preset.clone()),
                preset_name(&base_preset),
                None,
                None,
                Some(*predict),
                &mut out_file,
            )
            .await?;
        }
    }

    println!(
        "[bench] mode={} repeat={} out={}",
        bench_mode.as_str(),
        repeat,
        out_path
    );
    let phase_summary_path = match write_bench_phase_summary(&out_path) {
        Ok(path) => {
            println!("[bench] phase_summary={path}");
            Some(path)
        }
        Err(e) => {
            eprintln!("[bench-warn] phase summary generation failed: {e}");
            None
        }
    };
    if let Err(e) =
        append_bench_worklog_summary(&out_path, bench_mode, repeat, phase_summary_path.as_deref())
    {
        eprintln!("[bench-warn] worklog summary append failed: {e}");
    }
    Ok(())
}

#[tokio::main]
async fn main() {
    let cli = match parse_cli_args() {
        Ok(cli) => cli,
        Err(e) => {
            eprintln!("{e}");
            eprintln!("Use --help for usage.");
            return;
        }
    };

    let mut config = load_config(&cli.config_path);
    apply_cli_overrides(&mut config, &cli);

    if let Some(baseline_path) = cli.bench_compare_baseline.as_deref() {
        let side_path = match cli.bench_compare_side.as_deref() {
            Some(v) => v,
            None => {
                eprintln!(
                    "[bench-compare-error] --compare-side is required when --bench-compare is used"
                );
                return;
            }
        };
        match write_bench_compare_summary(
            baseline_path,
            side_path,
            cli.bench_compare_out.as_deref(),
        ) {
            Ok(path) => println!("[bench-compare] out={path}"),
            Err(e) => eprintln!("[bench-compare-error] {e}"),
        }
        return;
    }

    if let Some(input_path) = cli.bench_report_input.as_deref() {
        match write_bench_mode_summary(
            input_path,
            cli.bench_report_out.as_deref(),
            cli.bench_report_format,
        ) {
            Ok(paths) => {
                for path in paths {
                    println!("[bench-report] out={path}");
                }
            }
            Err(e) => eprintln!("[bench-report-error] {e}"),
        }
        return;
    }

    if cli.bench_mode.is_some() {
        if let Err(e) = run_benchmark(&config, &cli).await {
            eprintln!("[bench-error] {e}");
        }
        return;
    }

    let app = match App::new(config.clone()) {
        Ok(app) => app,
        Err(e) => {
            eprintln!("{e}");
            return;
        }
    };

    if !cli.quiet {
        print_effective_config(&config, &app.effective);
        if let Some(false) = keep_alive_observability_min_ok(app.effective.keep_alive.as_deref()) {
            eprintln!(
                "[warn] keep_alive={:?} may make stream+rocprof phase windows unstable; prefer keep_alive>=10s",
                app.effective.keep_alive
            );
        }
    }

    if let Some(prompt) = cli.prompt.as_deref() {
        for i in 0..cli.repeat {
            if cli.repeat > 1 && !cli.quiet {
                println!("[run {}/{}]", i + 1, cli.repeat);
            }

            let stream_inline = config.should_stream_inline() && !cli.quiet;
            if stream_inline {
                print!("AI > ");
                let _ = io::stdout().flush();
            }

            let response = app.infer(prompt).await;

            if stream_inline {
                if response.starts_with("[error]") {
                    println!("{response}");
                } else {
                    println!();
                }
            } else if !cli.quiet || response.starts_with("[error]") {
                println!("AI > {response}");
            }
        }
        return;
    }

    println!("チャットクライアントを開始します（`/bye` または空行で終了）");

    loop {
        print!("You > ");
        let _ = io::stdout().flush();

        let mut prompt = String::new();
        if io::stdin().read_line(&mut prompt).is_err() {
            println!("入力エラー");
            break;
        }

        let prompt = prompt.trim();
        if prompt.is_empty() || prompt == "/bye" {
            println!("バイバイ！またね！");
            break;
        }

        let stream_inline = config.should_stream_inline();
        if stream_inline {
            print!("AI > ");
            let _ = io::stdout().flush();
        }

        let response = app.infer(prompt).await;

        if stream_inline {
            if response.starts_with("[error]") {
                println!("{response}");
            } else {
                println!();
            }
        } else {
            println!("AI > {response}");
        }
    }
}
