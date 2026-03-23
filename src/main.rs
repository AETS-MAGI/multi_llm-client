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
         \t--quiet                     Suppress banner and response echo in one-shot mode\n\
         \t-h, --help                  Show this help\n",
        Preset::all_names().join(", ")
    );
}

fn parse_bool_arg(value: &str) -> Result<bool, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Ok(true),
        "false" | "0" | "no" | "off" => Ok(false),
        _ => Err(format!("invalid bool value: {value}")),
    }
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

    let app = match App::new(config.clone()) {
        Ok(app) => app,
        Err(e) => {
            eprintln!("{e}");
            return;
        }
    };

    if !cli.quiet {
        print_effective_config(&config, &app.effective);
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
