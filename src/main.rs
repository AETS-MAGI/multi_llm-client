use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::time::Duration;

use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::process::Command;

const DEFAULT_CONFIG: &str = r#"{
  "model_name": "tinyllama:latest",
  "endpoint": "http://127.0.0.1:11434/api/generate",
  "use_local_model": true,
  "local_framework": "ollama",
  "openai_compatible": false,
  "max_tokens": 128,
  "api_key": null,
  "stream": true,
  "temperature": 0.7,
  "num_ctx": null,
  "num_batch": null,
  "keep_alive": "10m",
  "request_timeout_secs": 300,
  "connect_timeout_secs": 5,
  "gfx900_preset": true
}"#;

#[derive(Debug, Deserialize, Clone)]
struct Config {
    model_name: String,
    endpoint: Option<String>,
    use_local_model: bool,
    local_framework: Option<String>,
    openai_compatible: bool,
    max_tokens: Option<u32>,
    api_key: Option<String>,

    stream: Option<bool>,
    temperature: Option<f32>,
    num_ctx: Option<u32>,
    num_batch: Option<u32>,
    keep_alive: Option<String>,
    request_timeout_secs: Option<u64>,
    connect_timeout_secs: Option<u64>,
    gfx900_preset: Option<bool>,
}

impl Config {
    fn stream_enabled(&self) -> bool {
        self.stream.unwrap_or(true)
    }

    fn endpoint_for_ollama(&self) -> String {
        self.endpoint
            .clone()
            .unwrap_or_else(|| "http://127.0.0.1:11434/api/generate".to_string())
    }

    fn framework(&self) -> &str {
        self.local_framework.as_deref().unwrap_or("ollama")
    }

    fn gfx900_enabled(&self) -> bool {
        self.gfx900_preset.unwrap_or(false)
    }

    fn effective_max_tokens(&self) -> u32 {
        if self.gfx900_enabled() {
            self.max_tokens.unwrap_or(128)
        } else {
            self.max_tokens.unwrap_or(256)
        }
    }

    fn effective_num_ctx(&self) -> Option<u32> {
        if self.gfx900_enabled() {
            Some(self.num_ctx.unwrap_or(4096))
        } else {
            self.num_ctx
        }
    }

    fn effective_num_batch(&self) -> Option<u32> {
        if self.gfx900_enabled() {
            Some(self.num_batch.unwrap_or(256))
        } else {
            self.num_batch
        }
    }

    fn connect_timeout(&self) -> Duration {
        Duration::from_secs(self.connect_timeout_secs.unwrap_or(5))
    }

    fn request_timeout(&self) -> Duration {
        Duration::from_secs(self.request_timeout_secs.unwrap_or(300))
    }

    fn should_stream_inline(&self) -> bool {
        self.stream_enabled() && !self.openai_compatible && self.framework() == "ollama"
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

struct App {
    config: Config,
    client: reqwest::Client,
}

impl App {
    fn new(config: Config) -> Result<Self, String> {
        let client = reqwest::Client::builder()
            .connect_timeout(config.connect_timeout())
            .timeout(config.request_timeout())
            .tcp_nodelay(true)
            .pool_max_idle_per_host(8)
            .pool_idle_timeout(Duration::from_secs(90))
            .build()
            .map_err(|e| format!("HTTPクライアント生成エラー: {e}"))?;
        Ok(Self { config, client })
    }

    async fn infer(&self, prompt: &str) -> String {
        if self.config.use_local_model {
            match self.config.framework() {
                "python" => self.python_inference(prompt).await,
                "ollama" => self.ollama_inference(prompt).await,
                other => format!("サポートされていないローカルフレームワークです: {other}"),
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
            return "Pythonスクリプトが見つかりません（./src/llm_interface.py または ./llm_interface.py）".to_string();
        };

        let output = Command::new("python")
            .arg(script_path)
            .arg(prompt)
            .output()
            .await;

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
        options.insert(
            "num_predict".to_string(),
            json!(self.config.effective_max_tokens()),
        );

        if let Some(temp) = self.config.temperature {
            options.insert("temperature".to_string(), json!(temp));
        }
        if let Some(num_ctx) = self.config.effective_num_ctx() {
            options.insert("num_ctx".to_string(), json!(num_ctx));
        }
        if let Some(num_batch) = self.config.effective_num_batch() {
            options.insert("num_batch".to_string(), json!(num_batch));
        }

        let mut root = serde_json::Map::new();
        root.insert("model".to_string(), json!(self.config.model_name));
        root.insert("prompt".to_string(), json!(prompt));
        root.insert("stream".to_string(), json!(self.config.stream_enabled()));
        root.insert("options".to_string(), Value::Object(options));

        if let Some(keep_alive) = &self.config.keep_alive {
            root.insert("keep_alive".to_string(), json!(keep_alive));
        }

        Value::Object(root)
    }

    async fn ollama_inference(&self, prompt: &str) -> String {
        let endpoint = self.config.endpoint_for_ollama();
        let request_body = self.build_ollama_request(prompt);
        let mut req = self.client.post(endpoint).json(&request_body);

        if let Some(api_key) = &self.config.api_key {
            req = req.bearer_auth(api_key);
        }

        let res = match req.send().await {
            Ok(response) => response,
            Err(e) => return format!("[error] Ollama推論エラー: {e}"),
        };

        if self.config.stream_enabled() {
            self.consume_ollama_stream(res, true).await
        } else {
            self.consume_ollama_non_stream(res).await
        }
    }

    async fn consume_ollama_non_stream(&self, response: reqwest::Response) -> String {
        let text = match response.text().await {
            Ok(text) => text,
            Err(e) => return format!("[error] レスポンス本文の取得に失敗: {e}"),
        };

        let mut collected = String::new();
        for line in text.lines() {
            if line.trim().is_empty() {
                continue;
            }
            match parse_ollama_line(line) {
                Ok(Some((token, _done))) => collected.push_str(&token),
                Ok(None) => {}
                Err(e) => return format!("[error] {e}"),
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
                    Ok(Some((token, _done))) => {
                        if !token.is_empty() {
                            if render_stdout {
                                print!("{token}");
                                let _ = io::stdout().flush();
                            }
                            collected.push_str(&token);
                        }
                    }
                    Ok(None) => {}
                    Err(e) => return format!("[error] {e}"),
                }
            }
        }

        // 改行終端なしで最後に1行残るケースを処理
        if !pending.trim().is_empty() {
            match parse_ollama_line(pending.trim()) {
                Ok(Some((token, _done))) => {
                    if !token.is_empty() {
                        if render_stdout {
                            print!("{token}");
                            let _ = io::stdout().flush();
                        }
                        collected.push_str(&token);
                    }
                }
                Ok(None) => {}
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

        let json_val: Value = match res.json().await {
            Ok(v) => v,
            Err(e) => return format!("[error] OpenAI互換レスポンスJSON解析エラー: {e}"),
        };

        // completion API / chat completion API どちらも拾えるようにする
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
}

fn parse_ollama_line(line: &str) -> Result<Option<(String, bool)>, String> {
    let val: Value =
        serde_json::from_str(line).map_err(|e| format!("Ollamaレスポンス行のJSON解析失敗: {e}"))?;
    if let Some(err) = val.get("error").and_then(|e| e.as_str()) {
        return Err(format!("Ollamaエラー: {err}"));
    }

    let token = val
        .get("response")
        .and_then(|r| r.as_str())
        .or_else(|| val.get("thinking").and_then(|t| t.as_str()))
        .unwrap_or("")
        .to_string();
    let done = val.get("done").and_then(|d| d.as_bool()).unwrap_or(false);
    Ok(Some((token, done)))
}

#[tokio::main]
async fn main() {
    let config_path = "config.json";
    let config = load_config(config_path);
    let app = match App::new(config.clone()) {
        Ok(app) => app,
        Err(e) => {
            eprintln!("{e}");
            return;
        }
    };

    println!("モデル: {}", config.model_name);
    println!(
        "モード: {} / framework={}",
        if config.use_local_model {
            "local"
        } else {
            "online"
        },
        config.framework()
    );
    println!(
        "gfx900 preset: {} / stream: {}",
        if config.gfx900_enabled() { "on" } else { "off" },
        if config.stream_enabled() { "on" } else { "off" }
    );
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
