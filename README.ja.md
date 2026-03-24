# multi_llm-client (AETS-MAGI fork)

English version: [README.MD](./README.MD)

ローカル LLM 推論向けの軽量 Rust CLI クライアントです。  
このフォークは、Ollama を使った再現性のある実験、特に ROCm `gfx900`（Radeon Instinct MI25）での検証に焦点を当てています。

## 概要

`multi_llm-client` はローカルまたは互換エンドポイントにプロンプトを送信し、以下を提供します。

- ストリーミング / 非ストリーミング応答処理
- `gfx900` 向け preset ベースのランタイム調整
- 起動時の effective 設定表示
- 推論メトリクス（`TTFT`, `total_ms`, `tok/s` ※取得可能時）
- ベンチ比較向け JSONL ログ

## このフォークで追加したもの

- HTTP エラーハンドリング強化（Ollama / OpenAI互換とも status チェック）
- `stream=false` の安定パース経路
- `preset` ベースのパラメータ制御（`default`, `gfx900_safe` など）
- 旧キー `gfx900_preset` との後方互換
- `python_command` 対応（既定 `python3`、必要時 `python` へフォールバック）
- 日別 JSONL ログ（`log_dir`）

## 要件

- Linux
- Rust（`rustup` で入れた新しめの stable）
- ローカル Ollama（既定: `http://127.0.0.1:11434`）
- 任意: ROCm + `gfx900` GPU（MI25 実験向け）

## クイックスタート

1. `rustup` 側の cargo が使われていることを確認

```bash
export PATH="$HOME/.cargo/bin:$PATH"
cargo --version
```

2. `config.json` を設定

3. 実行

```bash
cargo run
```

4. プロンプト入力。`/bye` または空行で終了

## `config.json` 例

```json
{
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
}
```

## 設定キー

| キー | 型 | 既定値 | 説明 |
|---|---|---|---|
| `model_name` | string | 必須 | モデル名 |
| `endpoint` | string or null | `http://127.0.0.1:11434/api/generate` | 接続先 |
| `use_local_model` | bool | `true` | ローカル経路を使用 |
| `local_framework` | `ollama`/`python` | `ollama` | ローカル推論バックエンド |
| `openai_compatible` | bool | `false` | OpenAI互換経路を使用 |
| `max_tokens` | number or null | preset依存 | `num_predict` 相当 |
| `api_key` | string or null | `null` | Bearer トークン |
| `stream` | bool | `true` | 通信方式 |
| `inline_stream` | bool | `true` | CLI逐次表示 |
| `temperature` | number or null | `null` | サンプリング温度 |
| `num_ctx` | number or null | preset依存 | コンテキスト長 |
| `num_batch` | number or null | preset依存 | バッチサイズ |
| `num_thread` | number or null | `null` | Ollama `options.num_thread` 上書き |
| `keep_alive` | string or null | `10m` | Ollama keep-alive |
| `request_timeout_secs` | number | `300` | リクエストタイムアウト |
| `connect_timeout_secs` | number | `5` | 接続タイムアウト |
| `preset` | string | `default` | ランタイム preset |
| `gfx900_preset` | bool | `false` | 旧互換キー |
| `python_command` | string | `python3` | Python 実行コマンド |
| `log_dir` | string | `logs` | JSONL ログ保存先 |

## Preset

`max_tokens` / `num_ctx` / `num_batch` を未指定にすると `preset` から決定されます。

| Preset | max_tokens | num_ctx | num_batch |
|---|---:|---:|---:|
| `default` | 256 | null | null |
| `gfx900_safe` | 128 | 4096 | 256 |
| `gfx900_balanced` | 192 | 4096 | 512 |
| `gfx900_longctx` | 128 | 8192 | 128 |
| `gfx900_tinybench` | 32 | 2048 | 64 |
| `gfx900_anchor_baseline` | 128 | 8192 | 512 |
| `gfx900_anchor_side1024` | 128 | 8192 | 1024 |

観測メモ:

- stream+rocprof の phase-window 実験では、`keep_alive` が 10 秒未満だと
  dispatch/phase 証跡が不安定になる場合があります。
- クライアントは effective `keep_alive < 10s` のとき警告を表示します。

## 実行時出力

起動時に effective 設定（モデル、preset、stream設定、実効パラメータ、timeout）を表示します。  
推論ごとに以下のようなメトリクスを表示します。

```text
[stats] ttft=152ms total=3120ms output_chars=384 eval_count=96 tok/s=28.40
```

ログは JSONL 形式で保存されます。

- パス: `log_dir/infer-YYYY-MM-DD.jsonl`
- 1推論につき1行
- prompt/response長、時間、effective設定、Ollama最終メトリクスを含む

## 非対話モード

スクリプト実行向けに one-shot モードを使えます。

```bash
./target/debug/multi_llm_client \
  --prompt "short test" \
  --preset gfx900_safe \
  --stream false \
  --inline-stream false \
  --num-thread 4 \
  --repeat 3
```

主な CLI 上書き:

- `--config <path>`
- `--prompt <text>`
- `--repeat <n>`
- `--preset <default|gfx900_safe|gfx900_balanced|gfx900_longctx|gfx900_tinybench|gfx900_anchor_baseline|gfx900_anchor_side1024>`
- `--model <model_name>`
- `--keep-alive <value|none>`
- `--num-thread <n|none>`
- `--stream <true|false>`
- `--inline-stream <true|false>`
- `--bench <preset-sweep|thread-sweep|keepalive-sweep|all>`
- `--out <path>`
- `--threads <csv>`
- `--keep-alive-values <csv>`
- `--quiet`

## Rust内蔵ベンチモード

シェルラッパーを使わず、Rust バイナリ単体で反復 sweep を実行できます。

```bash
cargo run -- --bench preset-sweep --repeat 3 --prompt "short test"
cargo run -- --bench thread-sweep --preset gfx900_safe --threads 2,4,6 --repeat 3
cargo run -- --bench keepalive-sweep --preset gfx900_anchor_baseline --keep-alive-values 10s,30s,5m --repeat 3
```

TSV 列:

- `ts_unix`, `mode`, `model`, `preset_effective`, `requested_preset`
- `num_thread`, `keep_alive`, `repeat_idx`
- `ttft_ms`, `total_ms`, `tok_s`, `response_chars`
- `keep_alive_observability_min_ok`, `rc`, `error`

出力先:

- 既定: `worklog/bench_<mode>_<unix_ts>.tsv`
- 上書き: `--out <path>`

## Phase3 自動ベンチスクリプト

`scripts/phase3_bench.sh` で TSV 出力付きの反復測定を実行できます。

- このスクリプトは Rust 内蔵 `--bench` の薄いラッパーです。

```bash
scripts/phase3_bench.sh preset-sweep --repeat 3 --prompt "short test"
scripts/phase3_bench.sh thread-sweep --repeat 3 --preset gfx900_safe --threads 2,4,6
scripts/phase3_bench.sh keepalive-sweep --repeat 3 --keep-alive-values 10s,30s,5m
```

既定の出力先: `worklog/bench_<mode>_YYYYmmdd_HHMMSS.tsv`

## gfx900 ベンチ推奨手順

1. プロンプトとモデルを固定
2. 比較時は `temperature: 0.0` に固定
3. preset を固定し、`num_batch` / `num_ctx` / `max_tokens` を1つずつ変更
4. JSONL の TTFT / total / tok/s を比較
5. `rocm-smi` / `nvtop` の外部計測も併記

## トラブルシュート

### 古い cargo で `Cargo.lock` エラー

```bash
export PATH="$HOME/.cargo/bin:$PATH"
cargo --version
```

### HTTP 404 / model not found

```bash
ollama list
```

### connection refused

```bash
curl http://127.0.0.1:11434/api/tags
```

### Python backend command not found

`python_command` を `python3` などに明示設定してください。

## スコープ

このリポジトリは、実運用前の軽量な推論・計測クライアントとしての利用を主目的にしています。  
フル機能のチャット基盤ではなく、再現実験向けツールです。

## ライセンス

現時点のリポジトリスナップショットには明示ライセンスファイルがありません。  
外部配布前にライセンスを明示してください。
