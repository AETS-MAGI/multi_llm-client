# multi_llm-client (AETS-MAGI fork)

Japanese version: [README.ja.md](./README.ja.md)

Lightweight Rust CLI client for local LLM inference.
This fork is focused on reproducible experiments with Ollama, especially on ROCm `gfx900` (Radeon Instinct MI25).

## Overview

`multi_llm-client` sends prompts to a local or compatible inference endpoint and provides:

- Streaming and non-streaming response handling
- Preset-based runtime tuning for `gfx900`
- Effective configuration printout at startup
- Basic inference metrics (`TTFT`, `total_ms`, `tok/s` when available)
- JSONL logging for repeatable benchmark workflows

## What This Fork Adds

Compared to the original early version, this fork adds:

- Hardened HTTP error handling (status checks for Ollama/OpenAI-compatible endpoints)
- Stable parsing path for `stream=false` responses
- `preset`-based parameter control (`default`, `gfx900_safe`, etc.)
- Backward compatibility with legacy `gfx900_preset`
- `python_command` support (default `python3`, fallback to `python`)
- Daily JSONL inference logs in configurable `log_dir`

## Requirements

- Linux
- Rust (recent stable via `rustup`)
- Ollama running locally (`http://127.0.0.1:11434` by default)
- Optional: ROCm + `gfx900` GPU for MI25 experiments

## Quick Start

1. Ensure Rust toolchain is available from rustup.

```bash
export PATH="$HOME/.cargo/bin:$PATH"
cargo --version
```

2. Configure `config.json`.

3. Run the client.

```bash
cargo run
```

4. In the prompt, type text and press Enter. Use `/bye` or empty line to exit.

## Example `config.json`

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
  "keep_alive": "10m",
  "request_timeout_secs": 300,
  "connect_timeout_secs": 5,
  "preset": "gfx900_safe",
  "python_command": "python3",
  "log_dir": "logs"
}
```

## Configuration Reference

| Key | Type | Default | Description |
|---|---|---|---|
| `model_name` | string | required | Model name for backend |
| `endpoint` | string or null | `http://127.0.0.1:11434/api/generate` | Target endpoint |
| `use_local_model` | bool | `true` | Use local backend path |
| `local_framework` | `ollama`/`python` | `ollama` | Local backend selection |
| `openai_compatible` | bool | `false` | Use OpenAI-compatible request/response path |
| `max_tokens` | number or null | preset-based | `num_predict` equivalent |
| `api_key` | string or null | `null` | Bearer token when needed |
| `stream` | bool | `true` | Transport mode |
| `inline_stream` | bool | `true` | Render streaming tokens inline in CLI |
| `temperature` | number or null | `null` | Sampling temperature |
| `num_ctx` | number or null | preset-based | Context size override |
| `num_batch` | number or null | preset-based | Batch size override |
| `keep_alive` | string or null | `10m` | Ollama keep-alive |
| `request_timeout_secs` | number | `300` | Request timeout |
| `connect_timeout_secs` | number | `5` | Connect timeout |
| `preset` | string | `default` | Runtime preset |
| `gfx900_preset` | bool | `false` | Legacy compatibility key |
| `python_command` | string | `python3` | Python executable |
| `log_dir` | string | `logs` | JSONL output directory |

## Presets

If `max_tokens`, `num_ctx`, `num_batch` are not explicitly set, values are derived from `preset`.

| Preset | max_tokens | num_ctx | num_batch |
|---|---:|---:|---:|
| `default` | 256 | null | null |
| `gfx900_safe` | 128 | 4096 | 256 |
| `gfx900_balanced` | 192 | 4096 | 512 |
| `gfx900_longctx` | 128 | 8192 | 128 |
| `gfx900_tinybench` | 32 | 2048 | 64 |

## Runtime Outputs

At startup, the client prints effective settings (model, preset, stream mode, resolved limits, timeouts).

Each inference prints a metrics line such as:

```text
[stats] ttft=152ms total=3120ms output_chars=384 eval_count=96 tok/s=28.40
```

Logs are stored as JSONL:

- Path: `log_dir/infer-YYYY-MM-DD.jsonl`
- One record per inference
- Includes prompt/response lengths, timing, effective config, and Ollama final metrics

## Recommended gfx900 Benchmark Flow

1. Fix prompt/template and model.
2. Use `temperature: 0.0` for stable comparisons.
3. Keep preset fixed, then vary one parameter at a time (`num_batch`, `num_ctx`, `max_tokens`).
4. Compare TTFT/total/tok/s from JSONL logs.
5. Record external GPU telemetry (`rocm-smi`, `nvtop`) in parallel.

## Troubleshooting

### `Cargo.lock` version error on older cargo

If you see lockfile version errors, your shell may still resolve `/usr/bin/cargo` first.

```bash
export PATH="$HOME/.cargo/bin:$PATH"
cargo --version
```

### HTTP 404 / model not found

Check model availability in Ollama:

```bash
ollama list
```

### Connection refused

Ensure Ollama is running and endpoint matches `config.json`:

```bash
curl http://127.0.0.1:11434/api/tags
```

### Python backend command not found

Set `python_command` explicitly, e.g. `python3`.

## Project Scope

This repository is currently focused on practical local inference and tuning workflows.
It is not a full-featured chat platform; it is a lightweight experimentation client.

## License

No explicit license file is currently included in this repository snapshot.
Set a repository license before broad external redistribution.
