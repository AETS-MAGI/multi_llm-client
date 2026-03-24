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
  "num_thread": null,
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
| `num_thread` | number or null | `null` | `options.num_thread` override for Ollama |
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
| `gfx900_anchor_baseline` | 128 | 8192 | 512 |
| `gfx900_anchor_side1024` | 128 | 8192 | 1024 |

Observability note:

- For stream+rocprof phase-window experiments, `keep_alive` below 10 seconds can
  make dispatch/phase evidence unstable.
- The client now emits a warning when effective `keep_alive < 10s`.

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

## Non-Interactive Mode

For scripted benchmarks, use one-shot mode:

```bash
./target/debug/multi_llm_client \
  --prompt "short test" \
  --preset gfx900_safe \
  --stream false \
  --inline-stream false \
  --num-thread 4 \
  --repeat 3
```

Supported CLI overrides:

- `--config <path>`
- `--prompt <text>`
- `--repeat <n>`
- `--preset <default|gfx900_safe|gfx900_balanced|gfx900_longctx|gfx900_tinybench>`
- `--preset <default|gfx900_safe|gfx900_balanced|gfx900_longctx|gfx900_tinybench|gfx900_anchor_baseline|gfx900_anchor_side1024>`
- `--model <model_name>`
- `--keep-alive <value|none>`
- `--num-thread <n|none>`
- `--stream <true|false>`
- `--inline-stream <true|false>`
- `--quiet`

## Phase3 Sweep Script

Use `scripts/phase3_bench.sh` to automate repeatable TSV output:

```bash
scripts/phase3_bench.sh preset-sweep --repeat 3 --prompt "short test"
scripts/phase3_bench.sh thread-sweep --repeat 3 --preset gfx900_safe --threads 2,4,6
scripts/phase3_bench.sh keepalive-sweep --repeat 3 --keep-alive-values 10s,30s,5m
```

Output TSV (default): `worklog/bench_<mode>_YYYYmmdd_HHMMSS.tsv`

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
