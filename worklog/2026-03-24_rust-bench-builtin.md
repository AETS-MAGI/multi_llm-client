# 2026-03-24 Rust built-in benchmark mode

## Context

Promoted recurring shell-side benchmark patterns into `multi_llm-client` itself to reduce drift between
exploratory scripts and client-native workflows.

## Implemented

- Added built-in benchmark mode in `src/main.rs`:
  - `--bench preset-sweep|thread-sweep|keepalive-sweep|all`
  - `--out <path>`
  - `--threads <csv>`
  - `--keep-alive-values <csv>`
- Added TSV writer with stable schema:
  - `ts_unix,mode,model,preset_effective,requested_preset,num_thread,keep_alive,repeat_idx,ttft_ms,total_ms,tok_s,response_chars,keep_alive_observability_min_ok,rc,error`
- Added `preset_name` mapping and CSV parsers for thread/keep_alive sweeps.
- Reused JSONL as source-of-truth for metrics extraction per run.

## Validation

- `cargo check` passed.
- `cargo run -- --help` shows new bench options.
- Smoke run passed:

```bash
cargo run -- --bench thread-sweep --threads 2 --repeat 1 --prompt "short test" --quiet --out worklog/bench_smoke.tsv
```

- Output file generated and populated:
  - `worklog/bench_smoke.tsv`

## Notes

- Bench mode intentionally forces `stream=false` and `inline_stream=false` for comparability.
- This is a promotion step; exploratory deep probes (`strace`, `rocprofv3`, phase-window slicing)
  remain shell-side for now.
