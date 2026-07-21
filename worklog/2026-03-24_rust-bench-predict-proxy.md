# 2026-03-24 Rust bench promotion (predict/proxy/worklog)

## Scope

Promoted next-stage benchmark workflow from shell exploration into `multi_llm-client` built-in bench mode.

## Added

- New built-in bench mode:
  - `predict-sweep`
- New CLI options:
  - `--predict-values <csv>` (default: `64,128,256,512,1024`)
- Extended TSV schema with prefill/decode proxy metrics:
  - `max_tokens`
  - `prompt_eval_count`, `prompt_eval_ms`
  - `eval_count`, `eval_ms`
  - `decode_tok_s_proxy`
  - `prefill_decode_ratio`
  - `phase_signature`
- Bench auto-summary append:
  - `worklog/bench_auto_summary_YYYY-MM-DD.md`

## Script alignment

- `scripts/phase3_bench.sh` now supports `predict-sweep` / `--predict-values`.
- Script remains a thin wrapper over Rust built-in `--bench`.

## Validation commands

```bash
cargo check
cargo run -- --bench predict-sweep --preset gfx900_anchor_baseline --predict-values 64,128 --repeat 1 --prompt "short test" --quiet --out worklog/bench_predict_smoke.tsv
cargo run -- --bench predict-sweep --preset gfx900_anchor_baseline --predict-values 64 --repeat 1 --prompt "short test" --quiet --out worklog/bench_predict_smoke2.tsv
./scripts/phase3_bench.sh predict-sweep --predict-values 64 --repeat 1 --prompt "short test" --out worklog/bench_predict_wrapper.tsv
```

## Validation notes

- `cargo check`: passed.
- Built-in `predict-sweep`: passed.
- Wrapper `predict-sweep`: passed.
- Confirmed header now emits `ttft_ms` correctly.
- Confirmed auto summary append line exists in `worklog/bench_auto_summary_2026-03-24.md`.

## Outcome

- Candidate #1 completed: `num_predict` expansion sweep is now native in Rust.
- Candidate #2 partially completed: prefill/decode proxy metrics are now emitted in bench TSV.
- Candidate #3 completed: bench run appends compact aggregate summary into worklog automatically.
