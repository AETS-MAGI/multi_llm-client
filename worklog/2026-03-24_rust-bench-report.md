# 2026-03-24 Rust bench-report (mode summary)

## Added

- New post-process command in `multi_llm-client`:
  - `--bench-report <input.tsv>`
  - optional `--report-out <path>`
- Generates grouped mode summary TSV without re-running inference.

## Default output

- Input: `worklog/bench_xxx.tsv`
- Output: `worklog/bench_xxx_mode_summary.tsv`

## Group keys

- `mode`
- `preset_effective`
- `requested_preset`
- `num_thread`
- `keep_alive`
- `max_tokens`

## Aggregates

- row counts:
  - `rows`, `ok_rows`
  - `prefill_decode_rows`, `decode_only_rows`, `prefill_only_rows`, `unavailable_rows`
- averages:
  - `avg_ttft_ms`, `avg_total_ms`, `avg_tok_s`
  - `avg_prompt_eval_ms`, `avg_eval_ms`
  - `avg_decode_tok_s_proxy`, `avg_prefill_decode_ratio`

## Validation

```bash
cargo check
cargo run -- --bench-report worklog/bench_predict_phaseagg_smoke.tsv --report-out worklog/bench_predict_phaseagg_mode_summary.tsv
```

## Observed

- `worklog/bench_predict_phaseagg_mode_summary.tsv` generated successfully.
- `--help` includes new options (`--bench-report`, `--report-out`).
