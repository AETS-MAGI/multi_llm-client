# 2026-03-24 Rust phase summary aggregation

## Summary

Completed the remaining phase-window style promotion in Rust bench workflow:

- Bench run now auto-generates grouped phase summary TSV:
  - `<out>_phase_summary.tsv`
- Grouping keys:
  - `mode`, `preset_effective`, `requested_preset`, `num_thread`, `keep_alive`, `max_tokens`, `phase_signature`
- Aggregated metrics:
  - `avg_ttft_ms`, `avg_total_ms`, `avg_tok_s`
  - `avg_prompt_eval_ms`, `avg_eval_ms`
  - `avg_decode_tok_s_proxy`, `avg_prefill_decode_ratio`

## Validation

```bash
cargo check
cargo run -- --bench predict-sweep --preset gfx900_anchor_baseline --predict-values 64 --repeat 1 --prompt "short test" --quiet --out worklog/bench_predict_phaseagg_smoke.tsv
```

Observed outputs:

- `worklog/bench_predict_phaseagg_smoke.tsv`
- `worklog/bench_predict_phaseagg_smoke_phase_summary.tsv`
- `worklog/bench_auto_summary_2026-03-24.md` appended with `phase_summary=` path.

## Notes

- This keeps shell probes for deep `strace/rocprofv3` exploration, while moving repeatable
  analysis shape into Rust-native benchmark tooling.
