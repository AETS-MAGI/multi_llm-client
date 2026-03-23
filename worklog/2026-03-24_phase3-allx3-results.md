# 2026-03-24 Phase3 all x3 Results

## Run

- command: `scripts/phase3_bench.sh all --repeat 3 --prompt 'short test' --out worklog/bench_phase3_all_20260324.tsv`
- model: `tinyllama:latest`
- output TSV: `worklog/bench_phase3_all_20260324.tsv`

## Steady summary (repeat_idx=2,3)

| condition | steady_ttft_ms | steady_tok/s |
|---|---:|---:|
| preset `gfx900_safe` | 103.0 | 207.47 |
| preset `gfx900_balanced` | 89.0 | 214.93 |
| preset `gfx900_longctx` | 84.5 | 215.28 |
| thread `2` (`safe`) | 104.0 | 202.91 |
| thread `4` (`safe`) | 88.5 | 208.41 |
| thread `6` (`safe`) | 128.5 | 216.37 |
| keep_alive `0s` (`safe`) | 1239.5 | 209.04 |
| keep_alive `10m` (`safe`) | 127.0 | 215.82 |

## Notes

- `keep_alive=10m` は TTFT に強い効果がある（`0s` 比で大幅改善）。
- `num_thread=6` は tok/s 最速だが TTFT が悪化。
- 研究基準として `gfx900_safe` 固定は維持し、`balanced/longctx/thread=6` は別線で運用最適化候補にする。
