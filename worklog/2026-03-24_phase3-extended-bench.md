# 2026-03-24 Phase3 Extended Bench

## Runs

- `preset-sweep` (`tinyllama:latest`, repeat=10)
  - `worklog/bench_preset_tinyllama_n10_20260324.tsv`
- `thread-sweep` (`tinyllama:latest`, threads=4,6, repeat=20)
  - `worklog/bench_thread46_tinyllama_n20_20260324.tsv`
- `thread-sweep` (`qwen2.5:7b`, threads=4,6, repeat=10)
  - `worklog/bench_thread46_qwen7b_n10_20260324.tsv`

## Summary (steady: rep>=2)

### tinyllama preset sweep

| preset | n | steady_ttft_ms | steady_tok/s |
|---|---:|---:|---:|
| gfx900_safe | 9 | 112.44 | 215.87 |
| gfx900_balanced | 9 | 94.44 | 216.46 |
| gfx900_longctx | 9 | 103.67 | 213.37 |
| gfx900_tinybench | 9 | 95.67 | 202.52 |

### tinyllama thread 4/6 (repeat=20)

| num_thread | n | steady_ttft_ms | steady_tok/s | tok/s range |
|---|---:|---:|---:|---:|
| 4 | 19 | 99.95 | 213.65 | 202.00 - 217.18 |
| 6 | 19 | 96.63 | 217.70 | 203.38 - 221.23 |

### qwen2.5:7b thread 4/6 (repeat=10)

| num_thread | n | steady_ttft_ms | steady_tok/s | tok/s range |
|---|---:|---:|---:|---:|
| 4 | 9 | 296.33 | 49.36 | 49.11 - 49.53 |
| 6 | 9 | 292.00 | 48.65 | 46.32 - 49.52 |

## Interpretation

- tinyllama では `balanced` が `safe` 比でわずかに優位（TTFT/tok/s とも）。
- tinyllama のみを見ると `num_thread=6` は有利だが、qwen2.5:7b では `num_thread=4` の方が tok/s が安定して高い。
- クロスモデルの既定値としては `num_thread=4` が保守的に妥当。

## Recommended defaults (cross-model baseline)

- preset: `gfx900_safe`（研究基準固定）
- keep_alive: `10m`
- num_thread: `4`

運用で tinyllama 専用の吞吐優先を狙う場合のみ `num_thread=6` を別プロファイル化する。
