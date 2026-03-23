# 2026-03-24 Phase3 n=10 Results

## Runs

- `thread-sweep` (`safe`, `num_thread=2/4/6`, `repeat=10`)
  - file: `worklog/bench_thread_tinyllama_n10_20260324.tsv`
- `keepalive-sweep` (`safe`, `0s/10m`, `repeat=10`, `tinyllama:latest`)
  - file: `worklog/bench_keepalive_tinyllama_n10_20260324.tsv`
- `keepalive-sweep` (`safe`, `0s/10m`, `repeat=10`, `qwen2.5:7b`)
  - file: `worklog/bench_keepalive_qwen7b_n10_20260324.tsv`

## Summary (all samples)

### tinyllama thread sweep

| num_thread | n | avg_ttft_ms | avg_tok/s |
|---|---:|---:|---:|
| 2 | 10 | 256.90 | 211.92 |
| 4 | 10 | 235.90 | 213.60 |
| 6 | 10 | 247.20 | 214.45 |

### tinyllama keep_alive sweep

| keep_alive | n | avg_ttft_ms | avg_tok/s |
|---|---:|---:|---:|
| 0s | 10 | 1270.40 | 203.08 |
| 10m | 10 | 238.80 | 212.65 |

### qwen2.5:7b keep_alive sweep

| keep_alive | n | avg_ttft_ms | avg_tok/s |
|---|---:|---:|---:|
| 0s | 10 | 3986.40 | 45.86 |
| 10m | 10 | 601.30 | 49.34 |

## Summary (steady: rep>=2)

### tinyllama thread sweep

| num_thread | n | steady_ttft_ms | steady_tok/s |
|---|---:|---:|---:|
| 2 | 9 | 90.89 | 213.61 |
| 4 | 9 | 100.00 | 214.24 |
| 6 | 9 | 105.11 | 215.26 |

### tinyllama keep_alive sweep

| keep_alive | n | steady_ttft_ms | steady_tok/s |
|---|---:|---:|---:|
| 0s | 9 | 1245.44 | 201.70 |
| 10m | 9 | 113.22 | 212.86 |

### qwen2.5:7b keep_alive sweep

| keep_alive | n | steady_ttft_ms | steady_tok/s |
|---|---:|---:|---:|
| 0s | 9 | 3413.89 | 46.91 |
| 10m | 9 | 284.33 | 49.47 |

## Interpretation

- `keep_alive=10m` は tinyllama / qwen2.5:7b の両方で TTFT を大幅改善。
- tok/s も `10m` 側が安定して高い。
- thread は `6` が tok/s 最速、`2` が TTFT 最短傾向、`4` は中庸。

## Recommended default (research baseline)

- preset: `gfx900_safe`
- keep_alive: `10m`
- num_thread: `4`（バランス重視）

運用吞吐を最優先する場合は `num_thread=6` を候補にする。
