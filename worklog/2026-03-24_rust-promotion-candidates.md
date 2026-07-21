# 2026-03-24 Rust Promotion Candidates (Shell -> Rust)

## Context

- [main-node confirmed] stream+rocprof 観測で `keep_alive` が短すぎると phase 証跡が不安定化。
- [main-node confirmed] `keep_alive>=10s` で baseline/side の両レーンで安定。
- [ROCm-vega historical + main-node confirmed] anchor 条件（baseline512 / side1024）が継続利用される。

## Promotion Rule (operational)

Rust 側へ昇格する条件:

1. 同一判定が複数回再現される（最低3回、可能なら cross-batch）
2. 指標が比較可能な形で揃う（TTFT / total / gate / phase）
3. 日常運用で繰り返し使うフローである

## Candidate Backlog

### P0 (promoted in this update)

- `keep_alive` observability guard
  - 起動時 warning: `keep_alive < 10s`
  - JSONL へ `keep_alive_observability_min_ok` を追加
  - 理由: 実測で `0s`/`1s` が再現的に unavailable
- Anchor presets
  - `gfx900_anchor_baseline` (`8192/512/128`)
  - `gfx900_anchor_side1024` (`8192/1024/128`)
  - 理由: shell probe の canonical lane を Rust 既定へ寄せる
- phase3 keepalive defaults
  - `scripts/phase3_bench.sh` の既定を `10s,30s,5m` へ変更

### P1 (next promotion candidates)

- `bench` サブコマンド（one-shot loop + TSV）
  - 既存 `phase3_bench.sh` の主要機能を Rust へ取り込む
  - 目標: shell依存を減らし、同一フォーマットを固定
- `anchor lane` 明示フラグ
  - `--lane baseline|side` で preset + recommended keep_alive を自動選択

### P2 (research helper candidates)

- phase summary parser（外部 summary TSV取り込み）
  - `g4_stream_*` / `g4_link_*` のテーブルを Rust で読み込み・比較
- gate health report
  - `direct/fallback/dispatch/phase` を簡易ステータス化して出力

## Local branch note

Rust 側を採用状態へ寄せたら、ローカル `main` は次で上書きする想定:

```bash
git checkout feature/gfx900-fast-client
git branch -f main HEAD
```

（リモート更新は別途判断）
