# 2026-03-24 Phase3 Script Automation

## Summary

`safe` 基準を維持したまま、Phase3（preset/thread/keep_alive比較）を反復実行できるようにクライアントを拡張。

## Implemented

- `src/main.rs`
  - `config.json` に `num_thread`（optional）を追加
  - Ollama request `options` に `num_thread` を反映
  - one-shot CLI を追加
    - `--prompt`
    - `--repeat`
    - `--preset`
    - `--model`
    - `--keep-alive <value|none>`
    - `--num-thread <n|none>`
    - `--stream <true|false>`
    - `--inline-stream <true|false>`
    - `--quiet`
- `scripts/phase3_bench.sh`
  - `preset-sweep`
  - `thread-sweep`
  - `keepalive-sweep`
  - `all`
  - TSV 出力と簡易平均サマリ

## Validation

- `cargo fmt`
- `cargo check`
- one-shot 実行確認
  - `./target/debug/multi_llm_client --prompt 'short test' --repeat 1 --preset gfx900_tinybench --stream false --inline-stream false --num-thread 4 --quiet`
- JSONL 反映確認
  - `effective.num_thread=4` を確認
- スクリプト smoke
  - `scripts/phase3_bench.sh preset-sweep --repeat 1 --prompt 'short test' --out worklog/bench_preset_smoke.tsv`
  - 4 preset 分の行が出力されることを確認

## Notes

- 既定方針は維持: `preset=gfx900_safe` を研究基準として固定。
- `balanced` は別線（吞吐優先）で評価可能。
- `fallback_confirmed` を崩さないため、まずはクライアント層の比較条件固定と自動化を優先。
