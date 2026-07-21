# 2026-03-25 anchor compare wrapper

## Summary

Added `scripts/anchor_compare.sh` to standardize the canonical anchor flow:

1. baseline bench (`gfx900_anchor_baseline`)
2. side bench (`gfx900_anchor_side1024`)
3. phase-summary compare (`--bench-compare`)
4. mode reports for both baseline/side (`--bench-report`)

The goal is to reduce manual command drift and keep baseline/side artifact naming consistent.

## Added file

- `scripts/anchor_compare.sh`

## Default behavior

- model: `gpt-oss:latest`
- mode: `predict-sweep`
- baseline preset: `gfx900_anchor_baseline`
- side preset: `gfx900_anchor_side1024`
- predict values: `64,128`
- report format: `all` (`.tsv/.md/.json`)

## Example

```bash
scripts/anchor_compare.sh \
  --model gpt-oss:latest \
  --repeat 1 \
  --predict-values 64,128 \
  --report-format all
```

## Output prefix

By default artifacts are created under:

- `worklog/anchor_compare_<timestamp>_*`

Users can override via `--prefix`.
