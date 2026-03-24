#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

usage() {
  cat <<'USAGE'
Usage:
  scripts/anchor_compare.sh [options]

Purpose:
  Run canonical baseline/side anchor flow and generate compare/report artifacts:
    baseline bench -> side bench -> phase-summary compare -> mode reports

Options:
  --model <name>            Model name override (default: gpt-oss:latest)
  --prompt <text>           Prompt text (default: short test)
  --repeat <n>              Repeat count per case (default: 1)
  --predict-values <csv>    Predict sweep values (default: 64,128)
  --baseline-preset <name>  Baseline preset (default: gfx900_anchor_baseline)
  --side-preset <name>      Side preset (default: gfx900_anchor_side1024)
  --prefix <path>           Artifact prefix (default: worklog/anchor_compare_<timestamp>)
  --report-format <fmt>     bench-report format (tsv|markdown|json|all, default: all)

Environment variables (optional):
  MODEL, PROMPT, REPEAT, PREDICT_VALUES, BASELINE_PRESET, SIDE_PRESET, PREFIX, REPORT_FORMAT
USAGE
}

MODEL="${MODEL:-gpt-oss:latest}"
PROMPT="${PROMPT:-short test}"
REPEAT="${REPEAT:-1}"
PREDICT_VALUES_CSV="${PREDICT_VALUES:-64,128}"
BASELINE_PRESET="${BASELINE_PRESET:-gfx900_anchor_baseline}"
SIDE_PRESET="${SIDE_PRESET:-gfx900_anchor_side1024}"
PREFIX="${PREFIX:-worklog/anchor_compare_$(date +%Y%m%d_%H%M%S)}"
REPORT_FORMAT="${REPORT_FORMAT:-all}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --prompt)
      PROMPT="$2"
      shift 2
      ;;
    --repeat)
      REPEAT="$2"
      shift 2
      ;;
    --predict-values)
      PREDICT_VALUES_CSV="$2"
      shift 2
      ;;
    --baseline-preset)
      BASELINE_PRESET="$2"
      shift 2
      ;;
    --side-preset)
      SIDE_PRESET="$2"
      shift 2
      ;;
    --prefix)
      PREFIX="$2"
      shift 2
      ;;
    --report-format)
      REPORT_FORMAT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[error] unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if ! [[ "$REPEAT" =~ ^[0-9]+$ ]] || [[ "$REPEAT" -lt 1 ]]; then
  echo "[error] --repeat must be >= 1" >&2
  exit 1
fi

case "$REPORT_FORMAT" in
  tsv|markdown|json|all)
    ;;
  *)
    echo "[error] --report-format must be one of: tsv|markdown|json|all" >&2
    exit 1
    ;;
esac

resolve_client_bin() {
  if [[ -x "$ROOT/target/release/multi_llm_client" ]]; then
    echo "$ROOT/target/release/multi_llm_client"
    return
  fi
  if [[ -x "$ROOT/target/debug/multi_llm_client" ]]; then
    echo "$ROOT/target/debug/multi_llm_client"
    return
  fi

  echo "[info] client binary not found. building debug binary..." >&2
  cargo build --quiet
  echo "$ROOT/target/debug/multi_llm_client"
}

CLIENT_BIN="$(resolve_client_bin)"

BASE_OUT="${PREFIX}_baseline.tsv"
SIDE_OUT="${PREFIX}_side.tsv"
COMPARE_OUT="${PREFIX}_baseline_vs_side.tsv"
BASE_REPORT="${PREFIX}_baseline_mode_summary.tsv"
SIDE_REPORT="${PREFIX}_side_mode_summary.tsv"

echo "[info] baseline: preset=${BASELINE_PRESET}"
"$CLIENT_BIN" \
  --bench predict-sweep \
  --model "$MODEL" \
  --prompt "$PROMPT" \
  --repeat "$REPEAT" \
  --preset "$BASELINE_PRESET" \
  --predict-values "$PREDICT_VALUES_CSV" \
  --out "$BASE_OUT"

echo "[info] side: preset=${SIDE_PRESET}"
"$CLIENT_BIN" \
  --bench predict-sweep \
  --model "$MODEL" \
  --prompt "$PROMPT" \
  --repeat "$REPEAT" \
  --preset "$SIDE_PRESET" \
  --predict-values "$PREDICT_VALUES_CSV" \
  --out "$SIDE_OUT"

BASE_PHASE="${BASE_OUT%.tsv}_phase_summary.tsv"
SIDE_PHASE="${SIDE_OUT%.tsv}_phase_summary.tsv"

echo "[info] compare: ${BASE_PHASE} vs ${SIDE_PHASE}"
"$CLIENT_BIN" \
  --bench-compare "$BASE_PHASE" \
  --compare-side "$SIDE_PHASE" \
  --compare-out "$COMPARE_OUT"

echo "[info] report: baseline (${REPORT_FORMAT})"
"$CLIENT_BIN" \
  --bench-report "$BASE_OUT" \
  --report-format "$REPORT_FORMAT" \
  --report-out "$BASE_REPORT"

echo "[info] report: side (${REPORT_FORMAT})"
"$CLIENT_BIN" \
  --bench-report "$SIDE_OUT" \
  --report-format "$REPORT_FORMAT" \
  --report-out "$SIDE_REPORT"

cat <<EOF
[done] anchor comparison completed
  baseline_tsv=${BASE_OUT}
  baseline_phase=${BASE_PHASE}
  side_tsv=${SIDE_OUT}
  side_phase=${SIDE_PHASE}
  compare_tsv=${COMPARE_OUT}
  baseline_report=${BASE_REPORT}
  side_report=${SIDE_REPORT}
EOF
