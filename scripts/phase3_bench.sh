#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

usage() {
  cat <<'USAGE'
Usage:
  scripts/phase3_bench.sh <mode> [options]

Modes:
  preset-sweep     Run repeated benchmark across gfx900 presets.
  thread-sweep     Run repeated benchmark across num_thread set (safe preset by default).
  keepalive-sweep  Run repeated benchmark across keep_alive values (safe preset by default).
  all              Run all three modes sequentially.

Options:
  --model <name>                Model name override (default: tinyllama:latest)
  --prompt <text>               Prompt text
  --repeat <n>                  Repeat per case (default: 3)
  --out <path>                  Output TSV path (default: worklog/bench_<mode>_timestamp.tsv)
  --preset <name>               Base preset for thread/keepalive sweep (default: gfx900_safe)
  --threads <csv>               Thread list for thread-sweep (default: 2,4,6)
  --keep-alive-values <csv>     Keep-alive set for keepalive-sweep (default: 10s,30s,5m)

Environment variables (optional):
  MODEL, PROMPT, REPEAT, OUT, PRESET, THREADS, KEEP_ALIVE_VALUES
USAGE
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

MODE="$1"
shift

if [[ "$MODE" == "-h" || "$MODE" == "--help" ]]; then
  usage
  exit 0
fi

case "$MODE" in
  preset-sweep|thread-sweep|keepalive-sweep|all)
    ;;
  *)
    echo "[error] unknown mode: $MODE" >&2
    usage
    exit 1
    ;;
esac

MODEL="${MODEL:-tinyllama:latest}"
PROMPT="${PROMPT:-short test}"
REPEAT="${REPEAT:-3}"
OUT="${OUT:-worklog/bench_${MODE}_$(date +%Y%m%d_%H%M%S).tsv}"
PRESET="${PRESET:-gfx900_safe}"
THREADS_CSV="${THREADS:-2,4,6}"
KEEP_ALIVE_CSV="${KEEP_ALIVE_VALUES:-10s,30s,5m}"

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
    --out)
      OUT="$2"
      shift 2
      ;;
    --preset)
      PRESET="$2"
      shift 2
      ;;
    --threads)
      THREADS_CSV="$2"
      shift 2
      ;;
    --keep-alive-values)
      KEEP_ALIVE_CSV="$2"
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

print_summary() {
  local tsv_path="$1"

  echo ""
  echo "[summary] avg tok/s by mode+preset_effective"
  awk -F'\t' '
    NR == 1 { next }
    $11 != "" {
      key = $2 "/" $4;
      tok[key] += $11;
      n[key] += 1;
    }
    END {
      for (k in tok) {
        printf "  %s -> avg_tok/s=%.2f (n=%d)\n", k, tok[k]/n[k], n[k];
      }
    }
  ' "$tsv_path" | sort

  echo ""
  echo "[summary] avg ttft_ms by mode+preset_effective"
  awk -F'\t' '
    NR == 1 { next }
    $9 != "" {
      key = $2 "/" $4;
      ttft[key] += $9;
      n[key] += 1;
    }
    END {
      for (k in ttft) {
        printf "  %s -> avg_ttft_ms=%.2f (n=%d)\n", k, ttft[k]/n[k], n[k];
      }
    }
  ' "$tsv_path" | sort
}

CLIENT_BIN="$(resolve_client_bin)"

"$CLIENT_BIN" \
  --bench "$MODE" \
  --model "$MODEL" \
  --prompt "$PROMPT" \
  --repeat "$REPEAT" \
  --preset "$PRESET" \
  --threads "$THREADS_CSV" \
  --keep-alive-values "$KEEP_ALIVE_CSV" \
  --out "$OUT"

echo "[info] benchmark output: $OUT"
if [[ -f "$OUT" ]]; then
  print_summary "$OUT"
fi
