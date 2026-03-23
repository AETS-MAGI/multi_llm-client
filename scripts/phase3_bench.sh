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
  --keep-alive-values <csv>     Keep-alive set for keepalive-sweep (default: 0s,10m)

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

MODEL="${MODEL:-tinyllama:latest}"
PROMPT="${PROMPT:-short test}"
REPEAT="${REPEAT:-3}"
OUT="${OUT:-worklog/bench_${MODE}_$(date +%Y%m%d_%H%M%S).tsv}"
PRESET="${PRESET:-gfx900_safe}"
THREADS_CSV="${THREADS:-2,4,6}"
KEEP_ALIVE_CSV="${KEEP_ALIVE_VALUES:-0s,10m}"

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

IFS=',' read -r -a THREAD_SET <<<"$THREADS_CSV"
IFS=',' read -r -a KEEP_ALIVE_SET <<<"$KEEP_ALIVE_CSV"

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

latest_log_line() {
  local latest
  latest="$(ls -1t "$ROOT"/logs/infer-*.jsonl 2>/dev/null | head -n1 || true)"
  if [[ -z "$latest" ]]; then
    return 1
  fi
  tail -n1 "$latest"
}

write_header() {
  mkdir -p "$(dirname "$OUT")"
  echo -e "ts_unix\tmode\tmodel\tpreset_effective\trequested_preset\tnum_thread\tkeep_alive\trepeat_idx\tttft_ms\ttotal_ms\ttok_s\tresponse_chars\trc\terror" > "$OUT"
}

append_row() {
  local ts="$1"
  local mode="$2"
  local model="$3"
  local preset_effective="$4"
  local requested_preset="$5"
  local num_thread="$6"
  local keep_alive="$7"
  local rep_idx="$8"
  local ttft_ms="$9"
  local total_ms="${10}"
  local tok_s="${11}"
  local response_chars="${12}"
  local rc="${13}"
  local error="${14}"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$ts" "$mode" "$model" "$preset_effective" "$requested_preset" "$num_thread" "$keep_alive" \
    "$rep_idx" "$ttft_ms" "$total_ms" "$tok_s" "$response_chars" "$rc" "$error" >> "$OUT"
}

run_case() {
  local mode="$1"
  local requested_preset="$2"
  local num_thread="$3"
  local keep_alive="$4"
  local rep_idx="$5"

  local -a args
  args=(--prompt "$PROMPT" --repeat 1 --model "$MODEL" --preset "$requested_preset" --stream false --inline-stream false --quiet)

  if [[ "$num_thread" == "none" ]]; then
    args+=(--num-thread none)
  else
    args+=(--num-thread "$num_thread")
  fi

  if [[ "$keep_alive" == "none" ]]; then
    args+=(--keep-alive none)
  else
    args+=(--keep-alive "$keep_alive")
  fi

  local rc=0
  if ! "$CLIENT_BIN" "${args[@]}" >/tmp/multi_llm_client_phase3.out 2>/tmp/multi_llm_client_phase3.err; then
    rc=$?
  fi

  local line=""
  if ! line="$(latest_log_line)"; then
    local err
    err="$(cat /tmp/multi_llm_client_phase3.err 2>/dev/null | tr '\t\r\n' ' ' | sed 's/  */ /g')"
    append_row "0" "$mode" "$MODEL" "" "$requested_preset" "$num_thread" "$keep_alive" "$rep_idx" "" "" "" "" "$rc" "$err"
    echo "[warn] no log line found mode=$mode preset=$requested_preset num_thread=$num_thread keep_alive=$keep_alive rep=$rep_idx rc=$rc" >&2
    return
  fi

  local ts preset_effective ttft_ms total_ms tok_s response_chars error
  ts="$(jq -r '.ts_unix_secs // 0' <<<"$line")"
  preset_effective="$(jq -r '.effective.preset // ""' <<<"$line")"
  ttft_ms="$(jq -r '.ttft_ms // ""' <<<"$line")"
  total_ms="$(jq -r '.total_ms // ""' <<<"$line")"
  tok_s="$(jq -r '.approx_tok_per_sec // ""' <<<"$line")"
  response_chars="$(jq -r '.response_chars // ""' <<<"$line")"
  error="$(jq -r '.error // ""' <<<"$line" | tr '\t\r\n' ' ' | sed 's/  */ /g')"

  append_row "$ts" "$mode" "$MODEL" "$preset_effective" "$requested_preset" "$num_thread" "$keep_alive" "$rep_idx" \
    "$ttft_ms" "$total_ms" "$tok_s" "$response_chars" "$rc" "$error"

  echo "[ok] mode=$mode preset=$preset_effective req=$requested_preset num_thread=$num_thread keep_alive=$keep_alive rep=$rep_idx ttft=$ttft_ms total=$total_ms tok/s=$tok_s rc=$rc"
}

run_preset_sweep() {
  local -a presets=(gfx900_safe gfx900_balanced gfx900_longctx gfx900_tinybench)
  local preset rep
  for preset in "${presets[@]}"; do
    for rep in $(seq 1 "$REPEAT"); do
      run_case "preset-sweep" "$preset" "none" "10m" "$rep"
    done
  done
}

run_thread_sweep() {
  local t rep
  for t in "${THREAD_SET[@]}"; do
    for rep in $(seq 1 "$REPEAT"); do
      run_case "thread-sweep" "$PRESET" "$t" "10m" "$rep"
    done
  done
}

run_keepalive_sweep() {
  local ka rep
  for ka in "${KEEP_ALIVE_SET[@]}"; do
    for rep in $(seq 1 "$REPEAT"); do
      run_case "keepalive-sweep" "$PRESET" "none" "$ka" "$rep"
    done
  done
}

print_summary() {
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
  ' "$OUT" | sort

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
  ' "$OUT" | sort
}

CLIENT_BIN="$(resolve_client_bin)"
case "$MODE" in
  preset-sweep|thread-sweep|keepalive-sweep|all)
    ;;
  *)
    echo "[error] unknown mode: $MODE" >&2
    usage
    exit 1
    ;;
esac

write_header

echo "[info] mode=$MODE model=$MODEL repeat=$REPEAT prompt_chars=${#PROMPT}"
echo "[info] output=$OUT"

action_done=0
case "$MODE" in
  preset-sweep)
    run_preset_sweep
    action_done=1
    ;;
  thread-sweep)
    run_thread_sweep
    action_done=1
    ;;
  keepalive-sweep)
    run_keepalive_sweep
    action_done=1
    ;;
  all)
    run_preset_sweep
    run_thread_sweep
    run_keepalive_sweep
    action_done=1
    ;;
esac

if [[ "$action_done" -eq 1 ]]; then
  print_summary
  echo "[done] TSV written: $OUT"
fi
