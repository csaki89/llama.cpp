#!/usr/bin/env bash
# Buffer/speed baseline for the hybrid CPU+GPU modes. Works in Git Bash and on Linux.
# Usage: tools/dualbuf/bench.sh [-b BIN_DIR] [-m MODEL.gguf] [-o OUT_DIR] [-c MODES.conf] [-n NGEN]
# Env:   NGL (gpu layers for the partial modes, default 14), THREADS (optional -t), REPS (default 3)
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
BIN="$ROOT/build/bin"; MODEL="$ROOT/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
OUT="$ROOT/bench-out"; CONF="$HERE/modes.conf"; NGEN=200; NGL="${NGL:-14}"; REPS="${REPS:-3}"
while getopts "b:m:o:c:n:" o; do
  case $o in b) BIN=$OPTARG;; m) MODEL=$OPTARG;; o) OUT=$OPTARG;; c) CONF=$OPTARG;; n) NGEN=$OPTARG;; *) exit 2;; esac
done
EXE="$BIN/llama-completion"; [ -x "$EXE" ] || EXE="$BIN/llama-completion.exe"
[ -x "$EXE" ] || { echo "llama-completion not found in $BIN" >&2; exit 1; }
[ -f "$MODEL" ] || { echo "model not found: $MODEL" >&2; exit 1; }

# LLAMA_ARG_* env vars would override the CLI flags (e.g. LLAMA_ARG_NO_HOST): report and remove them.
for v in $(env | grep -o '^LLAMA_[A-Za-z0-9_]*'); do echo "WARNING: unsetting $v" >&2; unset "$v"; done
echo "LLAMA_* env vars after cleanup: $(env | grep -c '^LLAMA_')"

mkdir -p "$OUT"
# Prompts: short (~100 tokens) and long (~1500 tokens), plain text, generated deterministically.
P="The history of computing is a story of steady abstraction. Early machines were programmed by rewiring panels, then by punched cards, then by assembly language and compilers. Each layer hid the details of the one below it, so that engineers could reason about larger systems with less effort."
printf '%s %s\n' "$P" "$P" > "$OUT/prompt_short.txt"
: > "$OUT/prompt_long.txt"
for i in $(seq 1 14); do printf 'Section %s. %s %s\n' "$i" "$P" "$P" >> "$OUT/prompt_long.txt"; done
printf 'Summarize the text above in one sentence.\n' >> "$OUT/prompt_long.txt"

# Untimed warm-up (shader/disk cache) so the first measured run is not an outlier.
"$EXE" -m "$MODEL" -ngl "$NGL" --load-mode none -n 8 -no-cnv --no-display-prompt -p "warm up" > /dev/null 2>&1
TSV="$OUT/results.tsv"
printf 'mode\trep\tprompt\tpp_tok\tpp_tps\ttg_tok\ttg_tps\tbuffers\tout_md5\tout_head\n' > "$TSV"
while IFS='|' read -r name args; do
  name="$(echo "$name" | xargs)"; [ -z "$name" ] && continue; case $name in \#*) continue;; esac
  args="${args//@NGL@/$NGL}"
  for pr in short long; do for rep in $(seq 1 "$REPS"); do
    base="$OUT/${name}__${pr}__r${rep}"
    # shellcheck disable=SC2086
    timeout 1200 "$EXE" -m "$MODEL" $args ${THREADS:+-t $THREADS} --load-mode none --log-verbosity 4 \
      -fa auto -b 512 -ub 512 --temp 0 --seed 1 -n "$NGEN" --ignore-eos -no-cnv --no-display-prompt --perf \
      -f "$OUT/prompt_$pr.txt" > "$base.out" 2> "$base.err"
    rc=$?
    pp=$(sed -n 's/.*common_perf_print: *prompt eval time = *\([0-9.]*\) ms \/ *\([0-9]*\) tokens.*, *\([0-9.]*\) tokens per second.*/\2 \3/p' "$base.err" | tail -1)
    tg=$(sed -n 's/.*common_perf_print: *eval time = *\([0-9.]*\) ms \/ *\([0-9]*\) runs.*, *\([0-9.]*\) tokens per second.*/\2 \3/p' "$base.err" | tail -1)
    bufs=$(sed -n 's/.*load_tensors: *\([A-Za-z0-9_]*\) model buffer size = *\([0-9.]*\) MiB.*/\1=\2/p' "$base.err" | tr '\n' ' ')
    md5=$(md5sum < "$base.out" | cut -c1-8)
    head=$(head -c 200 "$base.out" | tr -d '\r' | tr '\n\t' '  ')
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$rep" "$pr" "${pp% *}" "${pp#* }" "${tg% *}" "${tg#* }" "$bufs" "$md5" "$head" >> "$TSV"
    echo "[$name/$pr/r$rep] rc=$rc pp=${pp:-?} tg=${tg:-?}"
  done; done
done < <(grep -v '^[[:space:]]*#' "$CONF")
echo "results: $TSV"
