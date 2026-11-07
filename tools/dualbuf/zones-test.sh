#!/usr/bin/env bash
# Auto-zones test (DESIGN-3zone.md 6): a long conversation that grows the KV (flex layers go off), then a new short one (they come back).
# Env: SERVER (llama-server), MODEL, OUT (bench-zones), PORT (8091), ARGS (extra server args), LONG (long prompt file), NPRED (100).
# The band is forced with --fit-target (zone 1 + 2 from a small context); LLAMA_ZONES_HEADROOM_MIB sets the device memory budget.
cd "$(dirname "$0")/../.."
S=${SERVER:-build/bin/llama-server.exe}; [ -x "$S" ] || S=build/bin/llama-server; [ -x "$S" ] || S=llama-server
M=${MODEL:-models/qwen2.5-1.5b-instruct-q4_k_m.gguf}; O=${OUT:-bench-zones}; P=${PORT:-8091}; N=${NPRED:-100}
mkdir -p "$O"
LONG=${LONG:-$O/prompt_long4.txt}
if [ ! -f "$LONG" ]; then # 4 x the 14 sections of bench.sh (~6400 tokens)
  : > "$LONG"; for i in 1 2 3 4; do head -n 14 bench-out/prompt_long.txt >> "$LONG"; done; echo "Summarize the text above in one sentence." >> "$LONG"
fi
"$S" -m "$M" --port "$P" --load-mode none --log-verbosity 4 -fa auto -b 512 -ub 512 -np 1 --temp 0 ${ARGS:-} > "$O/server.log" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null' EXIT
for i in $(seq 1 120); do curl -sf "http://127.0.0.1:$P/health" > /dev/null && break; sleep 1; done
req(){ # $1 name, $2 prompt file
  node -e 'const fs=require("fs");process.stdout.write(JSON.stringify({prompt:fs.readFileSync(process.argv[1],"utf8"),n_predict:+process.argv[2],temperature:0,seed:1,cache_prompt:true}))' "$2" "$N" > "$O/$1.req.json"
  curl -s "http://127.0.0.1:$P/completion" -H 'Content-Type: application/json' --data-binary @"$O/$1.req.json" > "$O/$1.json"
  node -e 'const r=JSON.parse(require("fs").readFileSync(process.argv[1],"utf8"));const t=r.timings||{};console.log(process.argv[2]+"\tprompt_n="+t.prompt_n+"\tpp="+(t.prompt_per_second||0).toFixed(1)+"\ttg_n="+t.predicted_n+"\ttg="+(t.predicted_per_second||0).toFixed(2)+"\tmd5="+require("crypto").createHash("md5").update(r.content||"").digest("hex").slice(0,8)+"\t"+JSON.stringify((r.content||"").slice(0,120)))' "$O/$1.json" "$1"
}
printf 'Write a short story about a lighthouse keeper.\n' > "$O/short.txt"
req long1 "$LONG"
req short "$O/short.txt"
req long2 "$LONG"
sed 's/\x1b\[[0-9;]*m//g' "$O/server.log" | grep -E "auto zones|zones:|dynamic KV|resize:|flex_move: layer" | sed 's/^[0-9.]* . //' | cut -c1-200
