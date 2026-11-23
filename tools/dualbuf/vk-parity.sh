#!/bin/bash
# Vulkan parity run: dual-buffer + auto zones (+ flex band from a reduced device budget), a long then a short request
# usage: parity.sh <tag> [ENV=VAL ...]; EXTRA= extra llama-server args, CTX=, FIT= (MiB kept free, default 6500)
BIN=${BIN:-/c/Users/bl/Documents/projects/llama-dualbuf/build/bin/llama-server.exe}
M=${MODEL:-/c/Users/bl/Documents/projects/llama-dualbuf/models/qwen2.5-1.5b-instruct-q4_k_m.gguf}
P=18140; tag=$1; shift
env "$@" $BIN -m $M --port $P -c ${CTX:-16384} -np 1 --load-mode none -fa on --no-webui --log-verbosity 4 --fit-target ${FIT:-6500} ${MODE:---auto-zones --dual-buffer} ${EXTRA:-} > srv-$tag.log 2>&1 & sp=$!
until curl -s localhost:$P/health | grep -q ok; do sleep 0.5; kill -0 $sp 2>/dev/null || { echo died; tail -5 srv-$tag.log; exit 1; }; done
LONG=$(for i in $(seq 1 ${REP:-700}); do printf 'Line %d: the green frog sits on the old stone bridge. ' $i; done)
ask() { printf '%s' "{\"prompt\":\"$1\",\"n_predict\":24,\"temperature\":0,\"cache_prompt\":false}" > req-$tag.json; curl -s localhost:$P/completion -d @req-$tag.json | node -e 'let s="";process.stdin.on("data",d=>s+=d).on("end",()=>{const j=JSON.parse(s);console.log("prompt_n="+j.timings.prompt_n+" "+JSON.stringify(j.content))})'; }
echo "$tag long:  $(ask "$LONG What color is the frog? Answer:")"
echo "$tag short: $(ask "The capital of France is")"
echo "$tag short2: $(ask "The capital of Germany is")"
kill $sp; wait $sp 2>/dev/null
