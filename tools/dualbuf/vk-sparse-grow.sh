#!/bin/bash
# sparse KV growth on Vulkan: run once with the given env, print the answer and the KV log lines
BIN=${BIN:-/c/Users/bl/Documents/projects/llama-dualbuf/build/bin/llama-server.exe}
M=${MODEL:-/c/Users/bl/Documents/projects/llama-dualbuf/models/qwen2.5-1.5b-instruct-q4_k_m.gguf}
P=18130; tag=$1; shift
PROMPT=$(for i in $(seq 1 ${REP:-450}); do printf 'Line %d: the green frog sits on the old stone bridge. ' $i; done)
env "$@" $BIN -m $M --port $P -c ${CTX:-8192} -np 1 --load-mode none --auto-zones -fa on --no-webui --log-verbosity 4 > srv-$tag.log 2>&1 & sp=$!
until curl -s localhost:$P/health | grep -q ok; do sleep 0.5; kill -0 $sp 2>/dev/null || { echo died; tail -5 srv-$tag.log; exit 1; }; done
ask() { printf "%s" "{\"prompt\":\"$1\",\"n_predict\":16,\"temperature\":0,\"cache_prompt\":false}" > req-$tag.json; curl -s localhost:$P/completion -d @req-$tag.json | node -e 'let s="";process.stdin.on("data",d=>s+=d).on("end",()=>{const j=JSON.parse(s);console.log(JSON.stringify(j.content))})'; }
echo "== $tag long:  $(ask "$PROMPT What color is the frog? Answer:")"
echo "== $tag short: $(ask "The capital of France is")"
kill $sp; wait $sp 2>/dev/null
grep -E "sparse|set_dynamic|relayout|zones_state|growable|grows by copy" srv-$tag.log | cut -c1-230
