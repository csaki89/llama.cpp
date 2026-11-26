#!/bin/bash
# parked model on the CPU: the demote (repack of the park copies), kept across unpark/park, dropped by POST /park/drop
BIN=${BIN:-$(dirname "$0")/../../build/bin/llama-server.exe}
M=${MODEL:-$(dirname "$0")/../../models/qwen2.5-1.5b-instruct-q4_k_m.gguf}
P=18132; tag=${1:-demote}; shift
env "$@" $BIN -m $M --port $P -c 8192 -np 1 --load-mode none --auto-zones --park -fa on --no-webui --log-verbosity 4 > srv-$tag.log 2>&1 & sp=$!
until curl -s localhost:$P/health | grep -q ok; do sleep 0.5; kill -0 $sp 2>/dev/null || { echo died; tail -5 srv-$tag.log; exit 1; }; done
LONG=$(for i in $(seq 1 120); do printf 'Line %d: the green frog sits on the old stone bridge. ' $i; done)
ask() { printf '%s' "{\"prompt\":\"$1\",\"n_predict\":24,\"temperature\":0,\"cache_prompt\":true}" > req-$tag.json; curl -s localhost:$P/completion -d @req-$tag.json | node -e 'let s="";process.stdin.on("data",d=>s+=d).on("end",()=>{const j=JSON.parse(s);console.log("prompt_n="+j.timings.prompt_n+" gen="+j.timings.predicted_per_second.toFixed(1)+"t/s "+JSON.stringify(j.content))})'; }
post() { curl -s -X POST localhost:$P/$1; echo; }
Q1="$LONG What color is the frog? Answer:"
echo "ref:        $(ask "$Q1")"
echo "park:       $(post park)"
echo "cpu 1:      $(ask "The capital of France is")"
echo "cpu 2:      $(ask "The capital of Germany is")"
echo "unpark:     $(post unpark)"
echo "after:      $(ask "$Q1")"
echo "park:       $(post park)"
echo "cpu 3:      $(ask "The capital of Spain is")"
echo "drop:       $(post park/drop)"
echo "cpu 4:      $(ask "The capital of Italy is")"
echo "unpark:     $(post unpark)"
echo "after 2:    $(ask "$Q1")"
kill $sp; wait $sp 2>/dev/null
grep -E "park: |park_demote|park_drop" srv-$tag.log | cut -c1-230
