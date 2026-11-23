#!/bin/bash
# in-process park/unpark: answers before/while/after, timings and the measured device free memory
BIN=${BIN:-/c/Users/bl/Documents/projects/llama-dualbuf/build/bin/llama-server.exe}
M=${MODEL:-/c/Users/bl/Documents/projects/llama-dualbuf/models/qwen2.5-1.5b-instruct-q4_k_m.gguf}
P=18131; tag=${1:-park}; shift
env "$@" $BIN -m $M --port $P -c ${CTX:-8192} -np 1 --load-mode none --auto-zones --park -fa on --no-webui --log-verbosity 4 ${EXTRA:-} > srv-$tag.log 2>&1 & sp=$!
until curl -s localhost:$P/health | grep -q ok; do sleep 0.5; kill -0 $sp 2>/dev/null || { echo died; tail -5 srv-$tag.log; exit 1; }; done
LONG=$(for i in $(seq 1 120); do printf 'Line %d: the green frog sits on the old stone bridge. ' $i; done)
ask() { printf '%s' "{\"prompt\":\"$1\",\"n_predict\":24,\"temperature\":0,\"cache_prompt\":true}" > req-$tag.json; curl -s localhost:$P/completion -d @req-$tag.json | node -e 'let s="";process.stdin.on("data",d=>s+=d).on("end",()=>{const j=JSON.parse(s);console.log("prompt_n="+j.timings.prompt_n+" "+JSON.stringify(j.content))})'; }
post() { curl -s -X POST localhost:$P/$1; echo; }
Q1="$LONG What color is the frog? Answer:"
echo "ref:      $(ask "$Q1")"
for r in 1 2 3; do
  echo "park $r:   $(post park)"
  echo "parked:   $(ask "The capital of France is")"
  echo "unpark $r: $(post unpark)"
  echo "after $r:  $(ask "$Q1")"
done
kill $sp; wait $sp 2>/dev/null
grep -E "using device|park:|llama_context::park|parked:|unparked:|warmup" srv-$tag.log | grep -v "^.*sched_reserve" | cut -c1-260
