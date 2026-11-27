#!/bin/bash
# the model's GPU part moves to another GPU of the process (POST /move): the K/V stays (the repeated prompt is served from
# the cache), the output does not change. Two devices on one card for a local run: GGML_VK_VISIBLE_DEVICES=0,0.
# Args: tag, then extra llama-server flags (e.g. -ngl 20 --dual-buffer). Env: BIN, MODEL, PORT, DEV_A, DEV_B.
BIN=${BIN:-$(dirname "$0")/../../build/bin/llama-server.exe}
M=${MODEL:-$(dirname "$0")/../../models/qwen2.5-1.5b-instruct-q4_k_m.gguf}
P=${PORT:-18133}; A=${DEV_A:-Vulkan0}; B=${DEV_B:-Vulkan1}; tag=${1:-move}; shift
$BIN -m $M --port $P -c 4096 -np 1 --load-mode none --park -dev $A -fa on --no-webui --log-verbosity 4 "$@" > srv-$tag.log 2>&1 & sp=$!
until curl -s localhost:$P/health | grep -q ok; do sleep 0.5; kill -0 $sp 2>/dev/null || { echo died; tail -5 srv-$tag.log; exit 1; }; done
LONG=$(for i in $(seq 1 60); do printf 'Line %d: the green frog sits on the old stone bridge. ' $i; done)
ask() { printf '%s' "{\"prompt\":\"$1\",\"n_predict\":24,\"temperature\":0,\"cache_prompt\":true}" > req-$tag.json; curl -s localhost:$P/completion -d @req-$tag.json | node -e 'let s="";process.stdin.on("data",d=>s+=d).on("end",()=>{const j=JSON.parse(s);console.log("prompt_n="+j.timings.prompt_n+" "+JSON.stringify(j.content))})'; }
post() { curl -s -X POST localhost:$P/$1 ${2:+-H 'Content-Type: application/json' -d "$2"}; echo; }
Q="$LONG What color is the frog? Answer:"
echo "ref:      $(ask "$Q")"
echo "move B:   $(post move "{\"device\":\"$B\"}")"
echo "on B:     $(ask "$Q")"
echo "park:     $(post park)"
echo "cpu:      $(ask "$Q")"
echo "move A:   $(post move "{\"device\":\"$A\"}")"
echo "on A:     $(ask "$Q")"
echo "bad:      $(post move '{"device":"nope"}')"
kill $sp; wait $sp 2>/dev/null
grep -E "move_device|park: |llama_park|unparked|parked:" srv-$tag.log | cut -c1-240
