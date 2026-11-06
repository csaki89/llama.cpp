#!/usr/bin/env bash
# Flex-layer tests (DESIGN-3zone.md): long prompt from bench.sh (run it first), -ngl 14, 200 generated tokens.
# LLAMA_FLEX_TEST="<tokens>:<take|put>:<layer>,..." moves a flex layer after that many decoded tokens.
# LLAMA_FLEX_TEST_KV=0 keeps the K/V in place, LLAMA_FLEX_TEST_RESERVE=1 re-reserves the scheduler after a move.
cd "$(dirname "$0")/../.."; mkdir -p bench-flex
E=build/bin/llama-completion.exe; [ -x "$E" ] || E=build/bin/llama-completion
M=models/qwen2.5-1.5b-instruct-q4_k_m.gguf; O=bench-flex
run(){ n=$1; shift; "$E" -m $M -ngl 14 --load-mode none --log-verbosity 4 -fa auto -b 512 -ub 512 --temp 0 --seed 1 -n 200 --ignore-eos -no-cnv --no-display-prompt --perf -f bench-out/prompt_long.txt "$@" > $O/$n.out 2> $O/$n.err
  L=$(sed 's/\x1b\[[0-9;]*m//g' $O/$n.err)
  pp=$(echo "$L" | sed -n 's/.*prompt eval time = .*, *\([0-9.]*\) tokens per second.*/\1/p' | tail -1)
  tg=$(echo "$L" | sed -n 's/.*       eval time = .*, *\([0-9.]*\) tokens per second.*/\1/p' | tail -1)
  # Vulkan0 line of the final memory breakdown: self = model + context (KV) + compute
  bd=$(echo "$L" | grep "memory_breakdown_print: |   - Vulkan0" | tail -1 | sed 's/.*= *[0-9]* + (\([0-9]*\) = *\([0-9]*\) + *\([0-9]*\) + *\([0-9]*\)).*/self=\1 model=\2 ctx=\3 compute=\4/')
  moves=$(echo "$L" | grep -E "flex_move: layer" | sed -E 's/.*layer ([0-9]+) (K\/V )?moved to ([A-Za-z0-9_]+) \(([0-9.]+) MiB, ([0-9.]+) ms\).*/L\1\2->\3:\4MiB:\5ms/; s/K\/V /kv/' | tr '\n' ' ')
  # splits of the graphs actually computed (last pp and last tg graph realloc) and of the reserve graphs
  spl=$(echo "$L" | grep -E "sched_graph_realloc: graph reallocation" | sed 's/.*nodes = \([0-9]*\).*splits = \([0-9]*\).*/\1n:\2s/' | tr '\n' ' ')
  res=$(echo "$L" | grep -E "sched_reserve: graph \(pp" | sed 's/.*splits = \([0-9]* \/ [0-9]*\).*/\1/' | tr -d ' ' | tr '\n' ' ')
  printf '%s\tpp=%s\ttg=%s\tmd5=%s\t%s\tmoves=%s\tgraphs=%s\treserve_splits(pp/tg)=%s\n' "$n" "$pp" "$tg" "$(md5sum < $O/$n.out | cut -c1-8)" "$bd" "$moves" "$spl" "$res"
}
run base
run flex_nomove --flex-layers 2
LLAMA_FLEX_TEST="0:take:26,0:take:27" LLAMA_FLEX_TEST_KV=0 run off_all_kv_stays --flex-layers 2
LLAMA_FLEX_TEST="0:take:26,0:take:27" run off_all_kv_moves --flex-layers 2
LLAMA_FLEX_TEST="1611:take:26,1611:take:27" LLAMA_FLEX_TEST_KV=0 LLAMA_FLEX_TEST_RESERVE=1 run on_pp_off_tg_kv_stays --flex-layers 2
LLAMA_FLEX_TEST="1611:take:26,1611:take:27" LLAMA_FLEX_TEST_RESERVE=1 run on_pp_off_tg_kv_moves --flex-layers 2
LLAMA_FLEX_TEST="1661:take:27,1711:put:27" run take_put --flex-layers 2
LLAMA_FLEX_TEST="0:take:26,0:take:27,1611:put:26,1611:put:27" run off_pp_on_tg --flex-layers 2
LLAMA_FLEX_TEST="1661:take:27,1711:put:27" run dual_take_put --dual-buffer --flex-layers 2
