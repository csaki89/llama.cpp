#!/usr/bin/env bash
# Flex-band tests (DESIGN-3zone.md): long prompt from bench.sh (run it first), -ngl $NGL (14) --flex-layers 2, NGEN (200) tokens.
# Env: EXE, MODEL, OUT (bench-flex), NGL, NGEN, PROMPT (bench-out/prompt_long.txt).
# LLAMA_FLEX_TEST="<tokens>:<take|put>[:<layer>],..." moves a flex layer after that many decoded tokens (no layer: the band order).
# The token thresholds below assume the 1611-token long prompt of bench.sh (Qwen2.5 tokenizer); adjust them for another model.
# Test switches: LLAMA_FLEX_TEST_HOST=0 (no op offload from the host copies), LLAMA_FLEX_TEST_FILE=1 (put back from the file),
# LLAMA_FLEX_TEST_KV=0 (K/V stays), LLAMA_FLEX_TEST_COPY_LEGACY=1 (synchronous full copies), LLAMA_FLEX_TEST_RESERVE=1 (re-reserve).
cd "$(dirname "$0")/../.."
E=${EXE:-build/bin/llama-completion.exe}; [ -x "$E" ] || E=build/bin/llama-completion; [ -x "$E" ] || E=llama-completion
mkdir -p "${OUT:-bench-flex}"
M=${MODEL:-models/qwen2.5-1.5b-instruct-q4_k_m.gguf}; O=${OUT:-bench-flex}; N=${NGEN:-200}; G=${NGL:-14}
run(){ n=$1; shift; "$E" -m $M --load-mode none --log-verbosity 4 -fa auto -b 512 -ub 512 --temp 0 --seed 1 -n $N --ignore-eos -no-cnv --no-display-prompt --perf -f ${PROMPT:-bench-out/prompt_long.txt} "$@" > $O/$n.out 2> $O/$n.err
  L=$(sed 's/\x1b\[[0-9;]*m//g' $O/$n.err)
  pp=$(echo "$L" | sed -n 's/.*prompt eval time = .*, *\([0-9.]*\) tokens per second.*/\1/p' | tail -1)
  tg=$(echo "$L" | sed -n 's/.*       eval time = .*, *\([0-9.]*\) tokens per second.*/\1/p' | tail -1)
  # GPU line of the final memory breakdown: self = model + context (K/V) + compute
  bd=$(echo "$L" | grep -E "memory_breakdown_print: \|   - (Vulkan0|ROCm0|CUDA0)" | tail -1 | sed 's/.*= *[0-9]* + (\([0-9]*\) = *\([0-9]*\) + *\([0-9]*\) + *\([0-9]*\)).*/self=\1 model=\2 kv=\3/')
  moves=$(echo "$L" | grep -E "flex_move: layer" | sed -E 's/.*layer ([0-9]+) (K\/V )?moved to ([A-Za-z0-9_]+) \(([0-9.]+) MiB[^)]*, ([0-9.]+) ms[:)].*/L\1\2->\3:\5ms/; s/K\/V /kv/' | tr '\n' ' ')
  res=$(echo "$L" | grep -E "sched_reserve: graph \(pp" | sed 's/.*splits = \([0-9]* \/ [0-9]*\).*/\1/' | tr -d ' ' | tr '\n' ' ')
  printf '%s\tpp=%s\ttg=%s\tmd5=%s\t%s\tsplits(pp/tg)=%s\tmoves=%s\n' "$n" "$pp" "$tg" "$(md5sum < $O/$n.out | cut -c1-8)" "$bd" "$res" "$moves"
}
export LLAMA_FLEX_TEST_RESERVE=1 # prints the splits of the reserve graphs after a move
F="-ngl $G --flex-layers 2"
run base               $F
LLAMA_FLEX_TEST="0:take"               run off1_host        $F
LLAMA_FLEX_TEST="0:take" LLAMA_FLEX_TEST_HOST=0 run off1_nohost $F
LLAMA_FLEX_TEST="0:take,0:take"        run off2_host        $F
LLAMA_FLEX_TEST="0:take,0:take" LLAMA_FLEX_TEST_HOST=0 run off2_nohost $F
LLAMA_FLEX_TEST="1630:take,1630:take,1650:put,1650:put" run takeput_host $F
LLAMA_FLEX_TEST="1630:take,1630:take,1650:put,1650:put" LLAMA_FLEX_TEST_FILE=1 run takeput_file $F
LLAMA_FLEX_TEST="1611:take,1611:take"  run on_pp_off_tg     $F
# a layer that is off for the whole run vs. the same layer as a CPU layer with --dual-buffer
run ref_ngl${G}_dual -ngl $G --dual-buffer
run dual_nomove    $F --dual-buffer
run ref_ngl$((G-1))_dual -ngl $((G-1)) --dual-buffer
LLAMA_FLEX_TEST="0:take"        run dual_off1 $F --dual-buffer
run ref_ngl$((G-2))_dual -ngl $((G-2)) --dual-buffer
LLAMA_FLEX_TEST="0:take,0:take" run dual_off2 $F --dual-buffer
