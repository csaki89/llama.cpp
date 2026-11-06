#!/usr/bin/env bash
# Flex-layer tests (DESIGN-3zone.md): long prompt from bench.sh (run it first), -ngl 14, 200 generated tokens.
# LLAMA_FLEX_TEST="<tokens>:<take|put>:<layer>,..." moves a flex layer after that many decoded tokens.
cd "$(dirname "$0")/../.."; mkdir -p bench-flex
E=build/bin/llama-completion.exe; M=models/qwen2.5-1.5b-instruct-q4_k_m.gguf; O=bench-flex
run(){ n=$1; shift; "$E" -m $M -ngl 14 --load-mode none --log-verbosity 4 -fa auto -b 512 -ub 512 --temp 0 --seed 1 -n 200 --ignore-eos -no-cnv --no-display-prompt --perf -f bench-out/prompt_long.txt "$@" > $O/$n.out 2> $O/$n.err
  pp=$(sed -n 's/.*prompt eval time = .*, *\([0-9.]*\) tokens per second.*/\1/p' $O/$n.err | tail -1)
  tg=$(sed -n 's/.*       eval time = .*, *\([0-9.]*\) tokens per second.*/\1/p' $O/$n.err | tail -1)
  bufs=$(sed -n 's/.*load_tensors: *\([A-Za-z0-9_]*\) model buffer size = *\([0-9.]*\) MiB.*/\1=\2/p' $O/$n.err | tr '\n' ' ')
  bd=$(sed 's/\x1b\[[0-9;]*m//g' $O/$n.err | grep "memory_breakdown_print: |   - Vulkan0" | tail -1 | sed 's/.*= *[0-9]* + (\([0-9]*\) = *\([0-9]*\) +.*/self=\1 model=\2/')
  moves=$(sed 's/\x1b\[[0-9;]*m//g' $O/$n.err | grep "flex_move: layer" | sed 's/.*layer \([0-9]*\) moved to \([A-Za-z0-9_]*\) (\([0-9.]*\) MiB, \([0-9.]*\) ms).*/L\1->\2:\4ms/' | tr '\n' ' ')
  printf '%s\tpp=%s\ttg=%s\tmd5=%s\t%s\t%s\t%s\n' "$n" "$pp" "$tg" "$(md5sum < $O/$n.out | cut -c1-8)" "$bd" "$moves" "$bufs"
}
for r in 1 2; do
run base_r$r
run flex_nomove_r$r --flex-layers 2
LLAMA_FLEX_TEST="0:take:26,0:take:27" run off_all_r$r --flex-layers 2
LLAMA_FLEX_TEST="0:take:26,0:take:27,1611:put:26,1611:put:27" run off_pp_on_tg_r$r --flex-layers 2
LLAMA_FLEX_TEST="1611:take:26,1611:take:27" run on_pp_off_tg_r$r --flex-layers 2
LLAMA_FLEX_TEST="1611:take:26,1611:take:27" LLAMA_FLEX_TEST_RESERVE=1 run on_pp_off_tg_reserve_r$r --flex-layers 2
LLAMA_FLEX_TEST="1661:take:27,1711:put:27" run take_put_r$r --flex-layers 2
done
run dual --dual-buffer
LLAMA_FLEX_TEST="1661:take:27,1711:put:27" run dual_flex_take_put --dual-buffer --flex-layers 2
LLAMA_FLEX_TEST="0:take:26,0:take:27" run dual_flex_off_all --dual-buffer --flex-layers 2
