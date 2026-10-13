#!/bin/bash
# alight session 1 (under a gpuq lease, from ~/trashcan-exp/flashnext): GEMM microbenchmark v3 (grid order and a
# diagnostic with cached inputs), the mixer shapes (split bottleneck, up projection with the read gates), then
# trashcan-pf2b (split-K out projections, split expert gate/up pipelines, DeltaNet rows) stage-bench with
# per-dispatch profiles and p100 stage dumps.
cd ~/trashcan-exp/flashnext
./pf2-gemm.sh v3 "16 64 128" 10240x2560 qwen4_q8_gemm,qwen4_q8_gemm_s16r1p,qwen4_q8_gemm_s16r1po,qwen4_q8_gemm_s16r1pod,qwen4_q8_gemm_s16r2,qwen4_q8_gemm_s16r2o,qwen4_q8_gemm_s16r2od,qwen4_q8_gemm_s8r2o,qwen4_q8_gemm_s8r2po,qwen4_q8_gemm_s8r1po,qwen4_q8_gemm_s16r2po
BIN=trashcan-pf2c ./pf2-gemm.sh mixup "16 32 64 128" 10240x320 qwen4_q8_gemm_mix,qwen4_q8_gemm_s8r4om,qwen4_q8_gemm_s16r4om,qwen4_q8_gemm_s4r4om
BIN=trashcan-pf2c ./pf2-gemm.sh mixdn "16 32 64 128" 384x10240 qwen4_q8_gemm:16,qwen4_q8_gemm_s16r1po:16,qwen4_q8_gemm_s8r1po:16,qwen4_q8_gemm_s16r1po:32,qwen4_q8_gemm_s8r1po:32
BIN=trashcan-pf2c ./pf2-gemm.sh out "16 32 64 128" 2560x6144 qwen4_q8_gemm:4,qwen4_q8_gemm_s16r1po:4,qwen4_q8_gemm_s8r1po:4,qwen4_q8_gemm_s8r2po:4,qwen4_q8_gemm_s16r2o:4,qwen4_q8_gemm:8,qwen4_q8_gemm_s16r1po:8
BIN=trashcan-pf2d ./pf2-gemm.sh tiled "64 128" 10240x2560,6144x2560,2560x6144 qwen4_q8_gemm,qwen4_q8_gemm_t64x64_44k32,qwen4_q8_gemm_t128x64_84k32,qwen4_q8_gemm_t128x64_84k16,qwen4_q8_gemm_t128x64_48k32,qwen4_q8_gemm_t128x128_88k16,qwen4_q8_gemm_t256x64_88k16
./pf2-sb.sh "16 32 64 128" b=trashcan-pf2b
./pf2-dump.sh pf2-p100.ids "128 32" b=trashcan-pf2b
./pf2-sb.sh "32 128" e=trashcan-pf2e
./pf2-dump.sh pf2-p100.ids "128" e=trashcan-pf2e
./pf2-sb.sh "32 64" f=trashcan-pf2f flop=trashcan-pf2f:TRASHCAN_PF2_EXP=lop
./pf2-dump.sh pf2-p100.ids "128" f=trashcan-pf2f
echo "pf2-s1 done"
