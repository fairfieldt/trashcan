#!/bin/bash
# alight session 2 (under a gpuq lease, from ~/trashcan-exp/flashnext): trashcan-pf2g (lane-per-row GEMMs up to 32
# tokens, split-K out projections, DeltaNet rows with LDS, per-stream hc_norm, gate/up prefetch for small tiles):
# profiled stage-bench at every chunk size, p100 stage dumps (and main's at 16-token chunks), shader statistics.
cd ~/trashcan-exp/flashnext
./pf2-sb.sh "16 32 64 128" g=trashcan-pf2g
./pf2-dump.sh pf2-p100.ids "128 32 16" g=trashcan-pf2g
./pf2-dump.sh pf2-p100.ids "16" base=trashcan-4c1346d
MESA_SHADER_CACHE_DISABLE=true RADV_DEBUG=shaders,shaderstats,nocache ./trashcan-pf2g qwen-gemm-bench --shapes 2560x2560 \
    --n 16 --kernels qwen4_q8_gemm --iterations 1 --gpu 0 > /dev/null 2> results/pf2-isa-g.txt
./pf2-sb.sh "32 64 128" ivx=trashcan-pf2i:TRASHCAN_PF2_EXP=vx idp=trashcan-pf2i:TRASHCAN_PF2_EXP=dp idpvx=trashcan-pf2i:TRASHCAN_PF2_EXP=dp+vx
./pf2-sb.sh "32 64" j=trashcan-pf2j
echo "pf2-s2 done"
