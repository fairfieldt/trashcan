#!/bin/bash
# dk2 stress check (one Mac, both GPUs, under a gpuq lease, from ~/trashcan-exp/flashnext): layers 0-7 (GPU 0: 0-3 with
# the n-gram module, GPU 1: 4-7) decode the first N tokens of a real text one token at a time (and the longest also
# in chunks of 4), dumping every captured tensor of the last token; main and the candidate alternate, twice each.
# The dumps are compared afterwards (cmpdump.py): a change that keeps the arithmetic must give the same bits.
# usage: dk2-stress.sh TAG CANDIDATE_BINARY [MAIN_BINARY (default trashcan-9baedca)]
set -uo pipefail
cd ~/trashcan-exp/flashnext
tag=$1 cand=$2 base=${3:-trashcan-9baedca}
M=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
R=results/dk2/$tag
mkdir -p $R dk2
for n in 520 1030 2060 2600; do
    [ -f dk2/long-$n.ids ] || tr ' ' '\n' < long-30000.ids | head -$n | tr '\n' ' ' > dk2/long-$n.ids
done
for c in 0 1; do
    f=/sys/class/drm/card$c/device/power_dpm_force_performance_level
    if echo high | sudo -n tee $f > /dev/null; then echo "card$c high ok"; else echo "card$c high FAILED"; fi
    echo auto | sudo -n tee $f > /dev/null
done
dump() {
    name=$1 bin=$2 b=$3 ids=$4
    rm -rf $R/$name
    ./$bin qwen4 stage-bench $M --ranges 0:4,4:8 --gpus 0,1 --context 32768 --batch $b --tokens-file $ids \
        --out $R/$name > $R/$name.json 2> $R/$name.err
    echo "$name rc=$? $(date +%T)"
}
for rep in 1 2; do
    for n in 520 1030 2060 2600; do
        dump main-n$n-r$rep $base 1 dk2/long-$n.ids
        dump cand-n$n-r$rep $cand 1 dk2/long-$n.ids
    done
    dump main-n2600b4-r$rep $base 4 dk2/long-2600.ids
    dump cand-n2600b4-r$rep $cand 4 dk2/long-2600.ids
done
echo "dk2-stress $tag done"
