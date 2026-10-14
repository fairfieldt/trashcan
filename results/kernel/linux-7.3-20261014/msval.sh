#!/bin/bash
# Main-suite validation on 7.3 against the CPU references, and its dumps against the 7.2.9 main suite of the same
# binary (results/qwen4/main-suite-20261013/ms-15e5457, this morning): same binary and inputs, so every dump file
# should be byte-identical unless the kernel changes what the GPUs compute.
# usage: msval.sh TAG    (TAG like ms-15e5457-k73, in ./main-suite/)
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
tag=${1:?usage: msval.sh TAG}
new=$here/main-suite/$tag
old=$repo/results/qwen4/main-suite-20261013/ms-15e5457
REF14=${REF14:-$HOME/dev/claude-work/trashcan-20261006/qwen-flash/ref/g-iq4-cpu}
REF100=${REF100:-$HOME/dev/claude-work/flashnext-20261012/ref-p100}
for v in val14 val100; do
    ref=$REF14; [ $v = val100 ] && ref=$REF100
    echo "== $tag-$v against the CPU reference"
    python3 "$repo/tools/compare_activations.py" "$new/$tag-$v" "$ref" 2>&1 | grep -i "logits\|worst\|greedy"
    n=0 s=0
    for f in "$new/$tag-$v"/*; do
        n=$((n + 1))
        cmp -s "$f" "$old/ms-15e5457-$v/$(basename "$f")" && s=$((s + 1))
    done
    echo "byte-identical to 7.2.9's ms-15e5457-$v: $s of $n files"
done
