#!/bin/bash
# Bit-for-bit comparison of two validation dump directories: tensors in A, how many differ from B (names listed).
# usage: valcmp.sh A_DIR B_DIR
a=$1 b=$2 n=0 d=0
for f in "$a"/*.f32; do
    x=$(basename "$f")
    n=$((n + 1))
    if ! cmp -s "$f" "$b/$x"; then d=$((d + 1)); echo "differs: $x"; fi
done
echo "$(basename "$a") vs $(basename "$b"): $n tensors, $d differ"
