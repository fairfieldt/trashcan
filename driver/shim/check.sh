#!/bin/bash
# Syntax-check imported or shim C files against the shim (run from driver/): shim/check.sh FILE.c ...
cd "$(dirname "$0")/.."
cc=${CC:-x86_64-linux-gnu-gcc}
for f in "$@"; do
	echo "=== $f"
	$cc $(tr '\n' ' ' < shim/cflags.txt) -fsyntax-only "$f" 2>&1 | grep -E "error|warning: (implicit|incompatible)" | sort | uniq | head -${N:-40}
done
