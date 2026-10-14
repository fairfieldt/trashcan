#!/bin/bash
# ISA audit of the kernels that run after light barriers (the engine's SCALAR_SAFE list in engine/src/vulkan.rs) in a
# RADV dump of every pipeline (tools/vmstats.sh output): no scalar loads of anything an earlier dispatch wrote.
# Rerun after any change to these shaders or to Mesa (the dump must come from the Mesa the Macs run).
# usage: audit-decode.sh VULKAN_RS RADV_DUMP > OUT   (exit 1 on a violation or a listed kernel missing from the dump)
here=$(dirname "$0")
python3 "$here/coherence_audit.py" --rule smem "$1" "$2" --scalar-safe
