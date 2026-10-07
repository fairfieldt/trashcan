#!/bin/bash
# Uses only archived source and the local shader compiler/validator; any cwd.
set -euo pipefail
export PATH="/opt/homebrew/bin:$PATH"
baseline=b0a2e6884d06ac8798df70f7057910ff1ddff3fb
experiment=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
tmp_gdn=$(mktemp -d /tmp/trashcan-gdn-proof.XXXXXX)
trap 'rm -rf "$tmp_gdn"' EXIT
tar -xzf "$experiment/source.tar.gz" -C "$tmp_gdn"
cp "$experiment/candidate-source/qwen_gdn-baseline.comp" "$tmp_gdn/baseline.comp"
cd "$tmp_gdn"
set -x
glslc --version
spirv-val --version
shasum -a 256 "$tmp_gdn/baseline.comp" \
    engine/shaders/common.glsl engine/shaders/qwen_gdn.comp \
    engine/shaders/qwen_gdn_body.glsl engine/shaders/qwen_gdn_math.glsl \
    engine/shaders/qwen_gdn_prepare.comp engine/shaders/qwen_gdn_prepared.comp \
    engine/src/qwen/gdn.rs engine/src/qwen/gpu.rs engine/src/main.rs \
    engine/src/qwen/mod.rs engine/build.rs engine/src/vulkan.rs
glslc --target-env=vulkan1.2 -O -Iengine/shaders "$tmp_gdn/baseline.comp" -o "$tmp_gdn/baseline.spv"
glslc --target-env=vulkan1.2 -O -Iengine/shaders engine/shaders/qwen_gdn.comp -o "$tmp_gdn/default.spv"
glslc --target-env=vulkan1.2 -O -Iengine/shaders engine/shaders/qwen_gdn_prepare.comp -o "$tmp_gdn/prepare.spv"
glslc --target-env=vulkan1.2 -O -Iengine/shaders engine/shaders/qwen_gdn_prepared.comp -o "$tmp_gdn/prepared.spv"
for file in "$tmp_gdn/"*.spv; do spirv-val --target-env vulkan1.2 "$file"; done
cmp "$tmp_gdn/baseline.spv" "$tmp_gdn/default.spv"
shasum -a 256 "$tmp_gdn/"*.spv
printf 'PASS: all four SPIR-V modules validate; default is byte-identical to %s.\n' "$baseline"
