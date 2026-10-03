#!/bin/bash
# Recorded incite commands. Run from ~/trashcan with exclusive flock on ~/trashcan-exp/.lock.
# Saved binaries pin each measured phase; none are committed to the repository.
set -euo pipefail
export PATH="$HOME/.cargo/bin:$PATH"
R=results/engine/validated-followup-20261003
M=$HOME/models/gemma-4-12b-qat-gguf/gemma-4-12b-it-qat-q4_0.gguf
D=$HOME/models/gemma-4-12b-assistant/model.safetensors
run() { sudo -n python3 scripts/latency-run.py "$@"; }

# At 82e559f, before editing, save bin/baseline and baseline-source.sha256.
# After adding only norm_fused + norm-bench, force a fresh build and save bin/norm.
# After adding the wide merge and serial-reference check, force a fresh build and save bin/merge.
# Actual builds use rsync -ac, touch engine/build.rs engine/src/*.rs engine/shaders/*,
# then cargo build --release, with logs and SHA256 manifests for each phase.

for phase in baseline norm merge; do
    for rep in 1 2; do
        run "$R/bin/$phase" bench "$M" --pp 512 --tg 256 --batch 512 \
            --profile "$R/$phase-$rep.tsv" > "$R/$phase-$rep.json" 2> "$R/$phase-$rep.log"
    done
done
run "$R/bin/norm" norm-bench > "$R/norm-micro.jsonl" 2> "$R/norm-micro.log"
for rep in 1 2; do
    for tokens in 1 256 768 1536 8192; do
        for phase in baseline merge; do
            run "$R/bin/$phase" attention-bench --tokens "$tokens" --iterations 100 \
                > "$R/attn-$phase-$tokens-$rep.jsonl" 2> "$R/attn-$phase-$tokens-$rep.log"
        done
    done
done
for phase in baseline norm merge; do
    run "$R/bin/$phase" validate "$M" "$R/$phase-b1" --batch 1 --greedy 32 \
        > "$R/$phase-validation.log" 2>&1
    python3 tools/compare_activations.py "$R/$phase-b1" results/reference-vulkan --check \
        >> "$R/$phase-validation.log"
    run "$R/bin/$phase" chat "$M" 'Explain how a transformer works, step by step, in English.' \
        --batch 512 --temperature 0 --max-tokens 128 > "$R/$phase-chat.txt" 2> "$R/$phase-chat.log"
done
cmp "$R/baseline-chat.txt" "$R/norm-chat.txt"
cmp "$R/baseline-chat.txt" "$R/merge-chat.txt"
for phase in baseline merge; do
    run "$R/bin/$phase" bench "$M" --pp 4096 --tg 128 --batch 512 \
        --profile "$R/long-$phase.tsv" > "$R/long-$phase.json" 2> "$R/long-$phase.log"
done
run env PATH="$PATH" TRASHCAN_LONG_CHECK=1 bash scripts/test-engine.sh > "$R/validation-suite.log" 2>&1
run target/release/trashcan check-batches "$M" --tokens 250 --batch 128 --draft "$D" \
    > "$R/boundary-250.json" 2> "$R/boundary-250.log"
python3 "$R/run-chat-parity.py" > "$R/chat-parity.log" 2>&1
python3 "$R/run-mtp.py" > "$R/mtp-parity.log" 2>&1
# clockmon.sh ran for each actual phase; corresponding *-clocks.txt logs are saved.
