# Offline prefill candidate, October 7, 2026

`TRASHCAN_QWEN_GDN=prepared` selects a preparation pass before the Gated DeltaNet
recurrence for batches above eight tokens. The default remains unchanged. No Mac
was contacted and no GPU run or measured D700 speedup is reported here.

For Qwen3.8-27B TP4, each rank has four key heads and twelve value heads. The
existing recurrence normalizes each key head's Q/K in 96 waves per token (32
state-row waves per value head, three value heads per key head) and evaluates
each value head's gates in 32 waves. The candidate normalizes Q/K once per key
head and computes decay/beta once per value head, using the existing `conv_out`
and `ab` scratch buffers. TP queues retain separate scratch slices. Ordinary
dispatch barriers publish the preparation writes before the recurrence reads
them. The recurrent state, convolution state and snapshots retain their layouts.

The tradeoff is an added dispatch, barrier and scratch read/write traffic.
Target-hardware measurements are needed to determine whether this saves time.
Decode and verification retain their original path, and Qwen4 continues to use
the original GDN shader.

## Local evidence

Run from the repository root:

```sh
bash results/qwen/offline-20261007/prefill/verify-shaders.sh \
  > results/qwen/offline-20261007/prefill/shader-validation.log 2>&1
PATH=/opt/homebrew/bin:$PATH cargo test --release preparation_ \
  > results/qwen/offline-20261007/prefill/cpu-tests.log 2>&1
```

`shader-validation.log` records compiler/validator versions, exact commands,
source SHA256 hashes and compiled shader SHA256 hashes. The baseline shader is
retrieved from `b0a2e6884d06ac8798df70f7057910ff1ddff3fb`. All compiled modules pass
Vulkan 1.2 SPIR-V validation. The default shader is byte-identical to that
baseline, with SHA256
`0f15f97b89631a744bf08e8c196812d285f54ee26751c93ab087bc84e483ff3b`.

Two CPU tests check preparation write ownership (including partial workgroups),
preservation of value channels, and exact f32-model output and state equivalence.
They cover TP head mappings, zero/tiny norms, two epsilon values, softplus
threshold/extreme gates, nonzero initial state, fresh state and uneven
microbatch continuation. The model uses a fixed f32 reduction tree and operation
order; it does not establish Vulkan subgroup reduction, transcendental or GPU
output equivalence.

## Pending GPU checks

After the user confirms the Macs are available and the recovery checklist in
`results/qwen/prefill-slab-20261007/README.md` is complete, run under fresh paired
gpuq leases and `scripts/with-clocks.sh`:

```sh
trashcan qwen-gdn-bench --n 256 --k-heads 4 --v-heads 12 --position 0 --iterations 10
trashcan qwen-gdn-bench --n 256 --k-heads 4 --v-heads 12 --position 4096 --iterations 10
```

Also cover `--n 9`, `64`, `129`, `257` and `--k-heads 1 --v-heads 3`.
The harness compares every output and final-state element against the original
shader, reports bitwise equality and maximum errors, and rejects nonfinite
results or errors beyond `2e-5 + 2e-5 * abs(reference)`. It restores inputs/state
before every replay. GPU timestamp timing excludes restoration and includes
the full preparation pass, its barrier and the recurrence; each replay is a
separate submission. These commands have not been run on the GPUs.

If those checks pass, run model-level validation/checkpoint/rollback checks and
paired TP4 prefill benchmarks with `TRASHCAN_QWEN_GDN=prepared` on both nodes.
Keep all other switches fixed, capture all four GPUs' sustained clocks, and
retain the candidate only if controlled measurements support it.
