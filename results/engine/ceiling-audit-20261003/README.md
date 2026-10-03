# Performance ceiling audit — 2026-10-03

This is an arithmetic/source audit, not a new GPU benchmark. It checks the two-D700 target engine
at the merged performance source `ee9bea2d2721c60138c800828ffe24a27c1ee555` and corrects the
peak-performance assumptions in `docs/hardware.md` and `docs/engine-plan.md`.

`accounting.json` records source SHA-256 hashes, exact shape/byte/FLOP totals and sensitivity cases.
`gguf-validation.json` records a CPU-only check of all **667 tensor names, shapes and types** against
incite's actual model header. The file is 6,975,879,296 bytes and has 11,907,350,576 parameters.
The saved header SHA-256 covers 15,823,468 header bytes through the tensor directory; it is not a
hash of the model weights. No GPU command, clock adjustment or model execution was performed for
this audit. `uname -r` on incite returned `6.12.111-p2p`.

## Reproduce

From the repository root:

```sh
python3 tools/performance-ceiling.py \
  --source-ref ee9bea2d2721c60138c800828ffe24a27c1ee555 \
  --profile results/engine/performance-20261003/final/profile.tsv \
  > results/engine/ceiling-audit-20261003/accounting.json
scp tools/performance-ceiling.py incite:/tmp/trashcan-ceiling-audit.py
ssh incite 'python3 /tmp/trashcan-ceiling-audit.py --gguf ~/models/gemma-4-12b-qat-gguf/gemma-4-12b-it-qat-q4_0.gguf' \
  > results/engine/ceiling-audit-20261003/gguf-validation.json
```

The script reads metadata and seeks past tokenizer data; it reads no weight tensors and requires
only Python's standard library. `--prefix`, `--batch`, `--core-mhz`, `--memory-mhz` and `--hop-us`
vary assumptions. A nonmatching tensor directory or layer pattern fails validation rather than
silently applying these formulas to another model. `--source-ref` pins engine-file hashes using
`git show` of a resolved commit, independently of concurrent working-tree edits. It identifies
the audited source; it does not derive or change the formulas. Without it, the report explicitly
labels hashes as a working-tree snapshot, or unavailable when the script runs outside the repository.
The running audit script has its own separate SHA-256 field. The saved local report pins `ee9bea2`;
the remote report validates the GGUF header without claiming an engine-source checkout.

## Exact costs

There are 40 sliding and eight global layers, hidden width 3840, FFN width 15360, 16 Q heads and
262144 vocabulary entries. A local sliding layer has 112,066,560 matrix parameters; a local global
layer has 121,896,960. The global K projection's 1,966,080 parameters are replicated, not halved.

| Per-GPU quantity | Exact value |
| --- | ---: |
| Q4 layer matrix parameters | 5,457,838,080 |
| Repacked Q4 bytes (`parameters × 20/32`) | 3,411,148,800 |
| Local Q6 head parameters (`3840 × 131072`) | 503,316,480 |
| Repacked Q6 head bytes (`parameters × 5/4`) | 629,145,600 |
| Decode matrix bytes, excluding activations/KV/norms | 4,040,294,400 |
| Decode matrix FLOPs (`2 × local matrix parameters`) | 11,922,309,120 |
| Four-column verification matrix FLOPs | 47,689,236,480 |
| FFN gate/up/down bytes across 48 layers | 2,654,208,000 |
| P2P payload per allreduce, per direction, per token | 15,360 |
| P2P payload per pp512 chunk, per direction (`96 × 512 × 15360`) | 754,974,720 |

The repacks expand Q4 from 18 to 20 bytes per 32 parameters and Q6_K from 210 to 320 bytes per
256 parameters. The original, unsharded GGUF matrix payload is **6,956,974,080 bytes**. All tensor
payloads including the small f32 tensors total 6,960,054,464 bytes; metadata/alignment accounts
for the remaining file bytes. File size is not the current engine's per-token weight traffic.

Dense-equivalent work counts a multiply/add pair as two FLOPs. It omits the integer unpacking and
scale multiplies needed by quantization. Attention counts QK and PV dot products only; the actual
online softmax also rescales accumulators, computes exponentials, and reduces within waves.

Ordinary pp512 computes all layers for all 512 inputs but only **one** LM-head column. Its per-GPU
work is 5,588,826,193,920 layer-matrix FLOPs plus 1,006,632,960 head FLOPs and 60,246,982,656
attention-dot-product FLOPs for an empty prefix: **5,650,079,809,536 FLOPs**, or 11.035312128 GFLOP
per input. Verification computes every head column, so using the prefill per-token cost for
four-token verification understated the matrix work (44 versus 47.689 GFLOP).

## Hardware bounds and measured performance

Apple's [D700 specifications](https://support.apple.com/en-la/112025) round the physical peaks to
264 GB/s and 3.5 TFLOP/s. At sustained 850 MHz core / 1370 MHz memory, the formulas are:

```text
memory: 1370e6 × 4 × (384 / 8) = 263,040,000,000 bytes/second/GPU
compute: 850e6 × 2048 × 2 = 3,481,600,000,000 FLOPs/second/GPU
```

These are hardware upper bounds, without refresh/transaction inefficiency or mixed-instruction
cost. Two cards only add their bandwidth when useful work executes concurrently on both.

| Calculation | Result | Interpretation |
| --- | ---: | --- |
| Original matrices / one card's physical bandwidth | 37.81 tok/s | Layer-split weight-only bound |
| Current per-GPU matrix bytes / physical bandwidth | 15.36 ms, 65.10 tok/s | TP weight-only bound |
| Above plus `96 × 42 us` | 19.392 ms, 51.57 tok/s | Historical-hop sensitivity, not a practical ceiling |
| pp512 counted FLOPs / 3.4816 TFLOP/s | 315.50 tok/s | Ideal arithmetic bound, many operations omitted |
| Same pp512 work / best isolated 2.2 TFLOP/s FFN rate | 199.36 tok/s | Optimistic extrapolation before transport/other work |
| Recorded ordinary pp512 | 160.87 tok/s | Complete measured application path |
| Recorded tg128 after pp512 | 22.14 tok/s | Complete measured application path |

The performance-run [README](../performance-20261003/README.md) identifies commands, exact
before/after binaries and sources, correctness checks and sustained clock logs. Its separate
tg256 confirmation reaches 21.88 tok/s. All six ordinary/speculative chat outputs match, and the
1700-token cache-wrap check passes. Those existing measurements establish the comparison baseline;
this audit adds no performance or correctness claim for new kernels.

The retained final-token timestamp sample reports 41.736741/41.770964 ms of GPU segment work and
2.749337/2.652888 ms of gaps on GPU0/GPU1. These are concurrent timelines: **do not add the two
cards' elapsed times**. The final head/selection segment is 2.994667/2.996889 ms. Eliminating all
gaps would improve this sample by only about 6%; it cannot double throughput. A 50 tok/s target
needs substantial improvements to GPU work itself, not simply lower semaphore latency.

Warmed FFN gate and down microbenchmarks count **original** Q4 bytes at 153.00 and 160.41 GB/s.
Their physical-payload equivalents are 170.00 and 178.23 GB/s because the actual block is larger.
The existing 180 GB/s original-byte milestone therefore means 200 GB/s repacked bytes, 76.0% of
the physical peak. Effective weight rates exclude activation traffic and are not DRAM counters.

## Context dependence and highest-value work

| Visible tokens | Unique live K/V payload per GPU | Logical f16 K/V requested across Q heads |
| ---: | ---: | ---: |
| 512 | 88 MiB | 224 MiB |
| 1024 | 176 MiB | 448 MiB |
| 8192 | 288 MiB | 1344 MiB |
| 32768 | 672 MiB | 4416 MiB |

Sliding K/V is read for two Q heads; global K/V is read for eight local Q heads. The shaders issue
separate head workgroups. Cache/coalescing may reuse those reads; the second column is a distinct
payload lower bound and the third is logical requested payload, neither is a measured DRAM total.
At 32K the earlier “0.5 GiB global attention read” was only the unique global cache size; global
head requests alone are 4 GiB, before counting sliding attention. Long-context speed is untested.

Priority hypotheses, to be accepted only after complete-path measurement:

1. FFN Q4 owns 65.7% of matrix payload. Improve streaming, unpacking and occupancy for all three
   projection shapes. Kernel-only speedups must survive the full forward and greedy-output checks.
2. Q6 head owns 15.6% of matrix payload. Attention projection kernels also underperform FFN
   bandwidth. Use the profile to choose between head, narrow projection and attention work.
3. Small-batch verification determines speculative benefit. The previous run's larger draft count
   remained slower than three drafts even after its kernel improved; acceptance and total proposal
   cost must accompany every speculative throughput number.
4. Prefill cannot inherit the best FFN GEMM's efficiency automatically. Improve GEMM and attention
   across complete pp512 and cache-wrap cases. Reusing already processed HTTP prefixes can avoid
   work, but requires independent KV lifetime and sliding-cache correctness validation.

The 40–50 tok/s decode and 1.5–2× speculative targets remain stretch goals. There is no evidence
that 52 tok/s is a sustainable implementation ceiling or that these goals have already been reached.
