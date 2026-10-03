# MTP verification optimization and exploration — 2026-10-03

This round starts from committed `013e343` on main and independently checks the
recovered verification leads on incite's two D700s. The source, pinned binary
hashes, commands, correctness checks and clock samples are saved here. The
pre-existing `investigation-20261003/` directory was preserved.

## Application result

Two 128-token runs per prompt and phase, temperature zero, batch512, the
unchanged three-proposal policy. Rates pool tokens divided by measured time;
chat logs round individual rates to two decimals. All ordinary/MTP output bytes
match across phases, and accepted/proposed/round counts are unchanged.

| Prompt | Baseline MTP, tok/s | Candidate MTP, tok/s | Accepted/proposed per run | Rounds |
|---|---:|---:|---:|---:|
| Transformer | 31.24 | 39.81 | 82/138 | 46 |
| Photosynthesis | 26.65 | 33.14 | 72/165 | 55 |
| CSV function | 39.97 | 50.55 | 90/111 | 37 |
| Pooled | **31.73** | **39.96** | | |

MTP improves **25.9%**. Mean verification latency, weighted by rounds, falls
from **75.77 to 57.87 ms (-23.6%)**. The pooled MTP/ordinary ratio rises from
**1.035x to 1.306x**. M5's typical-chat 1.5x goal remains open; one favorable
prompt above 50 tok/s does not establish that goal.

Ordinary performance is unchanged within variation. The fixed pp512/tg256
benchmark pools to **160.76/28.88 -> 162.08/28.99 tok/s**. Ordinary chats pool
to **30.67 -> 30.60 tok/s**. The one-column kernels, GEMM and attention kernels
are unchanged. These are measured workload rates, not hardware upper bounds.

The last four-column transformer verification traces agree with the latency
change. Mean per-GPU times over two runs and both GPUs are:

| Work | Baseline, ms | Candidate, ms |
|---|---:|---:|
| Attention segments, including projections | 21.87 | 17.12 |
| FFN segments | 41.71 | 32.32 |
| Final norm/head/selection segment | 6.78 | 3.74 |
| Total GPU work | **70.37** | **53.18** |
| Semaphore gaps | 3.40 | 2.63 |

The GPUs run concurrently; their times must not be added. These traces sample
the last forward rather than the whole run. All transformer rounds have three
proposals, so the final saved forward is four columns. The other two prompts'
final traces include their one-token tail and are not four-column evidence.

## Implementation and micro checks

`multivec_body.glsl` issues independent packed-weight, scale and activation
loads before dequantization and accumulation. It keeps the original dot
products, per-lane FMA chains and subgroup reduction order. Two- and
three-column variants remove unused accumulators and conditional work. Padded
columns read a valid clamped input and never store outside the batch.

The engine selects those specializations for two/three columns, the revised
four-column kernel for Q4 and Q6 batches four through eight. Weight formats, sampler,
draft policy, attention partition thresholds and synchronization are unchanged.

The original four-column kernel is retained as `multivec4_reference.comp` for
the benchmark. No recovered arithmetic-changing FMA chain, unbarriered LDS
reduction or event relay was adopted.

`matrix-micro.jsonl` contains **392 passing records**: seven real model matrix
shapes, batches two through eight, two GPUs, two repetitions and two kernels.
The second repetition reverses kernel order. All 196 candidate comparisons
match every output bit of the original kernel. Each record also checks three
rows per column against an f64 dot product. Inputs use a deterministic xorshift
generator and mixed scales 0.001/0.1/1/10, rather than exactly representable
fraction-only inputs. Each command owns one live weight allocation per Device,
avoiding the recovered harness's recycled-buffer descriptor bug.

All 49 paired shape/batch cases improve. Means over both GPUs and repetitions:

| Matrix | Columns | Original, us | Candidate, us |
|---|---:|---:|---:|
| FFN gate | 2 | 223.34 | 134.49 |
| FFN gate | 3 | 256.84 | 173.43 |
| FFN gate | 4 | 293.06 | 221.48 |
| Vocabulary head | 2 | 4877.65 | 2929.75 |
| Vocabulary head | 3 | 5794.73 | 3051.68 |
| Vocabulary head | 4 | 6715.78 | 3681.08 |

Warmup is 500 ms before 100 timed dispatches. A second paired head sweep compares
against the previously selected eight-column kernel at batches five through eight,
with reversed order, two repetitions and both GPUs. Times are 8.563/9.765/11.050/12.422
ms before and 7.212/7.247/7.282/7.325 ms after (15.8–41.0% faster). This supports
using the revised four-column kernel for those head batches too. Live depth-four
chat rates remain within variation: transformer 27.62 -> 27.77, photosynthesis
23.05 -> 22.99, CSV 35.37 -> 35.50 tok/s. The final default-depth-three transformer
confirmation is byte-identical and uses the same kernels as the initial timing.

## Further exploration

Two repetitions of each greedy depth preserve all output bytes. Pooled tok/s:

| Prompt | Depth 1 | Depth 2 | Depth 3 | Depth 4, final head |
|---|---:|---:|---:|---:|
| Transformer | 41.07 | 42.58 | 39.81 | 27.77 |
| Photosynthesis | 37.69 | 36.15 | 33.14 | 22.99 |
| CSV | 44.26 | 47.22 | 50.55 | 35.50 |

Depth one/two can help these explanation prompts, while depth three wins CSV.
Three prompts do not justify changing the default policy; `--draft-tokens`
already lets callers choose. Depth-four includes partially filled four-column
tiles and illustrates why micro throughput alone does not predict chat speed.

Global decode attention at 8192/80000 keys measures 0.502/4.385 ms with the
selected shared-head 64-partition path versus 1.237/11.747 ms with the existing
scalar 128-partition fallback. All CPU and merge-bit checks pass. Keep the
selector; a shared-head 128-partition kernel is a separate unmeasured lead.
The dim256/80K micro is synthetic: production sliding layers cap visible keys
at 1024. No draft policy or attention partition selector change was made.

Other reviewed leads remain separate from this accepted change:

- Packed six-bit head weights reduce the recovered one-column micro from
  about 2.90 to 2.25 ms and free about 202.5 MiB/GPU in the tight format. They
  still need matching multi-column kernels and live application validation.
- Q4 interleaving and f16 scale streams show roughly 7–14% gains in recovered
  gate/down micros. Some variants reformulate arithmetic; current-build
  activation and application checks are needed before replacing the layout.
- Tiled prefill is the largest structural lead for the measured 80K workload.
  The recovered prototype allocates a full score matrix, about **1.22 GiB/GPU
  at batch512/context80K**. The 80K run used up to 5751 MiB of each GPU's 6144
  MiB, so that prototype does not fit the spare VRAM. A bounded query slab that reuses a score buffer is the next experiment.
- Short-context partition thresholds were tuned before the faster merge.
  The preceding sweep favored 32 over 16 partitions at 256 keys, but changing
  that boundary still needs batch-specific application and continuation checks.
- The recovered platform and scheduling variants did not establish a reliable
  gain. Visible-heap/command-pool pressure remains a source of timing variation;
  removing dependency barriers and restarting event-relay experiments are not
  supported by the existing evidence.

## Correctness, clocks and reproduction

Fresh baseline/candidate batches two, three and four each compare all **133
activation/logit files and 16 greedy tokens byte for byte**. Seven-proposal
validation exercises the final eight-column head and preserves 133 files plus
64 greedy tokens. Seed17/temperature0.8 outputs match for depth three and for
the final depth-four head change. Target-only 1700-token batch512 ring wrap
preserves all 64 continuation tokens (relative logits 0.0012545, top64 64/64).

The final binary passes 30 CPU tests, tokenizer/chat fixtures, 10,000 changing
bidirectional P2P rounds, six independent 132-tensor/logit/greedy reference
paths, and the 1700-token assistant rollback check. The saved 250-token batch128
boundary check also passes. Both GPUs sustain 850/1370 MHz during compute;
complete sampling counts and temperatures are in `summary.json`.

All timed commands use the scoped 0-us CPU latency request and hold the shared
`~/trashcan-exp/.lock`. `*-clocks.txt` includes loading and idle intervals as
well as compute; timed matrix kernels warm clocks before timestamp collection.
Final clock and reset statistics are recorded with the acceptance results.

The baseline's 40 engine files match Git content at `013e343`. The timed
candidate and intermediate load-first build each have 43-file manifests,
reconstructed with the saved `micro.rs` and `model.rs` snapshots. The final
build changes the head selector for batches five through eight and corrects
benchmark default selection/GEMM checking. The final 43-file source manifest
matches the committed source. All stages retain their binary hashes; final
acceptance reruns on the head-selector build.

| Phase | Binary SHA256 |
|---|---|
| Baseline | `b73b45e585402b2fd65233c5f6e807e3cc09b3bdc605be6412162efa80acb930` |
| Timed candidate | `b8f3116e7890834a6b8847e4dda8269e9aee69e46a9132bdf60a88a8711129eb` |
| Intermediate depth/partition exploration | `35ffd4b81c626f1de4d862bad0ffa4d7d8e8c03ce7f220e533deb2444e8a75fe` |
| Final head selector and acceptance | `7a636984471a4a4dabf005a177a700104d758d2ea26abe440f098450c5287efb` |

Sources were copied with `rsync -ac` and touched before builds, avoiding the
recovered stale-mtime hazard. Pinned binaries remain on incite in this result
directory's `bin/` and are not committed. [commands.sh](commands.sh) and the
four `run-*.py` files record execution; [summarize.py](summarize.py) rebuilds
`summary.json` and checks the completed evidence.
