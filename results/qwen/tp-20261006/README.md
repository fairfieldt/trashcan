# Tensor parallelism for Qwen: TP2 on one Mac, TP4 across incite and invite (2026-10-06)

`--tp` splits every layer across GPUs instead of giving each GPU a range of layers (`engine/src/qwen/tp.rs`,
`Split` in `engine/src/qwen/gpu.rs`). With `--tp-stream` or `--tp-connect` the other Mac's GPUs take the later
ranks; `trashcan qwen-tp-node` serves them. Two all-reduces per layer go through host memory and, across Macs,
over the lanes of `engine/src/link.rs` (the design measured in `../tp4-viability-20261006/`).

```sh
# invite
gpuq run ID -- scripts/with-clocks.sh results/clocks/node.txt ./trashcan qwen-tp-node MODEL --stream /dev/tbstream0,/dev/tbstream1,/dev/tbstream2
gpuq run ID -- scripts/with-clocks.sh results/clocks/node.txt ./trashcan qwen-tp-node MODEL --listen 0.0.0.0:7072
# incite
gpuq run ID -- ./trashcan qwen-validate MODEL OUT --tp                      # both local GPUs
gpuq run ID -- ./trashcan qwen-validate MODEL OUT --tp-stream /dev/tbstream0,/dev/tbstream1,/dev/tbstream2
gpuq run ID -- sudo -n python3 scripts/latency-run.py ./trashcan qwen-bench MODEL --tp-stream ... --pp 512 --tg 128
gpuq run ID -- sudo -n python3 scripts/latency-run.py ./trashcan qwen-bench MODEL --tp-connect invite.tb0:7072,invite.tb1:7072,invite.tb2:7072 --pp 512 --tg 128
# the pipeline baseline (invite: ./trashcan qwen-node MODEL --listen 0.0.0.0:7070)
gpuq run ID -- sudo -n python3 scripts/latency-run.py ./trashcan qwen-bench MODEL --nodes invite.tb0:7070 --pp 512 --tg 128
```

Models: `Qwen3.5-9B-trashcan-Q4_0.gguf`, `Qwen3.8-27B-trashcan-Q4_0.gguf` (docs/qwen.md), default context 8192.
USB4STREAM: the three `xmac` streams of `../tp4-viability-20261006/tbstream-up.sh`.

## Correctness (`qwen-validate`, the 14-token reference prompt)

| Run | Against our CPU reference (`qwen-ref`) | Against llama.cpp CPU | Greedy tokens |
|---|---|---|---|
| 9B, TP2 on incite | worst 3.74e-4 over 474 tensors; logits 2.35e-4, top-10 10/10 | logits 2.14e-2 | 10/10 identical |
| 9B, TP4 across both Macs | worst 3.75e-4; logits 2.32e-4, top-10 10/10 | logits 2.14e-2 | 10/10 |
| 27B, TP4 across both Macs | worst 4.11e-4 over 946 tensors; logits 4.11e-4, top-10 10/10 | logits 3.28e-2 (the pipeline bring-up: 3.30e-2) | 10/10 |

TP2 against the pipeline GPU path (9B): 3.0e-5; TP4 against TP2: 3.2e-5. llama.cpp's CPU path quantizes
activations to Q8, hence its 2-3e-2 against every GPU path. Comparisons: `*-vs-*.txt` (the dumps themselves are
in `~/dev/claude-work/trashcan-20261006/tp/` on the Mac).

## Speed (`qwen-bench`, pp512 then tg128)

| Model, layout | pp512 tok/s | tg128 tok/s |
|---|---|---|
| 9B, pipeline over incite's two GPUs (bring-up, schedutil) | 127.2 | 22.8 |
| 9B, TP2 on incite, a `vkQueueSubmit` per segment | 207.5, 210.3 | 30.65, 30.66 |
| 9B, TP2 on incite, segments submitted ahead (default) | 208.9, 208.0 | 31.95, 31.98 |
| 9B, TP4 across both Macs, USB4STREAM × 3 (schedutil) | 203.0 | 28.98 |
| **27B, pipeline over both Macs' four GPUs (PP4)** | **38.0, 38.0** | **9.58, 9.59** |
| **27B, TP4 across both Macs, USB4STREAM × 3** | **91.1, 91.8** | **16.58, 16.57** |
| 27B, TP4 across both Macs, TCP × 3 | 81.9 | 15.63 |

Performance governor and a 0 µs CPU latency request on both Macs unless noted (restored to schedutil afterwards);
GPU clocks 850/1370 MHz while computing (`incite/clocks`, `invite/clocks`; the low samples are model loading and
idle time). PP4 is the existing chain (`--nodes`, plan `[(0,18),(18,35)] [(35,53),(53,64)]`); TP4 uses 4,276 MiB per
GPU, PP4 4,510 and 4,277 MiB on incite's.

For Qwen3.8-27B on two Macs, TP4 decodes 1.73× and prefills 2.4× faster than the pipeline.

### With MTP drafting (`--mtp`, two drafts from a 64K-token draft head)

| Model, layout | pp512 tok/s | tg128 tok/s | drafts accepted | tokens per round | verify / draft ms per round |
|---|---|---|---|---|---|
| **27B, TP4 across both Macs, USB4STREAM × 3** | **89.5, 89.2** | **26.08, 25.93** | 72% | 2.44 | 86 / 7.8 |

Against 16.6 tok/s without drafting and 9.6 for the pipeline without drafting: 2.7× the pipeline. The MTP block is
split like a trunk layer (its eh projection replicated); a draft step is three segments with two all-reduces, and
extending its KV cache inside a target batch needs no extra all-reduce because only its K and V are kept.

`qwen-check --mtp` (600-token prompt, 256 steps; schedutil): 9B TP2 and 27B TP4 both pass: restore bit for bit, 170
verification rounds with rollbacks and no column mismatch (score error 0), speculative output identical to greedy
decoding (9B: 65% accepted, 40.4 against 24.5 tok/s; 27B: 68.5%, 21.4 against 13.4). Without `--mtp` (verify 3,
128 steps): both pass with 85 rounds. `*-check.json`.

### Where a decode token goes (ms, `tg_tp_ms` in the JSON)

| | submit | wait for GPUs | add partials | exchange | total |
|---|---|---|---|---|---|
| 27B TP4, USB4STREAM | 0.94-1.03 | 45.86-45.91 | 0.69-0.80 | 11.98-11.99 | 60.3 |
| 27B TP4, TCP | 1.27 | 46.05 | 0.85 | 15.14 | 64.0 |
| 9B TP2, submit per segment | 4.60-4.71 | 27.44-27.55 | 0.29 | | 32.6 |
| 9B TP2, submitted ahead | 0.42 | 30.43-30.45 | 0.25 | | 31.3 |
| 9B TP2, no all-reduces (`TRASHCAN_TP_FREE`, wrong results: the GPU floor) | 0.07 | 27.32-27.35 | | | 27.8 |

- Synchronisation is not the main cost. Without all-reduces the 9B TP2 token still takes 27.3 ms on the GPUs, about
  102 GB/s per GPU against 123 GB/s for a pipeline stage: narrower per-rank matrices, and a cache flush at each of the
  65 (9B) or 129 (27B) submission boundaries.
- Submitting segments ahead and releasing them with a host-signalled timeline semaphore saves 4.3 ms of
  `vkQueueSubmit` per 9B token but adds 3 ms of release latency (RADV's submission thread): +4%.
- The 27B's 128 cross-Mac exchanges take 94 µs each in the model against 34 µs in the benchmark: the exchange
  includes waiting for the slower Mac.

`scripts/test-server.py` passes against `trashcan serve 27B --tp-stream ... --mtp ...` with `qwen-tp-node --mtp` on
invite (`test-server-27b-tp4-mtp.txt`, server log `incite/serve-27b.log`).

## Not yet

Prefill all-reduces are still the naive host path (about 21.6 ms per 512-token all-reduce in the benchmark); GPU
efficiency at the per-rank shapes; USB4STREAM setup and device permissions outside `tbstream-up.sh`.
