# Flash-Next prefill: the two GPUs of a Mac work on consecutive chunks at once (2026-10-12)

**Result.** Six-Mac prefill of real text (`qwen4 bench --tokens-file long.ids`, 128-token chunks) goes from 209.9 to
323.7 tok/s on 2048 tokens (9.76 to 6.33 s, 1.54x), 218.4 to 363.9 tok/s on 3000 tokens (13.74 to 8.24 s, 1.67x) and
132.5 to 157.8 tok/s on 512 tokens (3.86 to 3.25 s, 1.19x). Decode is unchanged (14.67 against 14.69 tok/s after 512
tokens, 13.35 against 13.37 after 2048). The chunk results are bit-identical to main's: greedy tokens, logits and all
583 dumped tensors, with 3 and 5 chunks per prompt.

## The change

Before, a Mac took a chunk, ran GPU0's four layers and then GPU1's (`GpuStage::forward` submits a stage's per-layer
command buffers and waits), sent the residual on and only then read the next chunk. One GPU of a Mac worked at a
time, ~200 ms each per 128-token chunk, and while the GPUs worked nothing read the socket, so once the socket buffers
were full the previous Mac's send of the next chunk (5 MiB) waited for this Mac's GPUs.

Now the stages of a Mac form a pipeline: GPU0 works on chunk k+1 while GPU1 works on chunk k.

- `gpu.rs`: `GpuStage::forward` is `submit` (write the input, submit the layers) plus `collect` (wait, read the
  output). Same command buffers, same buffers; a stage holds at most one chunk.
- `pipeline.rs`: `Local::push` starts a chunk on the first stage without waiting for it. Each stage in turn waits for
  its chunk, starts the next one at once and hands its output to the next stage; the chunk that leaves the last stage
  is returned. `drain` waits for the chunks in flight and `finish` runs a chunk behind them and returns its output
  (the earlier ones are passed on as they leave). When nothing is in flight (decode), `finish` is the old `forward`.
  A CPU machine (one stage) runs synchronously.
- `net.rs`: prompt chunks without a reply go through `push` on the driver (`Chain::forward`) and on every node
  (`session`), and leave for the next Mac as soon as they are out; the chunk with the reply goes through `finish`; a
  chunk with captures (`validate`) waits for the others and runs alone. A thread per upstream connection reads
  requests into a channel (4 deep), so the previous Mac never waits for this one's GPUs to send. The reply protocol
  (`h` first, then the merged candidates) is main's.

In the bench JSON the driver's per-layer GPU times of the last prefill chunk are the same with both GPUs busy as
with one (pp2048: 240.9 and 236.2 ms for its two stages against 240.5 and 236.0 on main).

## Tested source

This commit on branch `worktree-agent-adf3537cc78b259ba`, rebased on main 6d48a71 (the parallel head slices).
Binaries cross-built with `BUILD_ONLY=1 scripts/cross-build.sh`, in `~/trashcan-exp/flashnext/` on all six Macs:

| variant | source | binary | md5 |
|---|---|---|---|
| main | 6d48a71 | `trashcan-main` | 730e3ee9f515e2ca98dde79f1e1328f7 |
| ovl | this commit (engine as built) | `trashcan-ovl` | 46d493735f4464fdb2ee78bb8a61a511 |

## Commands

One six-Mac session under `chain6.sh` (all leases at once), [tools/ab-ovl.sh](tools/ab-ovl.sh): validation (it stops
before the benches if an ovl run fails or differs from main), two alternating rounds of pp2048 and pp512 benches, one
pp3000 pair, then the server test with ovl. Every run is a `tools/q4run.py` run from
[../chain-20261012](../chain-20261012/README.md) (nodes under gpuq leases, clocks logged, 0 us CPU latency request):

```sh
chain6.sh "ovl prefill overlap val+A/B" 45 "Flash-Next GPU overlap agent" -- tools/ab-ovl.sh
# per run, e.g.
q4run.py --binary trashcan-ovl --name ovl-pp2048-ovl-1 --driver "bench --gpus 0,1 \
    --nodes indite.tb0:7071,inflight.tb1:7071,alight.tb0:7071,invite.tb1:7071,enlight.tb0:7071 \
    --head-split 12 --context 32768 --ngram disk --batch 128 --pp 2048 --tg 32 \
    --tokens-file /home/fairfieldt/trashcan-exp/flashnext/long.ids"
```

`ovl-p300.ids` is the first 300 ids of `long.ids`.

## Correctness

| check | result |
|---|---|
| six Macs, `validate --tokens-file ovl-p300.ids --batch 128` (chunks of 128, 128, 44), ovl against main | greedy 21 15 248046 198 248045 74455 198 248068 198 760 for both; logits and all 583 tensors rel 0 ([compare-val300-b128-vs-main.txt](compare-val300-b128-vs-main.txt)) |
| the same with `--batch 64` (64 x 4 + 44) | the same tokens; logits and all 583 tensors rel 0 ([compare-val300-b64-vs-main.txt](compare-val300-b64-vs-main.txt)) |
| six Macs, 14-token reference prompt (one chunk), against the CPU reference `qwen-flash/ref/g-iq4-cpu` | 9419 0 2500 628 353 1438 488 3242 30 248046, the reference's tokens; logits rel 2.84e-4, top-10 10/10, as before ([compare-val14-vs-ref.txt](compare-val14-vs-ref.txt); `Qcur-*` rel 0.94 is the 1/16 query scale) |
| CPU chain on the Mac (driver and two CPU nodes on 127.0.0.1, `--cuts 16,32 --head-split 3`, [tools/cpu-chain.sh](tools/cpu-chain.sh)), chunks of 4 and of 5 | logits and every tensor rel 0 against the CPU reference, greedy 9419 0 2500 628 ([compare-cpu-chain-b4-vs-ref.txt](compare-cpu-chain-b4-vs-ref.txt), [compare-cpu-chain-b5-vs-ref.txt](compare-cpu-chain-b5-vs-ref.txt)); `indexer_score-*` print rel inf for every dump (masked entries) |
| server (`tools/q4serve-test.sh` as a `--raw` driver with ovl, `--head-split 12 --context 16384 --batch 128 --ngram disk`, [incite/ovl-serve.json](incite/ovl-serve.json)) | `q4-turns.py` turn 2 reuses 78 of 99 prompt tokens (first token after 0.65 s, 64 tokens in 4.27 s); every `scripts/test-server.py` check passes |

## Speed

`qwen4 bench --tokens-file long.ids --head-split 12 --context 32768 --batch 128 --ngram disk`, alternating
([summary.txt](summary.txt), from [tools/summarize.py](tools/summarize.py); driver outputs in `incite/`, node logs and
clock logs per Mac):

| prompt | main, prefill | ovl, prefill | main, decode | ovl, decode |
|---|---|---|---|---|
| 512 (tg 64) | 133.0, 132.1 tok/s (3.86 s) | 158.0, 157.5 tok/s (3.25 s) | 14.73, 14.61 tok/s | 14.63, 14.75 tok/s |
| 2048 (tg 32) | 209.4, 210.3 tok/s (9.76 s) | 323.7, 323.7 tok/s (6.33 s) | 13.32, 13.37 tok/s | 13.28, 13.45 tok/s |
| 3000 (tg 32) | 218.4 tok/s (13.74 s) | 363.9 tok/s (8.24 s) | 13.35 tok/s | 13.28 tok/s |

Clocks: in every run of both binaries, every GPU of every Mac was sampled at 850/1370 MHz from the moment the first
chunk reached its Mac to the end of the prefill; the 300/150 samples in the prefill windows of `summary.txt` are Macs the first chunk had not reached
yet (the engine raises the level on a Mac's first work).

## Where the time goes now

With every GPU a pipeline stage taking about the same time per chunk, a prefill takes about the sum of the chunks'
stage times plus 11 times the largest one (the last chunk still crosses the 11 stages after the first). The
measurements fit: 2048 tokens, 16 chunks at 205-240 ms (QSA attention grows with the position): ~3.6 s + 11 x 0.24 s
= 6.2 s against 6.33; 3000 tokens ~8.4 s against 8.24; 512 tokens 4 x 0.21 + 11 x 0.21 = 3.1 s against 3.25. Long
prompts therefore run at one chunk per ~0.23 s (~550 tok/s), and the ~2.5 s pipeline fill decides short ones.
Smaller chunks would shorten the fill (estimated from the stage times in
[../prefill-batch-20261012](../prefill-batch-20261012/README.md), 118 ms per 4 layers at 64 tokens and 69 ms at 32:
512 tokens in ~2.2 s at 64 and ~1.9 s at 32, against 3.25 s now; not measured).

## Files

- `tools/ab-ovl.sh` (the session), `tools/cpu-chain.sh`, `tools/summarize.py`.
- `incite/`: driver outputs (`ovl-*.json`, `ovl-*.err`, `ovl-serve-server.log`), `commands.log`, `clocks/`.
- `indite/`, `inflight/`, `alight/`, `invite/`, `enlight/`: node logs and clocks.
- `compare-*.txt`: dump comparisons (the dumps themselves, ~21 MB each, are not kept).
