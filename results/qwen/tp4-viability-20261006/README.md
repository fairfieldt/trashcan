# Is 4-way tensor parallelism across incite and invite viable? (2026-10-06)

Qwen3.8-27B needs both Macs. Three layouts fit:

- **PP4**: each GPU a pipeline stage. Runs today.
- **TP2×PP2**: tensor parallelism inside each Mac, two pipeline stages over Thunderbolt.
- **TP4**: each GPU holds a quarter of every layer, so a token needs 128 all-reduces (two per layer) across
  the Macs.

TP2×PP2 and TP4 both stream 4.33 GB per GPU per token. In TP2×PP2 one Mac's GPUs stream while the other's
wait; in TP4 all four stream at once, and every all-reduce crosses the cable. TP4 wins if a cross-Mac all-reduce
is cheap enough. This directory measures that cost with `trashcan allreduce-bench`
(`engine/src/allreduce.rs`, lanes in `engine/src/link.rs`).

## Answer

Yes for decode. With every GPU streaming 32 MiB of VRAM per segment, about a TP4 slice of the 27B (4.33 GB per GPU
per token over 128 segments), a token of 128 cross-Mac all-reduces takes 37.3 ms over USB4STREAM and 41.6 ms
over TCP. The GPU work alone is 24 ms (128 × 188 µs), so a cross-Mac all-reduce costs about 104 µs over
USB4STREAM and 138 µs over TCP. Every element of every all-reduce total was checked on every GPU: 0 mismatches.

| Layout, 27B decode (per token) | Weights-only bound, 263 GB/s per GPU | At the benchmark's streaming rate (178 GB/s) | At today's Qwen decode rate (127 GB/s) |
|---|---|---|---|
| PP4 | 66 ms (15 tok/s) | 97 ms + 4 hops (10 tok/s) | 136 ms (7 tok/s) |
| TP2×PP2 (P2P all-reduces ~30 µs, two Macs in turn) | 33 ms (30 tok/s) | about 52 ms (19 tok/s) | about 72 ms (14 tok/s) |
| TP4 (measured all-reduce cost, USB4STREAM) | 16.4 ms (61 tok/s) | **37.3 ms measured (27 tok/s)** | about 47 ms (21 tok/s) |

Only the bold cell is a measurement; the others are bounds or projections from this benchmark's per-segment costs
and the 9B's measured decode rate (`docs/qwen.md`). MTP drafting added about 50% on the 9B and would apply on top.
Real TP4 kernels also replicate the norms and run on narrower matrices, so their efficiency still has to be
measured.

Prefill is not viable yet as written: a 512-token all-reduce (10 MiB per partial) costs 21.6 ms, so about 2.8 s per
chunk against an estimated 3-3.6 s of compute. The pieces and their fixes are below.

## Setup

- incite and invite, Linux `7.2.9-p2p` (rev 2), three Thunderbolt 2 cables, one per controller pair, addressed by
  tblink (`invite.tb0..2`). `thunderbolt_net` at `throttling=32000`.
- USB4STREAM: one stream per cable (`tbstream-up.sh incite invite xmac`), default ring of 256 frames and 8 µs
  throttle; `/dev/tbstream0..2` are buses 0..2 on both Macs. `chmod 666` on the devices for the runs.
- Binary cross-built from this commit (`HOSTS="incite invite" scripts/cross-build.sh trashcan-exp/xmac/trashcan`).
- GPU runs inside gpuq leases on both Macs (`xrun.sh`: clock log plus a 0 µs CPU latency request); the first
  host-only TCP runs (`host-decode`, `host-size*`) ran under the shared lock before gpuq covered invite. schedutil
  with turbo, except `host-size-perf.jsonl` (performance governor, restored afterwards).

## The benchmark

Each GPU runs `--steps` segments per token, each one submission: check the previous total, stream `--work-mib` of
VRAM, write this rank's partial (5120 × `--cols` floats) to host memory. The host waits for its two GPUs (`--notify
wait|poll`), adds their partials, exchanges the sum with the other Mac over 1-3 lanes, writes the total for both
GPUs and releases the next segment (`--release submit`: submit it; `signal`: it was submitted waiting on a timeline
semaphore). The next segment checks every element of the total against the sum of all ranks' patterns, and the CPU
checks each token's last total. A GCN 1.0 shader cannot wait for memory written outside its GPU
(`docs/findings.md`, 2026-10-02), so every all-reduce is a submission boundary.

```sh
# invite (serves configurations sent by incite)
gpuq run ID -- ~/trashcan-exp/xmac/xrun.sh ./trashcan allreduce-bench --stream /dev/tbstream0,/dev/tbstream1,/dev/tbstream2 --serve
gpuq run ID -- ~/trashcan-exp/xmac/xrun.sh ./trashcan allreduce-bench --listen 0.0.0.0:7071
# incite
gpuq run ID -- ~/trashcan-exp/xmac/xrun.sh ./trashcan allreduce-bench --stream /dev/tbstream0,/dev/tbstream1,/dev/tbstream2 --plan plan-decode.jsonl
gpuq run ID -- ~/trashcan-exp/xmac/xrun.sh ./trashcan allreduce-bench --connect invite.tb0:7071,invite.tb1:7071,invite.tb2:7071 --plan plan-decode.jsonl
gpuq run ID -- ~/trashcan-exp/xmac/xrun.sh ./trashcan allreduce-bench --plan plan-decode.jsonl   # one Mac, no lanes
```

21 tokens per configuration, the first not counted; medians over 2,560 steps.

## Decode: 128 all-reduces of 20 KiB per token (`incite/*-decode.jsonl`, `incite/stream*-lanes.jsonl`)

ms per token, 32 MiB streamed per segment (exec 187-189 µs per segment):

| | submit/wait | submit/poll | signal/wait | signal/poll |
|---|---|---|---|---|
| one Mac, two GPUs, no network | 34.59 | 33.76 | 32.59 | 32.05 |
| two Macs, USB4STREAM × 3 | 43.26 | 38.76 | **37.32** | 39.42 |
| two Macs, USB4STREAM × 2 | | 38.95 | 37.46 | 36.65 |
| two Macs, USB4STREAM × 1 | | 40.63 | 38.78 | 37.08 |
| two Macs, TCP × 3 | 49.36 | 48.35 | 43.05 | 41.60 |

Without streamed work (pure synchronisation, exec 14-15 µs): 9.42 ms per token on one Mac (signal/poll), 12.49
over USB4STREAM, 18.04 over TCP.

Per step, USB4STREAM × 3, signal/wait (µs, medians): 188 GPU execution, 53 release-to-start plus completion
notice, 3 local sum, 34 exchange (including waiting for the other Mac), 10 total write and release. With `submit`,
the release-to-notice part drops to 15 µs but two `vkQueueSubmit` calls take 60 µs; submitting from one thread per
GPU is the obvious next step.

The cable count barely matters for decode (37.1, 36.7 and 37.3 ms on 1, 2, 3 streams): skew between the Macs'
GPUs dominates the exchange. More cables matter for prefill bandwidth and for staying up when one drops.

`free` (all 128 segments submitted at once, no host in between) is not a clean GPU baseline: 128 submissions per
GPU, one GPU after the other, take longer than the work. Use 128 × exec instead.

## Host-only exchanges between the Macs (`incite/host-*.jsonl`, `incite/stream3-host.jsonl`)

Median µs for one symmetric exchange (both Macs send and receive the size below):

| bytes | TCP × 1 | TCP × 3 | USB4STREAM × 1 | USB4STREAM × 3 |
|---|---|---|---|---|
| 64 | 38.4 (32.5 performance governor) | 37.7 (performance) | | 13.5 |
| 4,096 | 58.4 | 54.1 (performance) | | 15.8 |
| 10,240 | | | | 19.0 |
| 20,480 | 77-116 (84 at MTU 65522) | 71-94 | 34.4 | 21.8 |
| 10 MiB | | | | 6,944 |

A single-threaded Python ping-pong over one stream (`tools/pingpong.py`) gives round trips of 23.4 µs (16 B),
32.0 µs (4 KiB) and 67.2 µs (20 KiB).

## Prefill: 512 columns, 10 MiB per partial (`incite/stream3-prefill.jsonl`)

21.6 ms per all-reduce over USB4STREAM × 3, no streamed work: 7.1 ms for each GPU to write 10 MiB to host memory
(shader stores into snooped memory run at about 1.5 GB/s, `docs/findings.md`), 4.0 ms to add two partials on one
CPU core, 7.4 ms to exchange 10 MiB each way, 3.2 ms to copy the total into both GPUs' buffers. Fixes, roughly in
order of value: copy partials out with the SDMA engine (12.7 GB/s measured), receive straight into memory both GPUs
read, add on several cores or reduce on the GPUs over P2P, send f16 partials. About 4 ms per all-reduce would make
TP4 prefill competitive with TP2×PP2's pipelined prefill.

## USB4STREAM between two Macs

- Patch 0005 removed Tx flow control (the Falcon Ridge NHI wedges with it), so nothing stops a sender from
  overrunning the receiver's 256-frame ring. The first 10 MiB exchange, from one polling thread on three streams,
  logged six `tbstream: RX CRC error` on invite and lost data (the exchange stalled). Exchanges now move at most
  256 KiB per stream per round, so less than the ring is ever in flight: no CRC errors and no stalls since.
- Frames sent before the other end opens its stream pile up and then arrive in a burst. A handshake in which both
  sides repeated hellos lost its `ready` record behind such a burst (`tools/greet.py`, `invite/greet-invite.log`).
  Now only the side that starts the session repeats its hello; the waiting side answers once.
- Closing a stream sends a CLOSE frame; the other end's reads then return end of file until it reopens.
- One-way bulk works: `dd` wrote 2 GB through one stream at 945 MB/s, all received, no CRC errors.
- Streams survive alongside `thunderbolt_net` on the same cables.

## Files

- `plan-*.jsonl`: the configurations; `incite/*.jsonl`: one result line per configuration (both Macs' timings);
  `incite/*.err`: stderr with clock summaries; `incite/clocks/`, `invite/clocks/`: clock logs; `invite/*.log`:
  server logs.
- `tbstream-up.sh`, `tbstream-down.sh`: create and remove one stream per cable between two nodes.
- `xrun.sh`: clock log and CPU latency request around one command, inside a gpuq lease.
- `tools/`: `pingpong.py`, `lag.py` (one-way delivery timing), `greet.py` (the handshake that lost its `ready`),
  `monitor.sh` (reachability and kernel-log watch used during the runs).

Clocks: incite's cards held 850/1370 MHz in every sample of every GPU run except one idle sample at the end of each.
invite's server logs span the time between client runs as well, at 300/150 idle and 850/1370 while serving.
