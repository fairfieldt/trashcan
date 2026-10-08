# Qwen3.8-27B TP8 over four Mac Pros (2026-10-07)

Result: 204 tok/s prefill (512-token prompts; 197 at 2048), 35.0 tok/s decode and 46.0 with MTP, against TP4's
149 / 28.3 / 38.6 on two of the same Macs that day.

Tensor parallelism over incite, invite, enlight and indite: eight D700s, one rank each. Before this, `TpGroup` had
one peer; a group is now 2, 4 or 8 machines (`Mesh` in `engine/src/qwen/tp.rs`), and with more ranks than the
model's 4 KV heads, each KV head is held and computed by the ranks whose query heads read it (`Split::kv_part`).

## Cables

Five Thunderbolt 2 cables; every pair is cabled except incite and indite. Bus 0 is the bottom row of ports,
bus 1 the left column and bus 2 the right column. Each cable is on its own controller at both ends.

| Cable | End A | End B | Stream (A, B) |
|---|---|---|---|
| 1 | incite bus 1 | invite bus 1 | tbstream0, tbstream0 |
| 2 | incite bus 2 | enlight bus 0 | tbstream1, tbstream0 |
| 3 | invite bus 0 | enlight bus 1 | tbstream1, tbstream1 |
| 4 | invite bus 2 | indite bus 2 | tbstream2, tbstream0 |
| 5 | indite bus 1 | enlight bus 2 | tbstream1, tbstream2 |

Machines: 0 incite (driver), 1 invite, 2 enlight, 3 indite. The all-reduce uses cables 1, 2, 4 and 5 (machine
pairs 0-1, 0-2, 1-3, 2-3); cable 3 is unused. A sixth cable, incite bus 0 to indite bus 0, would cable every pair.
Streams: `scripts/tbstream-up.sh A B tp8` for each pair ([streams.txt](streams.txt)).

After indite and enlight rebooted, four links kept the Mac firmware's peer or came back on one lane. Resetting
one Mac's controllers at a time moved the problem to its neighbours. Resetting all four Macs' controllers at
once fixed every link (2 lanes, no link errors): on each Mac, remove the stream groups, unload `thunderbolt_stream`,
`thunderbolt_net` and `thunderbolt`, write 1 to `/sys/bus/pci/devices/0000:{16:00.0,5d:00.0,a4:00.0}/reset`
(as `results/kernel/tb-loopback-20261006/tb-reset.sh`), and reload the modules.

## Design

- **All-reduce.** Recursive doubling: in round i, machine m exchanges with machine `m ^ 1 << i`. The vector is
  cut into one slice per round, and in round i slice j goes to the partner of round (i + j) % rounds, so both
  cables of a machine carry half the vector in every round. Each element is added in the same pairs on every
  machine (the first half (s0 + s1) + (s2 + s3), the second (s0 + s2) + (s1 + s3)), so all GPUs keep bitwise-identical
  residuals. Candidates merge in the same rounds.
- **Prefill-sized all-reduces** (at least 64K elements) use reduce-scatter then all-gather over the same slices and
  partner orders: each step keeps the half of a slice segment selected by this machine's bit for that partner and
  adds the partner's copy; the all-gather copies the finished segments back. Each cable moves 3/4 of the bytes.
  Every element is added on one machine in the same pairs as above, so totals are bitwise identical to the
  two-round path.
- **Operations.** The driver's forwards, drafts, rollbacks and checkpoints go down a binomial tree over the same
  cables: machine m hears from `m & (m - 1)` (0 to 1 and 2, then 2 to 3) and passes each message on before running
  it. Logit dumps go back up the tree.
- **Start.** Each machine waits on its lanes to lower machines and then starts its lanes to higher ones, so a
  session starts when the driver does. Start nodes before the driver.

```sh
indite:  trashcan qwen-tp-node MODEL --mtp MTP --machine 3 --peer 1=/dev/tbstream0 --peer 2=/dev/tbstream1
enlight: trashcan qwen-tp-node MODEL --mtp MTP --machine 2 --peer 0=/dev/tbstream0 --peer 3=/dev/tbstream2
invite:  trashcan qwen-tp-node MODEL --mtp MTP --machine 1 --peer 0=/dev/tbstream0 --peer 3=/dev/tbstream2
incite:  trashcan qwen-bench MODEL --tp-peer 1=/dev/tbstream0 --tp-peer 2=/dev/tbstream1 --pp 512 --tg 128
```

The two-machine options (`--tp-stream`, `--tp-connect`, node `--stream`, `--listen`) are unchanged.

## Method

Kernel 7.2.9-p2p on all four. Under gpuq leases on every Mac, `tools/tprun.py` (with `tools/tp8.sh`) runs each
process under `scripts/with-clocks.sh` and `scripts/latency-run.py` (0 µs CPU latency request), with the
performance governor on all four for the measurement windows. Every card ran at 850/1370 MHz while computing
(clock logs in `*/clocks/`, summaries at the end of each `.err` and node log); the hottest card was invite's first
at 84 °C. `qwen-bench` defaults: context 8192, Q4_0 trunk with Q6_K head, MTP k = 2 with a 64K draft head.

Binaries (each cross-built with `scripts/cross-build.sh`):

| Binary | Source |
|---|---|
| `trashcan-base` | 12826cb |
| `trashcan-tp8` | 12826cb + KV sharing and `Mesh`, whole vector to one partner per round |
| `trashcan-split` | + slices to both partners per round; `TRASHCAN_TP_SPLIT=0` restored the whole-vector rounds for the A/B; single-partner exchanges in the new polling loop |
| `trashcan-split2` | + the original `exchange_within` for single-partner rounds (TP4 path unchanged); 75fe01a is this with the switch removed |
| `trashcan-rsag` | 75fe01a + reduce-scatter/all-gather for prefill-sized vectors (`TRASHCAN_TP_RSAG=0` turned it off for the A/B); the next commit is this with the switch removed |

## Correctness

| Check | Result |
|---|---|
| `qwen-validate`, TP8 split, vs CPU reference (`qwen-ref`) | worst 4.02e-4 over 946 tensors; logits 4.02e-4, top-10 10/10; greedy 10/10 ([tp8-split-vs-ref.txt](tp8-split-vs-ref.txt)) |
| same, TP8 whole-vector rounds | worst 3.88e-4; logits top-10 10/10; greedy 10/10 ([tp8-vs-ref.txt](tp8-vs-ref.txt)) |
| TP8 vs TP4 dump | worst 3.92e-5 ([tp8-vs-tp4.txt](tp8-vs-tp4.txt)); TP4's record was 4.11e-4 vs the reference |
| `qwen-check --verify 3`, TP8 | pass: restore exact, no column mismatches, column score error 2.25e-4 |
| `qwen-check --mtp`, TP8 | pass: restore exact, speculative output identical, 83% accepted |
| `qwen-validate`, TP8 with reduce-scatter/all-gather | every tensor bitwise identical to the split run's dump (the 14-token prompt takes the new path) |
| `qwen-check --mtp`, TP8 with reduce-scatter/all-gather | pass: restore exact, speculative output identical (600-token prefill on the new path) |
| prefill candidates (`pp_top`), reduce-scatter on and off | bitwise identical at 512 and 2048 |
| `scripts/test-server.py`, `serve --tp-peer ... --mtp` | pass ([incite/test-server-tp8.txt](incite/test-server-tp8.txt)) |
| generated text | the same 128 tokens in every TP4 and TP8 `qwen-bench` run |

## Speed (tok/s, two repetitions per run)

| Qwen3.8-27B | pp512 | tg128 | tg128 with MTP | pp2048 |
|---|---|---|---|---|
| TP4, incite + invite, 3 cables (2026-10-06 record) | 150.4, 151.3 | 28.44, 28.24 | 39.48, 39.63 | 134.1, 133.6 |
| TP4, incite + invite, 1 cable | 148.1, 148.9 | 28.28, 28.27 | 38.33, 38.85 | 140.9, 136.0 |
| TP8, whole vector per round | 140.9, 142.1 | 33.87, 33.68 | 42.69, 42.76 | 139.5, 140.9 |
| TP8, split rounds | 190.4, 192.7 | 35.20, 35.01 | 45.98, 46.10 | 184.8, 185.1 |
| **TP8, + reduce-scatter/all-gather for prefill** | **202.5, 206.7, 200.6, 203.8** | **35.0** (decode unchanged) | **46.0** | **195.9, 198.1** |

Split against whole-vector rounds, alternating in one session (`ab-*`): pp512 190.4, 192.7, 189.6, 191.3 against
141.9, 141.2, 142.2, 142.6; tg128 35.20, 35.01, 35.00, 35.06 against 33.71, 33.77, 33.69, 33.63; exchange 7.4-7.6
against 8.3-8.7 ms per token. The final binary: pp512 187.7, 193.1, tg128 34.86, 35.03 (`split2-tg`).

Reduce-scatter/all-gather on against off, alternating (`rs-*`, tg64): pp512 202.5, 206.7, 200.6, 203.8 against
192.5, 192.2, 191.2, 193.3; pp2048 195.9, 198.1 against 186.0, 184.5 (decode does not take this path: tg64 35.35-35.49 against 35.02-35.28). Exchange
time per 512-token prompt fell only 7% (2.21 to 2.07 s) for 25% fewer bytes: four dependent steps instead of two, each
waiting for its slowest partner.

TP4 with the multi-machine code, alternating with `trashcan-base` on incite + invite: the first build (every
exchange in the new polling loop) decoded 0.56% slower over 8 runs each (28.07 against 28.22, `tp4-trashcan-*`).
With the original loop kept for single-partner rounds it matches: 28.22 against 28.20 over 6 runs each
(`tp4b-*`); prefill within the usual session spread (137-148).

## Where the time goes

Decode, TP8 with whole-vector rounds (`TRASHCAN_TP_PROFILE`, [prof-*.txt](.)): a step takes 30.1 ms, of which
19.1 ms (incite, enlight) to 19.75 ms (invite, indite) is GPU segments and about 10 ms is host time between
segments, nearly all exchange (60-96 µs per all-reduce). The head segment takes 1.19 ms. TP4's step is 35 ms
with 29-30 ms on the GPUs, so halving each GPU's weights cut GPU time by a third, not a half.

Prefill: of 2.53 s per 512-token prompt, the host spends 2.07 s in exchanges and 0.2 s waiting for GPUs. Each
all-reduce moves 3.75 MiB each way per cable in about 8.1 ms, about 0.5 GB/s each way, the rate a cable carried in
both directions at once before (a 10 MiB exchange over three streams took 6.9 ms, `../tp4-viability-20261006/`;
`thunderbolt_net`, 11 Gb/s combined, `../../kernel/tb-nodes-20261006/`). Prefill is bound by the cables.

Decode GPU time per dispatch ([timeline-tp8-decode.txt](timeline-tp8-decode.txt), incite rank 0, step 20): the
Q4_0 matrix kernels run at 130-140 GB/s (the FFN's gate and up, 13.9 MB, in 100 µs); `attention_merge64` takes 39 µs
per attention layer for three heads, `qwen_qk` 14 µs, and the head's three `qwen_topk` passes 214 µs after a 970 µs
`headvec`. `TRASHCAN_TP_PARTS` 1, 2 and 4 gave 34.3-35.1 tok/s with no consistent order (`parts*`), so the default
(2) stays.

## Not done

- A sixth cable (incite bus 0 to indite bus 0) for a one-round decode all-reduce and a direct reduce-scatter
  (half the bytes per cable of today's prefill).
- GPU decode efficiency at TP8 shard sizes: matrix kernels at half the D700's bandwidth, and fixed-cost kernels
  (attention merge, `qwen_qk`, head top-k).
