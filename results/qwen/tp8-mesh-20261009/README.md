# Qwen3.8-27B TP8 over six cables: full-mesh all-reduces (2026-10-09)

**Result.** With all six cables, full-mesh all-reduces decode Qwen3.8-27B TP8 at 37.2 tok/s and 50.1 with MTP (f16
payload), against 35.8 and 46.6 on the ring of four cables in the same sessions, and prefill at 211 tok/s (512-token
prompts; 204 at 2048) against 203 (196). Every run generated the same tokens and prefill candidates, and in f32 the
totals are the ring's bit for bit: the validation dump is the ring's, which is the record's. The driver now takes the
full mesh by itself when every machine reports a cable to every other.

## Measurements

Binary `trashcan-mesh`: ddc8df8 with the full-mesh paths behind `TRASHCAN_TP_MESH=full` (the code before the switch
gave way to automatic selection; the paths are the same). Same setup and method as
[results/qwen/tp8-opt-20261008](../tp8-opt-20261008/README.md#results-on-the-macs-2026-10-09), with every Mac's three
cables open in every run (`tools/tp8full.sh`), so ring and full mesh differ only in the all-reduce
(`tools/mesh.sh`, alternating the four configurations; tables from `tools/configs.py`).

### Decode (pp512, tg128, three rounds)

| configuration | n | tok/s (mean, range) | exchange ms/token | GPU wait ms/token | pp512 tok/s |
|---|---|---|---|---|---|
| ring, f32 | 6 | 34.90 (34.70-35.21) | 7.68 | 19.41 | 202 |
| full mesh, f32 | 6 | 36.43 (35.96-36.57) | 6.33 | 19.42 | 209 |
| ring, f16 | 6 | 35.77 (35.43-36.08) | 6.58 | 19.42 | 203 |
| full mesh, f16 | 6 | 37.23 (36.50-37.48) | 5.99 | 19.42 | 210 |

### MTP (k = 2, two rounds)

| configuration | n | tok/s (mean, range) | exchange ms/token | GPU wait ms/token | verification ms/round | tokens/round | pp512 tok/s |
|---|---|---|---|---|---|---|---|
| ring, f32 | 4 | 45.65 (45.55-45.80) | 5.81 | 13.97 | 51.0 | 2.47 | 199 |
| full mesh, f32 | 4 | 45.64 (42.18-47.18) | 6.00 | 13.96 | 51.2 | 2.47 | 205 |
| ring, f16 | 4 | 46.59 (46.47-46.75) | 4.96 | 14.23 | 48.9 | 2.42 | 201 |
| full mesh, f16 | 4 | 50.13 (49.86-50.47) | 4.03 | 13.96 | 46.2 | 2.47 | 207 |

The first repetition of the full mesh in f32 ran at 42.2 tok/s (an exchange stall: 7.3 ms per token); the other three
average 46.8. Every configuration generated the same text, and drafts were accepted alike (73.5%) except on the ring
with f16 (71.2%, one draft fewer over the 128 tokens).

### Prefill

| prompt | ring tok/s | full mesh tok/s | exchanges per prompt |
|---|---|---|---|
| 512 tokens (4 repetitions each) | 202.9 (201.8-205.2) | 210.9 (208.6-213.5) | 2.04 → 1.52 s |
| 2048 tokens (one repetition per run) | 196.8, 195.2 | 203.6 | 7.90 → 6.19 s |

The same candidates in every run. The second 2048-token run of the full mesh did not start: invite's node never
completed the lane handshake with incite and the driver timed out at the probe, before any all-reduce (one session of
about a hundred today).

### Which path for which size (f16, MTP)

| configuration | n | tok/s (mean, range) | exchange ms/token | GPU wait ms/token | verification ms/round | tokens/round | pp512 tok/s |
|---|---|---|---|---|---|---|---|
| one step for rows, direct for batches (default) | 4 | 50.25 (50.00-50.72) | 4.02 | 13.95 | 46.1 | 2.47 | 207 |
| direct for everything (TRASHCAN_TP_MESH=full:0) | 4 | 50.11 (50.03-50.31) | 4.06 | 13.96 | 46.2 | 2.47 | 207 |
| one step for everything | 4 | 49.35 (49.12-49.52) | 4.36 | 13.96 | 47.0 | 2.47 | 163 |

The direct path is the faster one for verification batches (46.1 against 47.0 ms per round) and by far for prefill.
For a decode row the one step is faster (f16, `qwen-bench --pp 512 --tg 128`, three alternated pairs):

| configuration | n | tok/s (mean, range) | exchange ms/token | GPU wait ms/token | pp512 tok/s |
|---|---|---|---|---|---|
| one step for decode rows (default) | 6 | 37.03 (35.31-37.53) | 6.15 | 19.42 | 209 |
| direct for decode rows (TRASHCAN_TP_MESH=full:0) | 6 | 36.66 (36.55-36.79) | 6.42 | 19.42 | 209 |

In five of the six repetitions the one step ran 37.2-37.5 tok/s against 36.6-36.8; the sixth (35.3) hit an exchange
stall, as single repetitions did a few times today on the ring and the full mesh alike (7-16 ms of exchanges per token
instead of 4-8). The committed rule: a single row in one step, a batch of rows by the direct path.

### Correctness gates (`tools/mesh.sh gates`)

| gate | full mesh, f32 | full mesh, f16 |
|---|---|---|
| `qwen-validate` (14-token prompt) | every tensor identical to the ring's dump, which is the record's; the same 10 greedy tokens | identical too (the prompt's all-reduces are f32); the same greedy tokens |
| `qwen-check --mtp` | pass: restore exact, speculative output identical, 83% of drafts accepted, column score error 2.25e-4 (the ring's) | pass: the same, 83%, column score error 0 |
| `qwen-check --verify 3` | pass: no column mismatches, column score error 2.25e-4 | pass: column score error 0 |

The first f16 `--verify 3` run did not start: invite's node had not opened its lanes when the driver's lane handshake
timed out (30 s). Rerun after the decode pairs above, it passed.

### The committed code (`trashcan-final`, `tools/final.sh`)

Built from the commit that adds these results, on ddc8df8: it chooses the full mesh from the probe instead of a switch,
and the f16 payload is the default. Started with all six cables it logged "full mesh"; with the ring's four it did not.
Defaults only, alternating:

| configuration | n | tok/s (mean, range) | exchange ms/token | GPU wait ms/token | pp512 tok/s |
|---|---|---|---|---|---|
| six cables (full mesh chosen) | 4 | 37.22 (36.76-37.49) | 6.02 | 19.43 | 210 |
| four cables (ring) | 4 | 36.08 (35.97-36.23) | 6.66 | 19.43 | 204 |

| configuration | n | tok/s (mean, range) | exchange ms/token | GPU wait ms/token | verification ms/round | tokens/round | pp512 tok/s |
|---|---|---|---|---|---|---|---|
| six cables, MTP | 4 | 50.14 (49.37-50.71) | 4.04 | 13.96 | 46.2 | 2.47 | 207 |
| four cables, MTP | 4 | 46.74 (46.59-46.88) | 4.91 | 14.23 | 48.7 | 2.42 | 200 |

With `TRASHCAN_TP_WIRE=f32` its validation dump over the full mesh is the ring's, every tensor; `qwen-check --mtp` passes
(speculative output identical, 83% accepted, column score error 0) and so does `qwen-check --verify 3` (column score
error 0).

## Files

- `tools/tp8full.sh`: one TP8 run over the four Macs naming all six cables (`../tp8-opt-20261008/tools/tprun.py`).
- `tools/mesh.sh`: the phases above (decode, mtp, prefill, direct, gates); `tools/dec-once.sh`: decode rows in one step
  against the direct path; `tools/final.sh`: the committed code; `tools/configs.py`: the tables.
- `incite/`, `invite/`, `enlight/`, `indite/`: every run's driver `.json`/`.err`, node logs and `clocks/`;
  `incite/commands.log` lists them.
- [streams.txt](streams.txt), [pingpong.txt](pingpong.txt): the six streams and their check.

## Design (`Mesh::once`, `Mesh::direct` in engine/src/qwen/tp.rs)

With every pair cabled, an all-reduce needs no rounds of recursive doubling:

- **One step** (`once`, decode and draft steps: one row). Every machine sends its whole vector to the three others
  at once and adds the four. Each cable carries the whole vector each way in one step, against half of it in each of
  the ring's two dependent rounds.
- **Direct reduce-scatter/all-gather** (`direct`, verification and prefill: more than one row). Machine m owns part m
  (a quarter of the vector; with the f16 payload a block-aligned quarter of each row), receives the three others'
  copies of it in one step, adds the four and sends the total to every machine in a second step. Each cable carries
  half the vector each way in two steps, against three quarters in four steps for the ring's reduce-scatter/all-gather
  and the whole vector in two rounds for its two-round path.

Both add every element in the pairs recursive doubling adds it: (m0 + m1) + (m2 + m3) in the first half (of the
vector in f32, of each row with the f16 payload) and (m0 + m2) + (m1 + m3) in the second, parts 0 and 1 lying in the
first half and parts 2 and 3 in the second. Every machine therefore holds the same total, and in f32 that total is the
ring's bit for bit, whichever path an all-reduce takes. With the f16 payload every machine's contribution is rounded
once (blocks of 256 values counted from the row, in both paths) and the total once more: `direct`'s all-gather sends
the owner's rounded total, and `once` rounds its total the same way, so a verification batch and its columns reduced
one at a time give the same bits. A lane moves the same number of bytes each way, so the f16 parts are padded to the
longest; an f32 vector that does not split into four equal parts takes the one-step path.

The driver takes them when four machines each report, in their answer to its probe, a cable to every other machine, and
passes the choice on at setup; every machine checks that it is cabled to every other. The binary of the A/B above
selected them with `TRASHCAN_TP_MESH=full` instead, and `full:N` took the direct path from N values.
Unit tests (`full_mesh_*` in tp.rs, `decode_tree_*` in wire.rs) run four machines over localhost TCP: f32 totals equal
the ring's pairs bit for bit at decode, verification, prefill and uneven sizes on every path; with the f16 payload every
machine is identical, verification columns equal their decode steps, and both paths give the same bits. They pass on
this Mac (software f16) and on incite (F16C).

## Cables

A sixth Thunderbolt 2 cable joins incite and indite, so every pair of the four Macs is cabled, each cable on its own
controller at both ends (bus 0 bottom row, 1 left column, 2 right column):

| Cable | End A | End B | Stream (A, B) |
|---|---|---|---|
| 1 | incite bus 1 | invite bus 1 | tbstream0, tbstream0 |
| 2 | incite bus 2 | enlight bus 0 | tbstream1, tbstream0 |
| 3 | invite bus 0 | enlight bus 1 | tbstream1, tbstream1 |
| 4 | invite bus 2 | indite bus 2 | tbstream2, tbstream0 |
| 5 | indite bus 1 | enlight bus 2 | tbstream1, tbstream2 |
| 6 | incite bus 0 | indite bus 0 | tbstream2, tbstream2 |

Streams were created with `scripts/tbstream-up.sh A B tp8` in the order 1-6 above, which keeps the numbering of
results/qwen/tp8-20261007 for cables 1-5 ([streams.txt](streams.txt)). Machines: 0 incite, 1 invite, 2 enlight,
3 indite.

Bringing the links up after the Macs rebooted:

- Resetting the controllers one Mac after another (`results/kernel/tb-loopback-20261006/tb-reset.sh` started on all
  four at once, but each Mac's reset finished at a different time) left all three of enlight's links broken.
  Resetting in two synchronized phases fixed them: first unload the Thunderbolt modules and reset the three NHIs on
  all four Macs, wait for all four, then load the modules on all four within a second. One such round brought five
  links back on two lanes; a second one brought back the last (a one-lane link the first time).
- Cable 3 then still did not come up: invite's end reported its second lane stuck connecting and enlight's end one
  lane up (port PHY state in the lane adapter's PHY capability, `/sys/kernel/debug/thunderbolt/*/portN/regs`: 2 up,
  1 connecting, 7 unplugged). Replugging both ends fixed it. The sixth cable had first gone into incite's left
  column, beside cable 1 on the same controller; it was moved to incite's empty bottom row.
- After the power cut below, the synchronized reset alone (two rounds) brought all six links back on two lanes.

Every stream passed a 16 B and a 20 KiB ping-pong with its data checked ([pingpong.txt](pingpong.txt), single-threaded
Python, so the latencies include the interpreter).

## Power

At 16:16 all four Macs lost power at once, about 45 s into a 7680-token prefill (the long-context decode run of the
merged-candidate round): their power strip tripped. The Macs' SMC telemetry (netconsole `CRASHCAPTURE` lines) shows
390-420 W each (both D700s 122-141 W, the CPU 88 W), about 1.65 kW for the four, sustained from the start of that
prefill. An ordinary benchmark run (two 512-token prefills and two 128-token decodes) draws 1.2-1.55 kW for 10-15 s,
and the four idle at about 350 W. Until the Macs have a better power arrangement, runs keep full-load stretches short
(pp2048 with one repetition, no long-context runs).
