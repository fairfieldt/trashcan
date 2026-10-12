# Flash-Next: all Macs score their head slices at once (2026-10-12)

**Result.** Six-Mac decode (`qwen4 bench --pp 512 --tg 128`, real text) goes from 13.53 to 14.70 tok/s (73.92 to
68.04 ms per token, -5.9 ms, +8.6%), with greedy tokens, logits and all 583 dumped tensors bit-identical to the
baseline. Prefill is unchanged (132 tok/s on 512 tokens).

## The change

With `--head-split 12` every D700 holds a twelfth of the LM head. Before, the reply to a decode token travelled up
the chain as one frame (candidates, final hidden state `h`, captures) and every Mac scored its two slices (~1.1 ms,
both GPUs at once) before passing it on; on enlight the ending stage also scored its own slice inside its last
layer's submission, before `h` left the GPU. After the last stage, the critical path was ~1 ms (enlight GPU1's
slice) + ~1.1 ms (enlight GPU0's slice) + 5 x (hop + 1.1 ms).

Now a reply is two frames:

1. `{"h": d}` + `h`: every machine passes it up as soon as it arrives, then scores its own rows;
2. `{"top", "captures"}`: the machine's merged candidates and those of the machines after it, sent after its own
   scoring and the second frame from below.

Every Mac therefore scores at about the same time and the critical path is ~1.1 ms + 5 hops. The merge (`merge_top`:
score, then id, truncated to `TOP_K`) does not depend on order, so the candidates are the same.

Two steps, measured separately:

- **a** (`net.rs`, `pipeline.rs`): the two-frame reply. `Local::forward` no longer scores the other stages' slices
  on the machine that ends the model; the caller sends `h` up first and then calls `head_slices`. Without `reply`
  (prefill chunks before the last) no slices are scored.
- **b** (`gpu.rs`, `pipeline.rs`): the stage that ends the model leaves `h` in its head buffer and returns
  `Output::Final(h)`; its own slice is scored with the others through `head_submit`, after `h` has been sent.
  CPU stages do the same.

## Tested source

Branch `worktree-agent-a966aea2cf0b152d3`, on top of c55128f. Binaries cross-built with `BUILD_ONLY=1
scripts/cross-build.sh` and deployed to `~/trashcan-exp/flashnext/` on all six Macs:

| variant | source | binary | md5 |
|---|---|---|---|
| base | c55128f | `trashcan-c55128f` | d3640264bd090462ba2d97ebfbdf3a11 |
| a | 5f5c3bd | `trashcan-head-a` | b89b5c77e7ec2e6894cd5ee38286443a |
| b | the commit adding this README (engine as built) | `trashcan-head` | 87b3904c3ca5a639bca125c2ed3967cc |

## Commands

One six-Mac session under `chain6.sh` (all leases at once), [tools/ab-head.sh](tools/ab-head.sh): validation of all
three binaries, three alternating rounds of benches, then the server test with `b`. Every run is a
`tools/q4run.py` run from [../chain-20261012](../chain-20261012/README.md) (nodes under gpuq leases, clocks logged,
0 us CPU latency request):

```sh
chain6.sh "head A/B" 40 "Flash-Next head-slice agent" -- tools/ab-head.sh
# per run, e.g.
q4run.py --binary trashcan-head --name head-tg128-b-1 --driver "bench --gpus 0,1 \
    --nodes indite.tb0:7071,inflight.tb1:7071,alight.tb0:7071,invite.tb1:7071,enlight.tb0:7071 \
    --head-split 12 --context 32768 --batch 128 --ngram disk --pp 512 --tg 128 \
    --tokens-file /home/fairfieldt/trashcan-exp/flashnext/long.ids"
```

## Correctness

- **Six Macs**, `qwen4 validate` (14-token reference prompt, `--head-split 12 --context 32768 --batch 128 --ngram
  disk --greedy 10`): base, a and b all give 9419 0 2500 628 353 1438 488 3242 30 248046, the CPU reference's
  tokens. Against the base dump, a and b match bit for bit: `logits.f32` identical (`cmp`), all 583 tensors rel 0
  ([compare-a-vs-base.txt](compare-a-vs-base.txt), [compare-b-vs-base.txt](compare-b-vs-base.txt)). Against the CPU
  reference (`qwen-flash/ref/g-iq4-cpu`): logits rel 2.84e-4, top-10 10/10, as before
  ([compare-b-vs-ref.txt](compare-b-vs-ref.txt); `Qcur-*` rel 0.94 is the 1/16 query scale, as in the chain README).
- **CPU chain on the Mac** (driver and two CPU nodes on 127.0.0.1, `--cuts 16,32 --head-split 3`, so a node relays
  `h` and both nodes score; [tools/cpu-chain.sh](tools/cpu-chain.sh)): logits rel 0 against the CPU reference, greedy
  9419 0 2500 628 ([compare-cpu-chain-vs-ref.txt](compare-cpu-chain-vs-ref.txt)); one CPU process alone (the
  local `Output::Final` path) also rel 0.
- **Server** (`tools/q4serve-test.sh` as a `--raw` driver with b, `--head-split 12 --context 16384 --batch 128
  --ngram disk`, [incite/head-serve-b.json](incite/head-serve-b.json)): `q4-turns.py` turn 2 reuses 78 of 99 prompt
  tokens (64 tokens in 4.24 s), and every `scripts/test-server.py` check passes.

## Speed

`qwen4 bench --pp 512 --tg 128 --tokens-file long.ids`, rounds alternated base, a, b ([summary.txt](summary.txt),
from [tools/summarize.py](tools/summarize.py); driver outputs in `incite/`):

| round | base tg128 tok/s (ms) | a | b |
|---|---|---|---|
| 1 | 13.52 (73.96) | 14.53 (68.83) | 14.71 (68.00) |
| 2 | 13.52 (73.99) | 14.50 (68.95) | 14.60 (68.49) |
| 3 | 13.55 (73.82) | 14.53 (68.83) | 14.78 (67.64) |
| mean | **13.53 (73.92)** | **14.52 (68.87)** | **14.70 (68.04)** |
| pp512 tok/s, mean | 131.9 | 132.6 | 132.2 |

- a saves 5.05 ms per token (five ~1.1 ms scorings off the critical path; 5.5 expected);
- b saves another 0.83 ms (the ending stage's own slice, ~1 ms) and beats a in every round.

Sustained clocks: in each run's decode window (the driver clock log's last `tg_seconds`, every Mac's log), every
sample of all 12 GPUs is at 850/1370 MHz (390-430 samples per run); the engine holds the high DPM level while work
arrives (`engine/src/clocks.rs`). Clock logs in `*/clocks/`, node logs in each Mac's directory.

Now about 1.1 ms of head scoring (one Mac's two slices) and five ~0.1 ms hops follow the last layer, against
~7.6 ms of scoring and the same hops before. Per decode token, 68 ms remain: 48 layers at ~1.2-1.6 ms, 12 stages of
host work and the hops.

## Files

- `incite/`: driver outputs (`head-*.json`, `*.err`), `ab-head.log`, `greedy-val.txt`, `clocks/`; the server test
  (`head-serve-b.json`, `head-serve-b-server.log`).
- `indite/`, `inflight/`, `alight/`, `invite/`, `enlight/`: node logs and clocks.
- `tools/`: `ab-head.sh` (the session), `summarize.py`, `cpu-chain.sh` (the local CPU chain).
