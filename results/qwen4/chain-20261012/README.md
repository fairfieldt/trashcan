# Qwen3.8-Flash-Next on six Mac Pros: first chain (2026-10-12)

**Result.** The G-iq4 GGUF (Q8_0 dense, IQ4_XS/IQ4_NL experts) runs as a 12-stage pipeline over all six Mac Pros,
4 layers and a twelfth of the LM head per D700, and reproduces the CPU reference: the same 10 greedy tokens, logits
rel 2.9e-4, top-10 identical. Prefill 126-132 tok/s (512-token prompt); decode 8.0-8.2 tok/s with amdgpu's automatic
clock levels and 13.5 tok/s with the GPUs held at the high level, against 14.1 projected in docs/qwen4.md.

## Cables

Six Thunderbolt 2 cables form one ring, each Mac with one cable on bus 0 (bottom row) and one on bus 1 (left column),
so every cable has its own controller at both ends; bus 2 (right column) is unused. After two rounds of
`scripts/tb-reset.sh` on all six ([tb-reset-1.log](tb-reset-1.log), [tb-reset-2.log](tb-reset-2.log); the first left
incite-enlight on one lane without carrier) every link is up on two lanes; incite-indite carries 8.99 Gb/s (iperf3).

| Cable | End A | End B |
|---|---|---|
| 1 | incite bus 0 | indite bus 0 |
| 2 | indite bus 1 | inflight bus 1 |
| 3 | inflight bus 0 | alight bus 0 |
| 4 | alight bus 1 | invite bus 1 |
| 5 | invite bus 0 | enlight bus 0 |
| 6 | enlight bus 1 | incite bus 1 |

The chain follows the ring from the driver: incite (layers 0-7, n-gram table) -> indite -> inflight -> alight ->
invite -> enlight (layers 40-47, final mixer). Results return along the same cables; the sixth cable is not used yet.

## Model on every Mac

`G-iq4-0000{1..6}-of-00006.gguf` with shard 3 (the Q8_0 n-gram table, 54 GB) as on incite, 125 GB per Mac under
`~/models/qwen3.8-flash-next/`. Shard 5 came from this Mac to incite over the tailnet (~70 MB/s); incite then streamed
everything along the ring with tar over netcat (`tools/relay.sh`, `tools/relay-node.sh`): 371 s, ~370 MB/s into all
five Macs at once. rsync over ssh managed 155 MB/s. Sizes checked on every Mac.

## The fix

The first run produced different tokens from layer 0 on: `qwen_gdn` (shared with qwen35) went from 16 state rows per
workgroup to 4 in eabff7b (TP8 round 2) and only qwen35's dispatch followed, so qwen4 computed a quarter of every
DeltaNet state. Both dispatches now use `qwen::gpu::GDN_ROWS_PER_GROUP`.

## Validation (`val6b`)

`qwen4 validate` on the 14-token reference prompt, `--head-split 12 --context 32768 --batch 128 --ngram disk`,
against the CPU reference of the same GGUF (`qwen4 ref`, 2026-10-06), [compare-val6b.txt](compare-val6b.txt):

| | |
|---|---|
| greedy | 9419 0 2500 628 353 1438 488 3242 30 248046: identical |
| logits | rel 2.85e-4, cos 1.000000, top-10 overlap 10/10 |
| residual after layer 47 | rel 1.8e-4 |
| largest per-tensor error | QSA outputs (f16 K/V) rel 5-6e-4, as in the single-Mac validation |

`Qcur-*` shows rel 0.94 at cos 1.0: the GPU captures queries after the 1/16 scale.

## Speed

Binary from this commit's tree (cross-built), `tools/q4run.py` (nodes under gpuq leases with clocks logged and the 0 us
CPU latency request), `qwen4 bench --pp 512 --tg 64 --head-split 12 --context 32768 --batch 128 --ngram disk`.
`--dpm high` runs every Mac under `tools/dpm-run.sh high`, which holds both D700s at amdgpu's high DPM level for the
run and restores `auto`. Runs alternated:

| run | DPM | pp512 tok/s | tg64 tok/s | ms/token | incite's 8 layers, GPU ms per token |
|---|---|---|---|---|---|
| bench6-pp512 | auto | 126.6 | 8.01 | 124.9 | 16.11 |
| bench6-pp512-high | high | 131.6 | 13.50 | 74.1 | 10.57 |
| bench6-pp512-auto-2 | auto | 125.7 | 8.24 | 121.4 | 12.93 |
| bench6-pp512-high-2 | high | 129.8 | 13.51 | 74.0 | 10.56 |

In decode each GPU works about 5 ms of every token, and on `auto` it drops to 300/150 MHz in between (clock logs:
mostly 300/150 samples), so every layer starts slow. Held at high: 850/1370 MHz throughout, and the layers take the
stage benchmark's times (1.2 ms per DeltaNet layer, 1.45 for QSA, 1.62 for layer 1 with the n-gram module).

Holding the high level costs power while idle: SMC telemetry (netconsole, `tools/power.py`) shows each D700 at
49-55 W idle on high against 14-22 W on auto, about 400 W more over 12 GPUs. Decode and pp512 runs peaked at
274-297 W per Mac (PSTR).

Prefill: 4 chunks of 128 tokens fill a 6-machine pipeline whose stages take ~400 ms per chunk (8 layers at 44-58 ms,
the two GPUs of a Mac in turn), (4 + 5) x 0.4 s = 3.6 s; longer prompts approach 1 chunk per 0.4 s (~320 tok/s).
Longer prefills were not run: all six Macs work at once (~1.7 kW).

Single-Mac check before the chain (`stage-mac1`, incite, layers 0-7 and two head twelfths): 5814 / 5781 MiB of
budgets 5865 / 5847; decode 1.18-1.59 ms per layer; head slice 1.1 ms.

## Files

- `tools/q4run.py`: one chain run (nodes last first, then the driver); `tools/dpm-run.sh`; `tools/summary.py`.
- `incite/`: driver outputs (`*.json`, `*.err`), `commands.log`, `clocks/`; `val6b/greedy.txt`, `index.tsv`.
- `indite/`, `inflight/`, `alight/`, `invite/`, `enlight/`: node logs and clocks.
- `val6`: the run before the fix (wrong tokens from layer 0).
