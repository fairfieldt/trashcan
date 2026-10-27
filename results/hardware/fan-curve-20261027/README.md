# Fan curve: full speed while the GPUs work, 1200 rpm when idle and cool (2026-10-27)

**Result.** `fancurve.sh` no longer holds the fan at 2350 rpm at every temperature. It runs at full speed (2350 rpm or
the SMC maximum if higher) while a GPU works, a compute ring's fence counter advancing or the DPM level held at `high`,
and for 120 s after; otherwise at 1200 rpm up to 50 C GPU / 60 C CPU, full speed from 65 C / 80 C, linear in between,
stepping down at most 100 rpm per 2 s poll. Installed on all six Macs with `deploy/trashcan-cluster install
--skip tblink,network,power,gpu,logs,panic,ssh,disk --apply` (only the fan component).

| check | result |
|---|---|
| idle, 15 min at the new policy (`idle-1200.log`, all six) | every fan at 1200 rpm throughout; GPUs 36-41 C at the start, 39-46 C after 15 min; CPU packages 37-46 C; nothing reached 50 C, so the curve never raised the fan |
| ramp on invite (`ramp-invite.log`, the Mac whose GPU 0 runs hottest): three Flash-Next stage prefills, layers 0-8 on both GPUs, 200 chunks of 128 tokens each | target 2350 within 2 s of the job's start, before the DPM level went `high` (the model upload's fences); 2304 rpm after 14 s, 2350 before the prefill proper; full speed through the 2.5 min of work, peak 71 C (GPU 0); held 2350 for 120 s after the last work, then 2350 -> 1200 in about 35 s; GPUs at 46-47 C at 1200 rpm |
| six-Mac smoke of 6-flashnext with MTP (`smoke-fans.log`), compute-ring-only version | every Mac's fan went from 1200 to 2350 during the run |
| `scripts/test-fancurve.py` | 48 tests, both applesmc layouts, on macOS bash 3.2 and on a node's bash 5.2 |

The ramp test ran with the first version, which counted every ring's fences; the version installed afterwards counts
only compute rings, as the port lights do, so a desktop drawing on a GPU does not hold the fan at full speed.
Installed, it held 1200 rpm at idle on all six, and the smoke above went to full speed.

For comparison, the morning's six-Mac A/B with the fan pinned at 2350 (results/qwen4/default-112k-20261027) peaked at
63 C, and the old idle temperatures at 2350 rpm were 35-41 C (gpuq readings after the boot).

## Tools

`tools/fan-watch.sh OUT MIN STEP [MACHINES]` samples fan rpm/target, both GPU edge temperatures, the CPU package and
the DPM level over ssh; `tools/fan-ramp.sh` is the ramp test under a gpuq lease on invite.
