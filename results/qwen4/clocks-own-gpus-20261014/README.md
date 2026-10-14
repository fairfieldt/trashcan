# Each engine process holds only its own GPUs' clock level (2026-10-14)

**Result.** The clock holder (`engine/src/clocks.rs`) holds the `high` DPM level of the GPUs a process uses (`--gpus`,
by PCI address) and returns only those to `auto`. Before, every process wrote the level of every amdgpu GPU of the
Mac, so with two engine processes on one Mac, one per GPU, a process that started, went idle for a second or ended
set the other's GPU to `auto` while it worked, and the other's holder never raised it again until its own idle
second. That only happens with two processes on one Mac (the one-Mac tests of results/qwen4/decode-gate-20261014,
where it slowed parts of runs); on the chain each Mac runs one process.

## Check on alight (Linux 7.3.0-rc7-p2p), `tools/clk-check.sh`

Two `stage-bench` processes started together: a long one on GPU 1 (layers 3-5, 1500 decode steps) and a short one on
GPU 0 (layers 0-2, 100 decode steps); both GPUs' `power_dpm_force_performance_level` sampled every 50 ms
(`alight/*-levels.txt`), and the journal's sudo lines name the level files each write touched (`alight/*-tee.txt`). A:
main 25a85140 (`trashcan-dgbase3`, main plus the `TRASHCAN_Q4_LAYERS` hook, unused here), B: this commit
(`trashcan-clk1`, sha256 9eee49bd34c818bd...).

| | level writes by the engine | of them naming both GPUs | GPU 1 after the short process ended | long run's decode step |
|---|---|---|---|---|
| A, main | 6 | 6 | `auto` in 129 of 129 samples (until its own process ended, 6.6 s) | 5.35 ms |
| B, this commit | 6 | 0 | `high` in 64 of 64 samples (until its own process ended) | 2.78 ms |

Level changes over the runs (seconds from the start):

```
A  +0.00 GPU0 auto GPU1 auto  short running
   +1.63 GPU0 high GPU1 high  short running
   +2.40 GPU0 auto GPU1 auto  short ended      (GPU 1's process still running: its GPU left at auto)
B  +0.00 GPU0 auto GPU1 auto  short running
   +1.04 GPU0 high GPU1 auto  short running    (the short process raises GPU 0 only)
   +1.10 GPU0 high GPU1 high  short running    (the long one GPU 1)
   +1.76 GPU0 auto GPU1 high  short ended      (only GPU 0 returns to auto)
```

The kernel log (amdgpu faults, PCIe AER) stayed at its boot baseline; `cargo test` passes (with a test of the PCI
address format against sysfs's names).
