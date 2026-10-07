# Project workflow

Work directly on `main` in this project. Do not create a feature branch or worktree unless the user
explicitly requests one. Commit measured, validated improvements on `main`.

Performance work targets `incite` (the dual-D700 Mac Pro). Record the tested source, commands,
correctness checks, sustained GPU clocks, and before/after results. Distinguish hardware upper bounds
from measured throughput and practical targets; retain only improvements supported by measurements.

Only add or retain an option when evidence shows a clear benefit for a supported use case. Experiments
are welcome, but remove failed or inconclusive candidate code and its switches after evaluation.
Keeping an experiment "off by default" is not a substitute for demonstrating a benefit. Performance
options need a repeatable improvement in the relevant end-to-end workload; an isolated kernel or
microbenchmark win is insufficient. Preserve measurements and reproduction snapshots under `results/`,
outside the active runtime.

# Shared GPU on inf

The RTX PRO 6000 on inf (192.168.0.118, tailnet `inf`) is shared through gpuq. Before using CUDA or loading
a model there, read `gpuq rules`, request and wait for a lease, and run GPU commands through
`ssh inf.tdf.sh 'gpuq run ID -- CMD'`. Release the lease promptly. Extend it if needed and honor
YIELD REQUESTED by finishing the current command and running `gpuq yield ID`.
Never stop or start `ds4-qwen.service` yourself; gpuq manages it.

# Shared GPUs on the Mac Pros

The Mac Pros' D700s (incite, invite) are shared by several sessions through one gpuq queue per machine
(dashboards https://gpuq.tdf.sh/m/incite/ and https://gpuq.tdf.sh/m/invite/). Before any GPU run, request a lease with
`gpuq --on MACHINE request LABEL --est MIN`, wait with `gpuq --on MACHINE wait ID`, run GPU commands as
`ssh MACHINE 'gpuq run ID -- CMD'`, and release promptly. For benchmarks, wrap the command in
`scripts/with-clocks.sh CLOCKLOG CMD` to record sustained clocks. Never write into `~/trashcan` on incite (another
checkout); use a `~/trashcan-exp/NAME` workspace. Build with `scripts/cross-build.sh` in the kbuild VM, not on
the Macs. amdgpu kills a GPU job after about 10 s, so keep each queue submission short.
