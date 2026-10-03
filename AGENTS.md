# Project workflow

Work directly on `main` in this project. Do not create a feature branch or worktree unless the user
explicitly requests one. Commit measured, validated improvements on `main`.

Performance work targets `incite` (the dual-D700 Mac Pro). Record the tested source, commands,
correctness checks, sustained GPU clocks, and before/after results. Distinguish hardware upper bounds
from measured throughput and practical targets; retain only improvements supported by measurements.

# Shared GPU on inf

The RTX PRO 6000 on inf (192.168.0.118, tailnet `inf`) is shared through gpuq. Before using CUDA or loading
a model there, read `gpuq rules`, request and wait for a lease, and run GPU commands through
`ssh inf.tdf.sh 'gpuq run ID -- CMD'`. Release the lease promptly. Extend it if needed and honor
YIELD REQUESTED by finishing the current command and running `gpuq yield ID`.
Never stop or start `ds4-qwen.service` yourself; gpuq manages it.
