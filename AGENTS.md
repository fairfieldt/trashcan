# Development guidelines

Performance work targets a Mac Pro 6,1 with two D700s. Record the tested source, commands, correctness checks,
sustained GPU clocks (`scripts/with-clocks.sh`), and before/after results. Distinguish hardware upper bounds
from measured throughput and practical targets; retain only improvements supported by measurements.

Only add or retain an option when evidence shows a clear benefit for a supported use case. Experiments
are welcome, but remove failed or inconclusive candidate code and its switches after evaluation.
Keeping an experiment "off by default" is not a substitute for demonstrating a benefit. Performance
options need a repeatable improvement in the relevant end-to-end workload; an isolated kernel or
microbenchmark win is insufficient.

Code comments cite measurements as `results/...`; those archives are kept outside this branch.

# GPU job length

amdgpu resets the GPU when one job runs or waits longer than 2 s (the `lockup_timeout` default on 7.3), so
keep each queue submission well under that, and never leave GPU work waiting on a host event without a bound.

# Light barriers in Flash-Next decode

Flash-Next graphs of up to 8 tokens end their dispatches with shader-write -> none barriers (`Device::compute_barrier`
mode 3). They rely on RADV (Mesa 25.0.7 on GFX6) doing an L2 write-back plus an L1 invalidation for a `SHADER_WRITE`
source, and on the kernels in `SCALAR_SAFE` (engine/src/vulkan.rs) never reading through scalar loads what an earlier
dispatch wrote. After any Mesa upgrade, or a change to one of those shaders, audit the compiled ISA for scalar loads
of buffers written earlier in the graph, and stress-test decode output against full barriers, before measuring.
