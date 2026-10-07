# Discarded LDS swizzle experiment, October 7

The candidate swizzled the attention K/V shared-memory vec4 columns by row bit 3,
mirroring gemm2. Both GPUs passed scalar tolerance and bit-for-bit equality against
the existing tiled path at five shapes. Each measurement held a gpuq lease on incite
and used `scripts/with-clocks.sh`. The candidate was approximately 1–3% slower and
was removed. It is not a retained optimization. The source archive and binary hash
identify the tested experiment, including the already pending cancellation changes.
The command was `qwen-attention-bench --tokens T --n N --heads H --iterations 8` for
(4096,64,6), (4096,129,6), (4096,256,6), (8192,256,6), (4096,129,2).
