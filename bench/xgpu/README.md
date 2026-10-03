# xgpu: cross-GPU sync and copy microbenchmark

This measures what a Vulkan tensor-parallel allreduce would pay on incite (2x D700, RADV) for each way of
moving a small partial result between the two GPUs and telling the other GPU it's ready. Decode allreduces are
about 15 KB (3840 fp32), so latency dominates. Prefill allreduces at 512 tokens are about 7.5 MB, where bandwidth
matters.

```
make                       # on incite: needs libvulkan-dev and glslc
./xgpu                     # all tests
./xgpu xsem spin           # a subset
```

| test | models |
|---|---|
| `fence` | submit tiny work + `vkWaitForFences` on one GPU: the CPU-to-GPU-to-CPU round trip floor |
| `relay` | CPU waits for GPU0, then submits to GPU1, and back: a fully blocking CPU-mediated handoff |
| `proxy` | PR #25051's fallback. Work is pre-submitted with waits on local timelines; a CPU thread polls one GPU's timeline and host-signals the other's |
| `xsem` | PR #25051's native path. Exported/imported OPAQUE_FD timeline semaphores, so each GPU waits on the other's timeline directly |
| `spin` | a shader on each GPU ping-pongs a flag in a host allocation imported into both GPUs. No submission boundaries; the CPU is out of the loop |
| `copy` | GPU-timed `vkCmdCopyBuffer` between VRAM and an imported host allocation, at 16 KiB to 8 MiB |
| `p2p` | GPU1 exports a VRAM buffer as a dma-buf and GPU0 imports it; copy bandwidth both ways, then the spin test with flags in that buffer |

Per-hop numbers are one-way: GPU A finishing something until GPU B can act on it.

The spin loop is bounded (`max_spin` in `run_spin`). If a GPU never sees the other's write, for example because a
stale cache line hides it, the test reports FAILED instead of hanging the GPU into a reset.
