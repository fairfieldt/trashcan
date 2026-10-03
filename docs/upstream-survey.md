# Upstream survey: `-sm tensor` on Vulkan (as of 2026-10-02)

Clone: `llama.cpp/` at upstream 99b95488c (b11366+5). Fetched for reference:

| local ref | what |
|---|---|
| `pr-25051` | https://github.com/ggml-org/llama.cpp/pull/25051, Vulkan allreduce (draft, pwilkin) |
| `pr-24720` | https://github.com/ggml-org/llama.cpp/pull/24720, Vulkan command-buffer graph reuse (open) |
| `bluematt/claude-tensor-par` | TheBlueMatt's Vulkan TP prototype (Intel Xe): P2P allreduce, vk graph caching, parallel per-backend dispatch |

## How TP works upstream

The meta backend comes from https://github.com/ggml-org/llama.cpp/pull/19378 and lives in
`ggml/src/ggml-backend-meta.cpp`.

- It wraps N backends and infers each tensor's split state through a callback.
  - The callback is `llama_meta_device_get_split_state` in `src/llama-model.cpp` (L379-880).
- It runs one subgraph per device and allreduces at split boundaries: after attention-out and after FFN-down.
- For the allreduce it calls a backend `ggml_backend_comm_init` / `ggml_backend_comm_allreduce_tensor` if one is
  exported. Otherwise it uses a generic butterfly made of `ggml_backend_tensor_copy_async` + ADD (L2329).
- Backends that export the comm hooks: CUDA/HIP, SYCL and Hexagon. **Vulkan doesn't, on master.**

gemma4 is allowed: `llm_arch_supports_sm_tensor()` in `src/llama-arch.cpp:1181` is a denylist and gemma4 isn't
on it. https://github.com/ggml-org/llama.cpp/pull/28965 fixed gemma4's fused-QKV split.

`-sm tensor` forces flash attention on (`src/llama-context.cpp:3822`), so FA correctness on GCN 1.0 is a
prerequisite.

## Why Vulkan TP is slow on master (verified in code)

1. `ggml_backend_vk_cpy_tensor_async` (`ggml-vulkan.cpp:13305`) returns false whenever the two devices differ.
2. So `ggml_backend_tensor_copy_async` (`ggml-backend.cpp:651`) synchronizes **both** backends and does a
   blocking `ggml_backend_tensor_copy`.
3. That calls `ggml_vk_buffer_copy` (`ggml-vulkan-buffers.cpp:688`), which runs src VRAM → `sync_staging`
   host buffer → fence wait → `ggml_vk_buffer_write` to the destination.
4. The butterfly does this in both directions for every allreduce. That's about 96 allreduces per token for
   Gemma 4 12B (2 per layer × 48 layers), each a full pipeline drain.

Measured elsewhere (discussion #10879, 2x 7900 XTX, RADV 26.2.1): Gemma 4 31B Q8_0 got tg 24.4 with layer
split vs 21.0 with tensor split. llama-2 7B Q4_0 got 157 vs 56.

## PR #25051 (the main starting point)

+897/-1, mostly `ggml-vulkan.cpp`. Single commit 29395e39b, rebased 2026-09-23.

- Adds the comm hooks to Vulkan.
- One host allocation is imported into every device through `VK_EXT_external_memory_host`.
- Partials are staged as F16 and accumulated in F32.
- Ring allreduce by default; recursive halving/doubling with `GGML_VK_COMM_TREE`.
- Cross-device sync:
  - Native path: `VK_KHR_external_semaphore_fd` OPAQUE_FD timeline semaphores, imported into peers. It's used
    only if every device has the same driverUUID and can export timeline semaphores.
  - Fallback: a CPU proxy thread. Force it with `GGML_VK_COMM_PROXY`.
  - Its own comment warns that "RADV on older Mesa" can't create exportable timelines. Our Mesa is 25.0.7, so
    this needs checking.
- A device-to-device experiment (`GGML_VK_COMM_D2D`) was much slower on AMD.
- Reported results:
  - 2x 7900 XTX, Qwen3.5-27B: tg 34.5 → 41.6 (layer → tensor).
  - 2x W7800: tg 29 → 43.6.
- Open problems: a multi-turn slowdown, and gibberish with 5 GPUs.

Related:

- https://github.com/ggml-org/llama.cpp/issues/22648, TheBlueMatt's explanation: importing *device* memory
  across VkDevices without a device group violates VUID-VkMemoryAllocateInfo-None-00644. So a shared host buffer
  plus imported semaphores is the realistic path.
- https://github.com/ggml-org/llama.cpp/pull/29266 resolves views in multi-buffers. Non-view tensors spanning
  several Vulkan buffers still abort under TP. Watch `GGML_VK_SUBALLOCATION_BLOCK_SIZE`.
- https://github.com/ggml-org/llama.cpp/pull/24720 adds command-buffer reuse, reported +25% under TP. It matters
  more on our slow CPU.

## GPU-to-GPU P2P on incite: not available on the stock kernel

- `CONFIG_PCI_P2PDMA` and `CONFIG_DMABUF_MOVE_NOTIFY` are both unset in Debian's 6.12 config.
- The host bridge `8086:0e00` (Ivy Bridge-EP) isn't in `pci_p2pdma_whitelist[]` (`drivers/pci/p2pdma.c:432`).
  Only Sandy Bridge-EP `3c00` and Haswell-EP `2f00` are.
- So amdgpu's `amdgpu_dma_buf_attach` sets `peer2peer = false`, and dma-buf imports fall back to system memory.
- Enabling it would take a custom kernel. For decode it hardly matters: each allreduce is about 15 KB, so it's
  latency-bound, not bandwidth-bound.

RADV on Tahiti does expose `VK_EXT_external_memory_host` (4 KB alignment), `VK_KHR_external_semaphore_fd`,
`VK_KHR_timeline_semaphore`, `VK_AMD_buffer_marker`, `VK_AMD_device_coherent_memory` and
`VK_EXT_external_memory_dma_buf`.

## Gemma 4 12B specifics under the meta backend

The global (full-attention) layers have **one KV head**. Mainline computes
`granularity_q = lcm(n_gqa*head_dim, blk)`, which covers all of Q. So on those 8 layers the whole attention block
lands on one device, alternating per layer, and the other device gets a zero-sized slice. That isn't replication;
expect an imbalance on those layers. Sliding-window layers split normally.

ik_llama.cpp handled the same model in its own graph split (https://github.com/ikawrakow/ik_llama.cpp/pull/1922).

## GCN 1.0 / Tahiti on ggml-vulkan

- `AMD_GCN` detection (`ggml-vulkan.cpp:8-60`) needs `VK_AMD_shader_core_properties`,
  `VK_KHR_shader_integer_dot_product` and `VK_EXT_subgroup_size_control`. Tahiti RADV exposes all three.
- GCN tuning: mat-vec uses `rm_stdq=2` and `rm_kq=4` (L2883). Without accelerated int dot there's no MMVQ, so
  Q4_0 runs through the fp32 dequant mat-vec shader.
- Known D700 issues:
  - https://github.com/ggml-org/llama.cpp/issues/19610 and https://github.com/ggml-org/llama.cpp/issues/19615
    (closed): DeviceLost at longer contexts. Workaround: `RADV_DEBUG=syncshaders,zerovram` with `-ub 256`.
  - https://github.com/ggml-org/llama.cpp/issues/29596 (open, 2026-09-28): a Mac Pro 6,1 on Mesa 26.2.3 gets
    1.4 t/s with GPU offload vs 5.5 t/s on CPU.
  - https://github.com/ggml-org/llama.cpp/issues/26197: FA gibberish on a mixed AMD setup that includes an
    R9 280X (Tahiti), on the Windows driver.
- A reference point: 0cc4m's FirePro S10000 (2x Tahiti, RADV), llama-7B Q4_0, gets tg128 25.3.

## Forks

- ik_llama.cpp split mode "graph" (https://github.com/ikawrakow/ik_llama.cpp/pull/1022 and per-arch follow-ups):
  TP is built into each architecture's graph rather than a generic meta backend. It's CPU/CUDA only, so it's a
  design reference.
- TheBlueMatt `claude-tensor-par`: the most complete Vulkan TP prototype found (fetched above).
