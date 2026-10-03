# Findings log

Newest first. Numbers are from incite unless noted; raw outputs are under `results/`.

## 2026-10-02: first baselines (stock Debian kernel 6.12.111, upstream 99b9548)

`scripts/bench-gemma.sh master-stock`, Gemma 4 12B QAT Q4_0, `-ngl 99`, both D700s:

| split | fa | pp512 t/s | tg128 t/s |
|---|---|---|---|
| layer | 0 | 36.1 | 12.27 |
| layer | 1 | 36.9 | 9.44 |
| tensor | 1 | 118.2 | crash (fixed below) |

- **Clocks are fine now.** Both cards hold 850 MHz core / 1370 MHz memory under load (max 77 C), so the
  patched amdgpu works. Idle drops to 300/150.
- **tg is bandwidth-starved.** Layer-split tg 12.27 t/s means about 85 GB/s of weight streaming, 32% of one
  card's 264 GB/s. The Q4_0 mat-vec shader reads 18-byte blocks with 16-bit loads, three load instructions per
  8 weights. That likely hits GCN 1.0's per-lane address rate before DRAM bandwidth. A repacked layout that
  allows 128-bit loads is a candidate. Check with `test-backend-ops perf -o MUL_MAT` first.
- **FA costs 23% of tg on GCN 1.0** (12.27 → 9.44), and `-sm tensor` forces FA on.
- **Tensor-split prefill is already 3.2x layer split**, even with the blocking host-staged allreduce.

## Bugs found

1. **`-sm tensor` asserts on Gemma 4 token generation**
   (`ggml-backend-meta.cpp:547 GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0)`).
   - Gemma 4's global layers have 1 KV head. `Kcur` `[256, n_tokens]` is reshaped to `[256, 1, n_tokens]` and
     then RMS-normed.
   - With `n_tokens == 1` the source has `ggml_n_dims == 1`, so `handle_reshape`'s "split is on the last dim"
     shortcut fires first. It keeps axis 0 for the 3D result.
   - The complete-rows branch below it, which moves the split to the singleton head axis, was meant for
     exactly this case. Prefill reaches it; decode doesn't.
   - Fix: check complete rows first. Local commit d99061874 on `tp-vulkan`.
2. **The tied output head is replicated on both GPUs.**
   - Gemma ties `output` to `token_embd`. The GPU copy is named `token_embd.weight`, misses the
     `output\.weight` rule in `llama_meta_device_get_split_state`, and defaults to MIRRORED.
   - So each GPU stores the full 262144x3840 head and computes all 262K logits every token.
   - Fix: give it the same vocab split (axis 1) as `output.weight`. The input embedding stays on the CPU
     (`llama-model.cpp:1606`), so only the output copy is affected. Local commit 226f2232d, not yet measured.

## Platform facts that shape the design

- No dedicated transfer (SDMA) queue family on RADV/Tahiti: 1 graphics queue plus a compute family with 2
  queues. ggml-vulkan uses the second compute queue as its "transfer" queue, so #25051's ring path (allreduces
  of 2 MB or more) is eligible.
- PCIe P2P between the cards needs a custom kernel. `CONFIG_PCI_P2PDMA` and `CONFIG_DMABUF_MOVE_NOTIFY` must both
  be on: without MOVE_NOTIFY, `amdgpu_dma_buf_pin` always pins exports into GTT. The host bridge `8086:0e00` also
  has to be whitelisted. See `kernel/`.
