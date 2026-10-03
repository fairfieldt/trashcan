# Findings log

Newest first. Numbers are from incite unless noted; raw outputs are under `results/`.

## 2026-10-02 (implementation): independent Rust/Vulkan engine

The target forward pass, tokenizer, chat, sampling, prefill and native BF16 MTP assistant are implemented.
Build and validation recipes, the pinned assistant manifest and milestone status are in `implementation.md`.
The executable runs without llama.cpp. The reference checks exercise single-token, four-token, GEMM,
CPU-relay and MTP paths against the saved activations and continuation.

### Transport and memory findings

- **Device addresses accidentally serialize the two GPUs on this RADV version.** Mesa 25.0.7's
  `radv_device.c` enables `use_global_bo_list` for buffer device addresses and descriptor indexing /
  update-after-bind. Every imported peer allocation then participates in each submission's implicit
  synchronization. Fixed descriptors with all these features disabled restore overlap.
- **Disjoint receive slots must be distinct DMA-BUF allocations.** Ranges in one allocation share implicit
  fences; separate exported buffers are necessary for the pre-recorded 97-segment timeline schedule.
- **The 256 MiB BAR limits large-batch P2P.** Ninety-six receive slots at batch 512 need 720 MiB. Slots
  are capped at batch 128 / 180 MiB. Larger batches now reuse two 512-token buffers (15 MiB), with GPU0/GPU1
  submissions interleaved per segment. Implicit fences then refer to preceding consumers, allowing reuse
  without CPU fence waits. This raised pp512 from about 123 to 159 tok/s in the transport trial. CPU relay
  remains a debug path; its preallocated staging was faster than per-hop allocation and recording.
- At context 8192 / batch 512, measured target allocations are 4552 MiB/GPU; the assistant adds 415 MiB.
  The 32K combined budget is about 5351 MiB/GPU, calculated rather than validated at that context.
- The user authorized temporary CPU PM QoS latency 0. `scripts/latency-run.py` holds the request only for
  its command. It reduced median decode semaphore gaps from about 123–133 us to about 41–42 us before
  the last dispatch-fusion changes. Display-manager configuration was not changed.

### Kernel and assistant findings

- Decode Q4_0 uses f32 scales followed by packed quants (20 bytes/block); the Q6_K head expands to signed
  bytes with f32 signed scales (320 bytes/256 weights). Original embeddings remain Q6_K on CPU.
- A 64×64×32 fp32 GEMM reached about 2.2 TFLOP/s per GPU in the FFN microbenchmark. Four-token
  verification still achieves much less compute throughput, limiting speculative gain.
- The final streamed-P2P benchmark reaches pp512 161.1 tok/s, satisfying M4. tg128 is 21.2 tok/s,
  below M3's 40–50 target. Three paired greedy chats have identical outputs and MTP gains of 1.22×,
  1.01× and 1.45× (combined 1.20×), below M5's 1.5× target. Maximum recorded temperatures are 84/82 C.
- Two-token FFN work is 41–49% faster with the four-row kernel; the final selector uses it for 2–4 tokens.
  Its reference checks pass. One-draft photosynthesis still runs at 22.89 tok/s versus 23.51 with three
  drafts, so the default stays at three. Acceptance and verification amortization both matter.
- Independent Q/K/V and gate/up operations share a trailing barrier. Post-normalization, residual/scaling
  and the following RMS normalization share a dispatch, with the same reduction order as separate norms.
- The pinned assistant contains no centroid tensors, despite centroid settings in its config. Its full
  BF16 vocabulary head is used. Its logit softcap is null; the target's is 30.
- Every draft step keeps the target query position and visible cache fixed, and reads target layers 46
  and 47. Stochastic verification uses P/Q acceptance and the residual distribution on rejection, not
  greedy comparison. A 50,000-trial unit test checks the target sampling law; seeded live runs repeat.
- Raw repetitive prompts can become unstable, as with the earlier llama.cpp investigation. The long
  verification fixture uses a rendered chat. Serial/chunked rounding can alter a near-tied greedy decision;
  the checker reports their common prefix separately and checks verification/rollback against the same
  chunked-prefill cache.

Final benchmark outputs and source hashes are retained under `results/engine/delivered-benchmark/`;
correctness logs are `delivered-validation-suite.log` and `delivered-ring-check.json`.

## 2026-10-02 (later): cross-GPU transport, P2P kernel, llama.cpp TP bugs, decision to build our own engine

### Cross-GPU sync and copy (`bench/xgpu`, `results/xgpu/`)

| one-way hop | stock kernel | P2P kernel |
|---|---|---|
| submit + fence wait round trip, one GPU (p50) | 41-65 us | 64-73 us |
| CPU relay (wait on fence, then submit to the other GPU) | 67.5 us | 73 us |
| CPU proxy (#25051 fallback: pre-submitted, host-signaled) | 72.6 us | 65 us |
| **exported OPAQUE_FD timeline semaphores, pre-submitted** | **44.4 us** | **41.9 us** |

- RADV on Mesa 25.0.7 can export and import timeline semaphores across the two GPUs, so #25051's native path
  works.
- 02:00.0, the card that drives the gdm login screen, has a slower fence round trip than 06:00.0.

**GPU-side spin-waiting across GPUs is impossible on GCN 1.0.**

- A shader ping-pong between the GPUs never completes, in any of these modes: `coherent volatile`, Vulkan
  memory-model atomics, RMW atomics, `nontemporal`. Nor with any memory: cached host import, write-combined GTT
  dma-buf, or P2P VRAM.
- A CPU thread watching host memory sees GPU0's store only when its dispatch ends (about 760 ms later), so the
  writer's L2 holds the store until the end-of-submission writeback.
- The same shader with both roles on one GPU runs at 0.6-0.9 us per hop.
- So cross-GPU sync has to go through the kernel: semaphores, about 42 us per hop.

**PCIe P2P** (custom kernel `6.12.111-p2p`: `CONFIG_PCI_P2PDMA`, `CONFIG_DMABUF_MOVE_NOTIFY`, Ivy Bridge-EP host
bridge whitelisted):

- A dma-buf exported from GPU1's VRAM and imported on GPU0 stays in VRAM (GPU1's own copies of it run at
  100 GB/s; on the stock kernel it migrates to GTT at 5-8 GB/s).
- GPU0 writing into GPU1's VRAM: 15.6-15.9 GB/s at 256 KiB-1 MiB, 9.3 GB/s at 8 MiB. Reading across: 2-3.7 GB/s.
- GPU writes into shared host memory (`external_memory_host`, a cached/snooped type): only 1.4 GB/s at 8 MiB;
  host -> VRAM about 6.5 GB/s.
- Takeaway: allreduces should push partials into the peer's VRAM.

### llama.cpp tensor split on Gemma 4 (now reference only)

1. **Fixed locally:** the `handle_reshape` assert at n_tokens == 1 (see below).
2. **Tied output head replicated** on both GPUs (see below).
3. **#25051's Vulkan allreduce corrupts output.** The text is coherent but has injected digits and repeated
   fragments: "two-2 sentences", "useds2".
   - It's deterministic, and the same with native and proxy sync.
   - With the meta backend's generic allreduce instead (`GGML_VK_COMM_DISABLE`, debug branch), the output is
     correct and matches layer split. tg is 6.4 vs 9.0 tok/s.
   - Root cause: in its small-tensor path, an inactive shard has no `GGML_TENSOR_FLAG_COMPUTE`. On Gemma 4's 1-KV-
     head global layers, one GPU has no attention heads, which makes its shard inactive. For such a shard the path
     zeroes what it sends but not its own local tensor, so it adds the peer's partial onto stale memory. CUDA's
     allreduce `cudaMemsetAsync`s the local tensor in the same case.
   - Not fixed: we stopped patching llama.cpp here.

- Raw (no chat template) prompts make this instruction-tuned model degenerate ("0111..."), on CPU too. Use the chat
  template when comparing outputs.
- Layer split with flash attention on GCN 1.0 gives the same text as without it.

**Decision:** build a purpose-built engine in Rust. See `engine-plan.md`.

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
