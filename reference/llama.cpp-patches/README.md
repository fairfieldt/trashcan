# llama.cpp patches from the 2026-10-02 investigation

These are local-only patches against upstream llama.cpp 99b95488c (2026-10-03), kept here because the `llama.cpp/`
checkout isn't tracked. They're reference material: the engine doesn't use llama.cpp. Don't submit them upstream as-is
(`llama.cpp/AGENTS.md` forbids AI-written PRs and comments; a human must own any upstream change).

| patch | what | status |
|---|---|---|
| 0001 | `ggml-backend-meta.cpp` `handle_reshape`: check the complete-row case before the "split on last dim" shortcut. Fixes `-sm tensor` asserting (`GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0)`) on Gemma 4 token generation, where the 1-KV-head `Kcur` `[512, 1]` is reshaped to `[512, 1, 1]` and RMS-normed. | verified: correct output with the generic allreduce |
| 0002 | `llama-model.cpp`: split a tied `token_embd` output copy by vocab under `-sm tensor`, like `output.weight`. Without it, Gemma's 787 MiB Q6_K head is replicated on every GPU. | untested |
| 0003 | `ggml-vulkan.cpp`: `GGML_VK_COMM_DISABLE` env var to skip #25051's Vulkan allreduce. Applies on top of https://github.com/ggml-org/llama.cpp/pull/25051 (commit 29395e39b). | debug aid |

Also found, not patched: #25051's small-tensor allreduce path zeroes an inactive shard's outgoing host buffer but not its
local tensor, so the GPU without attention heads on Gemma 4's global layers adds the peer's partial onto stale memory
(corrupted but coherent output). CUDA's allreduce does `cudaMemsetAsync(tensors[i]->data, 0, ...)` for that case.
