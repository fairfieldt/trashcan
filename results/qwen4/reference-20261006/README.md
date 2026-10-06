# Qwen3.8-Flash-Next CPU reference vs llama.cpp (2026-10-06)

Model: `unsloth/Qwen3.8-Flash-Next-GGUF` Q8_0, revision `766911a6`, six shards (188.2 GB), in
`~/models/qwen3.8-flash-next/Q8_0/`. Reference: llama.cpp `b8e9ce8` (CPU backend of the Metal build on the
M3 Ultra), dumped with `tools/dump-activations`. Prompt: the 14-token ChatML "Hello there" prompt
(`qwen4::cli::REFERENCE`, identical to the qwen35 one; the Flash-Next template renders the same ids).

```sh
DUMP_DETAIL=all DUMP_ALWAYS=l_last DUMP_THREADS=24 \
  DUMP_NAMES=hc_init,ple_embd,hc_norm,hc_gate,result_norm,result_output \
  tools/dump-activations/dump MODEL ref/q8-llama-cpu cpu 10 248045 846 198 9419 1017 248046 198 248045 74455 198 248068 271 248069 271
DUMP_DETAIL=999 DUMP_ALWAYS=l_last DUMP_FULL=l_last,hc_init,ple_embd ... dump MODEL ref/q8-llama-cpu-full cpu 2 ...
trashcan qwen4 ref MODEL ref/q8-rust-cpu [--emulate-ggml]
tools/compare_activations.py ref/q8-rust-cpu ref/q8-llama-cpu --all
trashcan qwen4 layer-check MODEL ref/q8-llama-cpu-full [--emulate-ggml]
```

| check | result |
|---|---|
| greedy tokens (both modes) | identical 10/10: 9419 0 2500 628 353 1438 488 3242 30 248046 ("Hello! How can I help you today?") |
| n-gram rows (`ple_embd`), initial residual | identical (rel 0) |
| f32 reference, end to end | logits rel 8.8e-2, cos 0.99628, top-10 overlap 10/10 ([compare-f32.txt](compare-f32.txt)) |
| f32 reference, teacher-forced per layer | layer update rel 0.8e-2 to 6.2e-2, cos >= 0.998 ([layercheck-f32.tsv](layercheck-f32.tsv)) |
| ggml rounding emulated, teacher-forced | last-token update rel ~5e-7 in 10 of 48 layers; others 1e-3 to 1e-2 ([layercheck-emulated.tsv](layercheck-emulated.tsv)) |
| ggml rounding emulated, end to end | logits rel 4.0e-2, cos 0.99920 ([compare-emulated.txt](compare-emulated.txt)) |

llama.cpp's CPU path quantizes activations to Q8_0 before every Q8_0 product, rounds BF16-weight inputs to
BF16 and keeps K/V and indexer keys in f16. That alone puts 2-3% relative noise on each expert's output, which
is why the f32 reference differs by a few percent per layer. With that rounding reproduced
(`--emulate-ggml`), whole layers agree to float rounding (5e-7); the remaining layers differ through discrete
events: an activation landing on the other side of a Q8 rounding boundary, or a near-tied 10th expert
(e.g. layer 4: experts 198 and 389 at router probabilities 0.01060 and 0.01065). Over 48 layers those events
accumulate to the end-to-end differences above, without changing the greedy tokens.
