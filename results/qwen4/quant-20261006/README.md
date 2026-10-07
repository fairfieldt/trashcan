# Qwen3.8-Flash-Next quantization quality (2026-10-06)

Every candidate is made from the Q8_0 release (`Qwen3.8-Flash-Next-Q8_0-0000k-of-00006.gguf`) with llama.cpp
`b8e9ce8` on the M3 Ultra, changing one group of tensors; everything not listed stays Q8_0:

```sh
llama-quantize --allow-requantize --keep-split --imatrix imatrix_unsloth.gguf_file \
    --tensor-type ple_conv1d=f32 [--tensor-type PATTERN=TYPE ...] [--output-tensor-type q6_k] \
    Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf NAME.gguf Q8_0 24
```

(`scripts/quant.sh`; `ple_conv1d` must stay f32 or llama-quantize falls back to F16.) Unless the n-gram table
itself is requantized, the candidate's table shard is the release's (a symlink), since it differs only in header
key order.

Quality is the KL divergence of the candidate's next-token distribution from the Q8_0 model's, over a fixed
129 KB corpus (`scripts/make_corpus.py`: the model card, llama.cpp build and server docs, C++ code and Italian
prose; 39K tokens in 19 chunks of 2048), on the CPU so that no GPU kernel is involved:

```sh
llama-perplexity -m Q8_0-00001-of-00006.gguf -f corpus.txt -c 2048 -t 24 -dev none -nr --kl-divergence-base base-q8.kld
llama-perplexity -m NAME-00001-of-00006.gguf -f corpus.txt -c 2048 -t 24 -dev none -nr \
    --kl-divergence-base base-q8.kld --kl-divergence
```

`kld.tsv` has all candidates (`scripts/kld_table.py`): mean, p99 KLD, same-top-token share, RMS of the change in
the top token's probability, perplexity ratio, and the GGUF size (which includes the Q8_0 n-gram table, 51,880 MiB,
in every candidate but T-q4_0).

| candidate | `--tensor-type` overrides | mean KLD | same top token |
|---|---|---|---|
| E-iq4xs | `ffn_gate_exps=iq4_xs ffn_up_exps=iq4_xs ffn_down_exps=iq4_nl` | 0.0357 | 95.27% |
| E-q4_0 | `ffn_gate_exps=q4_0 ffn_up_exps=q4_0 ffn_down_exps=q4_0` | 0.0402 | 94.87% |
| E-mxfp4 | `ffn_gate_exps=mxfp4 ffn_up_exps=mxfp4 ffn_down_exps=mxfp4` | 0.0656 | 93.72% |
| E-iq2xxs | `ffn_gate_exps=iq2_xxs ffn_up_exps=iq2_xxs ffn_down_exps=iq4_nl` | 0.2134 | 88.65% |
| N-q4_0 | every dense matrix q4_0 (`scripts/ne-q4_0.args`) | 0.1961 | 88.75% |
| A-hc | `hc_{attn,ffn}_{down,up,inject}=q4_0 output_hc_{down,up}=q4_0` | 0.1190 | 91.36% |
| T-q4_0 | `per_layer_token_embd=q4_0` (the n-gram table) | 0.0229 | 96.26% |

The dense matrices are the sensitive part, and within them the hyper-connection mixers (about 14% of the dense
parameters) carry most of the loss. Pending at the time of writing, from `scripts/queue2.sh` and
`scripts/queue3.sh`: A-gdn (DeltaNet projections), A-attn (QSA projections), A-shexp (shared experts), G-iq4 (the
six-Mac recipe), E-iq3xxs, E-iq4mx and E-iq2mx; `kld.tsv` and this table are updated as they finish.

The E-, N-, T- and A- candidates were deleted after measuring (each regenerates with the command above in 2-26
minutes). Kept on the Mac's external disk (`/Volumes/Green/Qwen/flash-next/cand/`) and on incite
(`~/models/qwen3.8-flash-next/`, shards 1, 2, 4 and for G-iq4 also 6) for GPU validation:

| GGUF | recipe (`scripts/make_g_*.sh`) | used for |
|---|---|---|
| G-q4_0 | every matrix the GPU reads in Q4_0, head Q6_K | first GPU validation, kernel timing |
| G-q8d | experts and shared expert Q4_0, dense Q8_0, head Q6_K | Q8_0 dense kernels |
| G-iq4 | experts and shared expert IQ4_XS gate/up, IQ4_NL down, dense Q8_0, head Q6_K (119,231 MiB) | the six-Mac recipe |

## Expert popularity

`expert-popularity.txt` (`scripts/expert_popularity.py` over the imatrix's per-expert counts, 406M routed
activations): caching each layer's most used 60% of experts covers 83.9% of activations on average (79.9% in the
worst layer), 65% covers 87.0% (83.5%). This is the hit rate a static VRAM cache would see on text like the
imatrix's calibration set; four Macs with host-RAM experts can cache about 63%.
