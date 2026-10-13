# Flash-Next MTP rounds: sampling, the draft step, the verification reply (2026-10-13)

Follows [results/qwen4/mtp-20261012](../mtp-20261012/README.md) (the MTP block drafting on the driver's CPU, adaptive
rounds of 0-3 drafts, greedy chat replies 18.3 -> 21.6 tok/s on six Macs). Branch `worktree-agent-a173c86cfe38f66c7`
from main `4c1346d`.

## Sampling (`spec::Draws`)

Main kept a draft when the sampled token equalled it, with the draft the drafter's argmax: correct (the output is the
target's sample, drawn as in plain decoding), but a greedy draft is accepted with probability p(argmax q) only.
Standard speculative sampling (accept x ~ q with min(1, p(x)/q(x)), otherwise sample normalize(max(0, p - q))) is
accepted with sum_v min(p_v, q_v), the most any coupling of a draft from q with a sample from p allows, but which
tokens it emits depends on whether a position had a draft: with adaptive rounds (the drafts per round follow measured
costs) a seed would no longer give reproducible output, and `scripts/test-server.py` checks that it does.

The engine now picks every token, plain or speculative, with the Gumbel-max trick on the sampler's distribution
(`Sampler::probabilities`: temperature, then top-k, then top-p over the candidates, renormalized), with the noise of
token id v at position t a hash of (seed, t, v): token(t) = argmax_v ln p_t(v) + g(seed, t, v). A draft for position t
is the same argmax over the drafter's candidates and distribution q_t (same temperature, top-k and top-p) with the same
noise, and is kept when the target's pick equals it. So:

- the output is plain decoding's for every seed, whatever the drafts and the round sizes (greedy decoding is the
  temperature-0 case: the top candidate, as before);
- each token is an exact sample of the target's distribution (Gumbel-max over independent per-token noise), and
  positions are independent;
- a draft is accepted with probability sum_v 1 / sum_w max(p_w / p_v, q_w / q_v) (the probability that both
  argmaxes pick v, summed over v): between the greedy draft's p(argmax q) and standard speculative sampling's
  sum_v min(p_v, q_v); exactly the latter when p and q share two tokens, close to it for real distributions.

### Acceptance by temperature (CPU reference)

`qwen4 mtp accept --temperature T --top-k 64 --top-p 0.95 --depth 0` samples 256 tokens of each chat prompt from the
trunk (CPU reference, G-iq4) and saves the trunk's top 64 candidates at every step; `qwen4 mtp replay --teacher` runs
the engine's drafter (64K draft vocabulary, 4-bit dense matrices) along each run, chained drafts fed the run's tokens
(the drafts a round reaches when the earlier ones were accepted), and saves its top 64 candidates at depth 1-3.
`sampling/sampling.py` computes the acceptance of each depth for the three ways of drafting and the expected tokens per
round with k drafts (1 + a1 + a1 a2 + ...), per step, then averaged (`sampling/summary-k64.txt`; `steps.csv.gz` has
every step). Each temperature samples its own run, so compare methods within a temperature:

| T | drafts | a1 | a2 | a3 | tokens/round k=1 | k=2 | k=3 |
|---|---|---|---|---|---|---|---|
| 0 | argmax (all methods) | 0.882 | 0.819 | 0.793 | 1.882 | 2.617 | 3.222 |
| 0.6 | greedy (main) | 0.864 | 0.803 | 0.763 | 1.864 | 2.573 | 3.140 |
| 0.6 | **coupled (the engine)** | 0.878 | 0.810 | 0.767 | 1.878 | 2.606 | **3.192 (+1.7%)** |
| 0.6 | standard speculative sampling | 0.880 | 0.811 | 0.770 | 1.880 | 2.610 | 3.200 (+1.9%) |
| 0.8 | greedy (main) | 0.832 | 0.773 | 0.750 | 1.832 | 2.496 | 3.022 |
| 0.8 | **coupled** | 0.859 | 0.791 | 0.757 | 1.859 | 2.557 | **3.116 (+3.1%)** |
| 0.8 | standard | 0.863 | 0.795 | 0.762 | 1.863 | 2.566 | 3.132 (+3.6%) |
| 1.0 | greedy (main) | 0.829 | 0.781 | 0.745 | 1.829 | 2.489 | 3.006 |
| 1.0 | **coupled** | 0.871 | 0.805 | 0.760 | 1.871 | 2.581 | **3.145 (+4.6%)** |
| 1.0 | standard | 0.877 | 0.813 | 0.768 | 1.877 | 2.598 | 3.174 (+5.6%) |

The coupled drafts recover 80-90% of what standard speculative sampling would add over greedy drafts, and keep seeded
output reproducible and equal to plain decoding. Top-k 20 instead of 64 changes nothing measurable
(`summary-k20.txt`), and drafting at 0.8x, 0.9x or 1.1x the target's temperature is slightly worse than the same
temperature (`summary-scale*.txt`). Sampling costs little acceptance on these replies (each temperature's run is a
different text; within the same text the gap between greedy and sampled acceptance is what the table shows).

### Correctness

- Unit tests (`spec::tests`): the keyed picks of three candidate lists (temperatures 0.6-1, top-k 8-64, top-p 0.95-1,
  a tie-heavy flat list) follow `Sampler::probabilities` (chi-square of 200,000 positions under the 0.9999 quantile);
  pairs of consecutive positions follow the product distribution (no correlation between positions), and seeds
  differ; a draft with the same noise agrees with the target's pick at the closed-form rate (200,000 positions, within
  0.005) and above a greedy draft's p(argmax q); greedy picks the top candidate. A Python replica of the hash on a
  real trunk distribution (water at T = 1, the third token: p = 0.879 / 0.121) gave 0.878 / 0.122 over 20,000 seeds.
- Speculative decoding returns plain decoding's tokens for every seed and temperature: the two-process CPU chain on
  the Mac (layers 0-23 and 24-47 over TCP, the drafts on the CPU head; greedy and T = 1, two seeds each) and the six-Mac
  chain below (greedy, T = 0.6, 0.8, 1.0 on the chat prompts, 30 seeds at T = 1).

## The draft step (`draft.rs`)

Profiled on incite with timers around every phase (`tools/instrument2.py`, a temporary patch; `draft-bench/db2`):
besides the products, a step spent its time in serial loops on the main thread between the parallel ones (10,240
sigmoids per mixer, the query norms and rotations and the output gate of 24 heads, the experts' SwiGLU and its
quantization, a libm call per value to round activations to Q8_0, which baseline x86_64 has no instruction for), and
from ~2K positions in the block's attention (block scores and a full sort on one thread, two dependent accumulator
chains per score). The step now does (`draft.rs`, all with the same draft candidates as before: replays of two runs give
identical top-64 candidates at every step and depth on the Mac and on incite, and unit tests compare the new kernels with
the old code bit for bit):

- a mixer's up rows of the four streams, the read gates and the stream mean in one parallel loop over ranges of the
  sublayer input;
- per attention head, the query's norm and rotation (one sine/cosine table per position), the attention and the output
  gate inside the head's task; block scores in parallel and a partial selection of the 512 best blocks;
- attention with AVX (incite has AVX without FMA): four keys' dot products at a time, each accumulated in the scalar
  code's order (lane j sums dimensions j, j + 8, ...), and the values four positions per pass over the output;
- the experts' SwiGLU and its Q8_0 quantization inside the gate/up tasks (parts of two Q8_0 blocks);
- rounding to Q8_0 by clamping, truncating and stepping by the exact remainder (vectorizable, equal to `round()`), no
  f32 copy where only quantized matrices read the activations, finer mixer and router tasks.

Workers that spin for the next job before sleeping (`tools/spinpool.py`) did not help (as the previous README found),
nor did they hurt; not kept.

`qwen4 mtp draft-bench MODEL MTP --context 8192 --start S --steps 64 --batch 1 --threads 24 --skip-head-rows 43008`
(the CPU's share of a draft step on the chain: head rows 43008-65535), p50 of 64 steps, main and this branch
alternating in one lease (`draft-bench/db2`, `db3`):

| start position | main 4c1346d | this branch | attention (main / branch) | experts | mixers (each) | head and final mixer |
|---|---|---|---|---|---|---|
| 512 | 7.54-7.84 ms | **4.81-4.88 ms** | 1.95-2.07 / 1.12 | 1.67 / 1.01 | 0.50 / 0.27 | 1.99 / 1.61 |
| 2048 | 7.23 | **5.14** | 2.96 / 1.63 | 1.24 / 0.97 | 0.37 / 0.25 | 1.72 / 1.54 |
| 3072 (sparse) | 9.49-9.54 | **5.35-5.77** | 4.39-4.65 / 1.87-2.01 | 1.55 / 0.96-1.07 | 0.47 / 0.24-0.28 | 1.94 / 1.56-1.61 |
| 4096 | 11.58 | **5.67** | 5.91 / 2.02 | 2.05 / 1.02 | 0.62 / 0.28 | 2.34 / 1.61 |

The block over a prompt window (64-token batches) went from ~215 to ~360 tokens per second. Main measured 7.5 ms here
against 5.9 in the previous README's draft-bench (session 2) with the same drafter code: incite's state differs between
days, so only the alternating comparison counts; on the chain the draft step is measured in place (below). At 512 a
step now reads 162 MB in 4.8 ms, 34 GB/s: the head (65 MB, 1.3 ms) and the experts' gate/up and down rows run at
39-49 GB/s, the projections into attention at 39 GB/s; the mixers (0.27 ms each for 4 MB) and the input projection
(0.46 ms for 8 MB, four streams decoding each weight block four times) are what remains below the bandwidth.

## The verification reply (`net.rs`)

A verification batch's reply carried the final hidden states of its rows and their residual rows (40 KiB a row, the
block's input) in one frame up the chain: every machine received and forwarded both before it could score its head
rows, so the rows (40-160 KiB per hop for 1-4 tokens, five hops) were on the critical path of every round, k = 0
rounds included (a k = 0 round, a plain step through the verification path, cost 0.4-1.4 ms more than a plain step in
the previous sessions). Now the hidden states go up first, each machine starts scoring them, then forwards the
residual rows as a frame of their own while its GPUs score, then the merged candidates follow (`send_rows`,
`recv_rows`, `verify_node`, `Chain::verify_with`). The driver needs only the accepted row and only for its next draft
step, after the candidates.

## Unsure drafts (`spec::UNSURE`)

The drafter's confidence predicts acceptance well. On the greedy replays of the three chat prompts (768 steps, depths
1-3, `gate/confsim-t0.txt`), the share of drafts the trunk accepts by the drafter's top probability (softmax at
temperature 1 over its 64 candidates):

| top probability | < 0.3 | 0.3-0.5 | 0.5-0.7 | 0.7-0.8 | 0.8-0.9 | 0.9-0.95 | 0.95-0.99 | >= 0.99 |
|---|---|---|---|---|---|---|---|---|
| accepted | 0.16 | 0.22 | 0.53 | 0.64 | 0.76 | 0.83 | 0.90 | 0.99 (53% of drafts) |

The policy picks a round's number of drafts from one pooled acceptance rate before drafting; it cannot skip a draft it
is unsure of. `gate/confsim.py` replays rounds on these traces (round costs 55 / 91 / 108 / 128 ms for 1-4 tokens, 5.1
ms draft steps; tok/s, mean of the three prompts):

| rule | greedy (T = 0) | same, verification 40 / 52 / 60 / 70 ms |
|---|---|---|
| the pooled policy's choice | 22.45 | 38.11 |
| always 3 drafts | 22.66 | 38.07 |
| an unsure draft (top < 0.6) is verified as the round's last | 23.22 | 38.49 |
| **an unsure draft is not verified, the round keeps the drafts before it** (its draft step still counted) | **23.89 (+6.4%)** | **39.42 (+3.4%)** |
| same, threshold 0.5 / 0.7 / 0.8 | 23.48 / 23.66 / 23.37 | 38.95 / 38.67 / 37.60 |

and on the sampled replays (`gate/confsim2.py`, coupled acceptance probabilities drawn 20 times; tok/s):

| rule | T = 0.6 | T = 0.8 | T = 1.0 |
|---|---|---|---|
| always 3 drafts | 22.22 | 21.28 | 21.53 |
| unsure draft verified as the last (top < 0.6) | 22.70 | 21.80 | 22.24 |
| **unsure draft not verified** (top < 0.6) | **22.95** | **22.06** | **22.56** |

A dropped draft saves its verification slot (36 ms for a first draft, 17-20 ms for a later one) and the next round
starts from the trunk's residual row, whose first draft is accepted more often than a chained one. The engine drops a
draft whose top candidate is below 0.6: an unsure first draft makes the round a plain step with that draft standing in
for its shadow draft; an unsure later draft ends the drafting, and when every draft before it is kept it is still
compared with the next token, so the acceptance estimate sees it. Rounds that measure a batch size's cost (`REFRESH`)
verify all their drafts.

## End to end (six Macs)

`qwen4 mtp spec MODEL --gpus 0,1 --nodes indite.tb0:7071,inflight.tb1:7071,alight.tb0:7071,invite.tb1:7071,enlight.tb0:7071
--head-split 12 --context 32768 --batch 128 --ngram disk --mtp mtp-G-iq4.gguf --draft-tokens 3 --tg N --runs R
[--temperature T --top-k 64 --top-p 0.95 --seed S] PROMPTS`: per prompt and run plain decoding, then speculative,
alternating, identical output required (sampled runs use seeds S, S + 1, ...); rates exclude the prompt. Run through
`chain6.sh` and `tools/q4run.py` with clocks logged (850/1370 MHz on every GPU while working, `chain*/clocks-*`);
commands in `chain*/commands.log`, summaries (`summary.py`) in `chain*/summary.txt`.

**Session 1** (`chain1/`, binary `23135cb`: the sampling, the draft step, the reply frames, and an unsure draft verified
as the round's last; main `4c1346d` alongside; `tools/mtp2-chain1.sh`):

| run | plain tok/s | MTP tok/s | gain | accepted/draft | rounds k = 0/1/2/3 | ms per round k = 0/1/2/3 | draft step |
|---|---|---|---|---|---|---|---|
| chat, greedy, main | 18.30 | 21.55 | +17.8% | 0.70 / 0.84 / 0.71 | | 54.6-55.4 / 90.9-92.9 / 107.5-109.5 / 128.1-128.8 | 5.77 ms |
| chat, greedy, branch, unsure draft verified last | 18.34 | **22.27** | **+21.4%** | 0.74 / 0.86 / 0.76 | | 54.6-55.0 / 90.0-91.6 / 107.3-109.3 / 126.9-128.2 | 5.05-5.10 |
| chat, greedy, branch, every chosen draft verified | 18.38 | 22.10 | +20.2% | 0.70 / 0.85 / 0.69 | | | 5.01-5.04 |
| reference (64 tokens), main | 18.49 | 18.17 | -1.7% | 0.57 | 159/1/12/1 | 54.7 / 91.0 / 107.5 / 134.8 | |
| reference, branch | 18.53 | 18.15 | -2.0% | 0.50 | 154/4/12/2 | 54.2 / 89.4 / 106.4 / 129.9 | |
| 3K-token document, main | 16.15 | 16.23 | +0.5% | 0.62 | 61/1/53/3 | 62.9 / 103.0 / 123.6 / 150.5 | 6.55 |
| 3K-token document, branch | 16.14 | **17.13** | **+6.2%** | 0.69 | 51/17/41/4 | 62.4 / 100.1 / 122.6 / 150.1 | 5.21 |

(chat: water / csv / transformer; per-prompt rows in `chain1/summary.txt`.) The draft step on the chain went from 5.77 to
5.05 ms at chat lengths and from 6.55 to 5.21 ms at ~3.1K positions; a k = 0 round (a plain step with the shadow draft)
costs 0.2-0.4 ms more than a plain step (54.2 against 53.97 ms on the reference prompt; main 54.7 against 54.08), and
2-4-token rounds 0.5-5 ms less. Sampled, same texts for both drafting methods (the keyed sampler makes the plain output
the same; seeds 1 and 2):

| T | plain | main's drafting (greedy drafts) | branch (coupled drafts, unsure verified last) | accepted/draft main / branch |
|---|---|---|---|---|
| 0.6 | 18.28-18.33 | 21.19 (+15.9%) | 21.07 (+15.0%) | water 0.67 / 0.67, csv 0.83 / 0.84, transformer 0.65 / 0.67 |
| 0.8 | 18.37-18.41 | 21.19 (+15.2%) | 21.34 (+16.2%) | 0.61 / 0.66, 0.83 / 0.84, 0.69 / 0.71 |
| 1.0 | 18.31 | (the run failed to start: a node refused the connection) | 21.30 (+16.3%) | -, 0.80, 0.69 |

The coupled drafts are accepted more often, as the CPU study predicted, but over 2 x 3 replies of 192 tokens the
end-to-end difference is within the noise (-0.6% at 0.6, +0.7% at 0.8); verifying an unsure draft as the round's last
turned many rounds into 1-draft rounds (water at 0.6: 29 against 3), which session 2 replaces with dropping it.

Every run of session 1 gave plain decoding's tokens (greedy, T = 0.6, 0.8, 1.0, and 30 seeds at T = 1 on the reference
prompt, 24 tokens each, 14 distinct outputs). The server (`tools/q4serve-test.sh ... --mtp mtp-G-iq4.gguf
--draft-tokens 3`, default temperature 1): every `scripts/test-server.py` check passes, seeded sampling reproducible
included; `q4-turns.py` 57 tokens in 2.72 s, then 64 (78 of 99 prompt tokens reused) in 2.51 s (session 4: 2.83 and
2.53 s).

**Session 2**: DRAFT

## Reproduction

```sh
# Mac (CPU reference): sampled runs, the drafter replayed along them, acceptance by temperature and method
sampling/study.sh                                   # accept --temperature T --depth 0; replay --teacher (64 candidates)
python3 sampling/sampling.py DIR [--top-k 20] [--draft-scale 0.9] [--csv steps.csv]
python3 gate/confsim.py DIR; python3 gate/confsim2.py DIR 1.0     # unsure drafts: fixed rules on the traces
python3 gate/policysim.py DIR --t 0 [--verify 40,52,60,70]          # the engine's policy variants on the traces
# Mac: two-process CPU chain (identity, greedy and sampled); the drafter's candidates against main's
tools/cpu2.sh 0 1; tools/same.sh
# incite (under a lease): draft-step timing, main and this branch alternating; the instrumented profile
tools/mtp2-db-ab2.sh; python3 tools/instrument2.py WORKTREE   # temporary timers, then draft-bench
# x86 unit tests (the SSSE3/AVX drafter kernels) built in the kbuild VM and run on incite
tools/xtest.sh; ./trashcan-mtp2-tests qwen4::draft qwen4::spec
# six Macs
chain6.sh mtp2-chain1 50 OWNER -- tools/mtp2-chain1.sh; tools/fetch_session.sh chain1 NAMES...
python3 summary.py chain1/*.json
```

## What is left

- Cheaper 2-4-token verification batches remain the largest lever (another agent's kernels): a round's verification is
  ~90% of its time (k = 3: 127 ms of verification against 15 ms of drafting), and the policy picks up cheaper batches
  by itself (`gate/policysim.py` with 40 / 52 / 60 / 70 ms rounds: 39 tok/s on the chat replies).
- The block's context in a conversation: a request that continues the live state prefills its new tokens in 16-token
  chunks, and the block reads only the last chunk's rows (`Speculator::prefill`, `first` reset), so from the second turn
  on it no longer sees the earlier turns (the 3K-token document lost 11% of its tokens per round to a 128-token window in
  the previous README). Keeping the block's keys needs the residual rows of every new token: one chunk for short
  continuations, or the last stage keeping the rows of a prefill's chunks (pipeline/prefill code).
- The draft step's remaining time below the bandwidth: the mixers (0.27 ms each for 4 MB) and the input projection
  (the four streams decode each weight block four times, 0.46 ms for 8 MB); at long contexts each of the 24 attention
  heads reads its KV head's 2,051 positions from L3 (100 MB per step at 3K positions, 0.73 ms), where the 12 heads
  sharing a KV head could read each row once (needs a different summation order than the scalar reference).
- Degenerate text (the reference prompt) still costs ~2%: the rounds that measure each batch size at the start and
  the drafts tried before the acceptance estimate falls.
- Standard speculative sampling would add ~1% more acceptance than the coupled drafts at T = 1 (CPU study) but makes
  seeded output depend on the round sizes, which follow measured costs; not done.
