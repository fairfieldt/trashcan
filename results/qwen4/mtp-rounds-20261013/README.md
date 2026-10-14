# Flash-Next MTP rounds: sampling, the draft step, the verification reply (2026-10-13)

Follows [results/qwen4/mtp-20261012](../mtp-20261012/README.md) (the MTP block drafting on the driver's CPU, adaptive
rounds of 0-3 drafts, greedy chat replies 18.3 -> 21.6 tok/s on six Macs). Branch `worktree-agent-a173c86cfe38f66c7`,
started from main `4c1346d` (session 1), measured on main `7da6f93` (session 2), `2293550` (session 3) and `eb4dbeb`
(session 4: the 1-4-token verification kernels).

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

Main's policy picks a round's number of drafts from one pooled acceptance rate before drafting, and verifies every draft
it chose. Fixed rules replayed on these traces (`gate/confsim.py`; round costs 55 / 91 / 108 / 128 ms for 1-4 tokens,
5.1 ms draft steps; tok/s, mean of the three prompts; `gate/confsim2.py` draws the coupled acceptance probabilities 20
times on the sampled replays):

| rule | T = 0 | T = 0.6 | T = 0.8 | T = 1.0 |
|---|---|---|---|---|
| always 3 drafts | 22.66 | 22.22 | 21.28 | 21.53 |
| an unsure draft (top < 0.6) verified as the round's last | 23.22 | 22.70 | 21.80 | 22.24 |
| **an unsure draft not verified**, the round keeps the drafts before it (its draft step counted) | **23.89** | **22.95** | **22.06** | **22.56** |
| same, threshold 0.5 / 0.7 / 0.8 (T = 0) | 23.48 / 23.66 / 23.37 | | | |

A dropped draft saves its verification slot (36 ms for a first draft, 17-20 ms for a later one), and the next round
starts from the trunk's residual row, whose first draft is accepted more often than a chained one. With drafts dropped,
one pooled rate misleads the policy (unsure drafts drag it down, so it drafts fewer even when the sure ones are
accepted): the policy now keeps two rates, the share of drafts the drafter is sure of (c) and the acceptance of those
(a), and rates k drafts by their expected tokens 1 + ca + ... + (ca)^k against the expected draft steps (1 + c + ... +
c^(k-1)) and verification (m < k drafts verified with probability c^m (1 - c)). `gate/policysim.py` replays the
engine's rounds (its policy, the shadow draft of k = 0 rounds, sizes measured at the start) on the traces, the
degenerate reference reply included (`tools/refrun.sh`); costs of session 1 (5.05 ms draft steps, 54.2 / 90 / 107.3 /
127.8 ms rounds; plain 18.52 tok/s):

| policy | T = 0 chat | T = 0 reference | T = 0.6 | T = 0.8 | T = 1.0 | T = 0, rounds 40 / 52 / 60 / 70 ms |
|---|---|---|---|---|---|---|
| main (pooled, every draft verified) | 22.37 | 18.25 | 22.44 | 21.72 | 21.73 | 37.68 |
| session 1 (unsure verified as the last) | 22.61 | 18.38 | 22.52 | 21.82 | 22.04 | 38.09 |
| pooled, unsure dropped | 23.00 | 18.48 | 22.80 | 22.04 | 22.18 | 38.41 |
| **two rates, unsure dropped (the engine)** | **23.67 (+5.8%)** | 18.28 | 22.76 (+1.4%) | 21.99 (+1.2%) | **22.37 (+2.9%)** | **39.04 (+3.6%)** |

Session 1 measured the second row at +0.8% over the first on the chain (simulated +1.1%). The engine drops a draft
whose top candidate is below 0.6 (0.5 and 0.7 simulate 23.22 and 23.49 at T = 0): an unsure first draft makes the round a
plain step with that draft standing in for its shadow draft; an unsure later draft ends the drafting. Shadow drafts
(their confidence from all 64K draft rows) feed both rates; rounds that measure a batch size's cost (`REFRESH`) verify
all their drafts.

The policy reads the batch costs from its own measurements, so cheaper verification needs no change there; the
threshold is fixed. With main `eb4dbeb`'s 1-4-token kernels (49.4 / 67 / 83 / 106 ms rounds, plain 49.2 ms;
`tools/policysim-eb4dbeb.sh`, `gate/policysim-eb4dbeb.txt`) 0.6 is still the best threshold at T = 0 (28.80 tok/s
against 28.53 at 0.5, 28.17 at 0.7, 27.94-28.15 at 0.3-0.4) and as good as 0.5 at T = 1 (27.20 / 27.29). Gates that
weigh each draft against the measured cost of verifying it (`gate/policysim2.py`: keep draft j while P g(c_j) >= R
(V(j) - V(j-1)), g the drafter's confidence scaled by the running ratio of accepted drafts to their confidence, P the
product over the round's earlier drafts, R the running tokens per ms; or always 3 drafts with that gate) do no better:
28.05-28.63 against 28.80 at T = 0, 27.29-27.55 against 27.20 at T = 1, and the 3-draft variant loses 8-10% on the
reference prompt (`gate/policysim2-eb4dbeb.txt`). Against main's drafting with its own costs (5.85 ms draft steps,
k = 0 rounds 2 ms over a plain step) the replays expect +5.5% at T = 0 (27.26 -> 28.77) and +2.1% at T = 1.

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
| chat, greedy, main | 18.30 | 21.55 | +17.8% | 0.70 / 0.84 / 0.71 | 85/7/318/254 | 54.6-55.4 / 90.9-92.9 / 107.5-109.5 / 128.1-128.8 | 5.77 ms |
| chat, greedy, branch, unsure draft verified last | 18.34 | **22.27** | **+21.4%** | 0.74 / 0.86 / 0.76 | 85/70/283/232 | 54.6-55.0 / 90.0-91.6 / 107.3-109.3 / 126.9-128.2 | 5.05-5.10 |
| chat, greedy, branch, every chosen draft verified | 18.38 | 22.10 | +20.2% | 0.70 / 0.85 / 0.69 | 74/7/281/287 | | 5.01-5.04 |
| reference (64 tokens), main | 18.49 | 18.17 | -1.7% | 0.57 | 159/1/12/1 | 54.7 / 91.0 / 107.5 / 134.8 | |
| reference, branch | 18.53 | 18.15 | -2.0% | 0.50 | 154/4/12/2 | 54.2 / 89.4 / 106.4 / 129.9 | |
| 3K-token document, main | 16.15 | 16.23 | +0.5% | 0.62 | 61/1/53/3 | 62.9 / 103.0 / 123.6 / 150.5 | 6.55 |
| 3K-token document, branch | 16.14 | **17.13** | **+6.2%** | 0.69 | 51/17/41/4 | 62.4 / 100.1 / 122.6 / 150.1 | 5.21 |

(chat: water / csv / transformer; per-prompt rows in `chain1/summary.txt`.) The draft step on the chain went from 5.77 to
5.05 ms at chat lengths and from 6.55 to 5.21 ms at ~3.1K positions; a k = 0 round (a plain step with the shadow draft)
costs 0.2-0.4 ms more than a plain step (54.2 against 53.97 ms on the reference prompt; main 54.7 against 54.08), and
2-4-token rounds 0.2-1.5 ms less on the chat replies. Sampled, same texts for both drafting methods (the keyed sampler makes the plain output
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
included; `q4-turns.py` 57 tokens in 2.72 s, then 64 (78 of 99 prompt tokens reused) in 2.51 s (main's MTP in the
previous README's session 4: 2.83 and 2.53 s).

**Session 2** (`chain2/`, this branch on main `7da6f93`: everything above with unsure drafts dropped and the two-rate
policy; `tools/mtp2-chain2.sh`). The same binary with every draft counted sure (`no-gate`: main's pooled policy, every
chosen draft verified) and with greedy drafts as well (`main's drafting`) ran through a temporary switch
(`TRASHCAN_MTP_AB`, removed after the session); main `7da6f93` alongside:

| run | plain tok/s | MTP tok/s | gain | water / csv / transformer | rounds k = 0/1/2/3 (chat: water) | dropped drafts (matching the next token where it was reached) |
|---|---|---|---|---|---|---|
| chat, greedy, main | 18.43 | 21.68 | +17.6% | 20.31 / 24.52 / 20.21 | 46/4/141/59 | |
| chat, greedy, this branch, no-gate | 18.38 | 21.89 | +19.1% | 20.49 / 24.81 / 20.37 | 31/4/131/74 | |
| **chat, greedy, this branch** | 18.41 | **23.11** | **+25.5%** | 21.84 / 25.27 / 22.22 | 50/22/80/82 | 194 (32) |
| reference (64 tokens), main | 18.65 | 18.00 | -3.5% | | 158/1/12/2 | |
| **reference, this branch** | 18.50 | 18.21 | **-1.6%** | | 154/3/5/8 | 5 (0) |
| 3K-token document, main | 16.14 | 16.10 | -0.3% | | 73/1/46/5 | |
| **3K-token document, this branch** | 16.27 | **17.16** | **+5.5%** | | 66/12/29/10 | 35 (3) |

Sampled (top-k 64, top-p 0.95, seeds 11-13 at T = 1, 11-12 below; the keyed sampler gives both drafting methods the
same replies; accepted/draft counts the verified drafts, which for this branch are the ones the drafter was sure of):

| T | plain | main's drafting | **this branch** | gain | accepted/draft, main's drafting / this branch (water, csv, transformer) |
|---|---|---|---|---|---|
| 0.6 | 18.41-18.43 | 20.98 (+14.0%) | **22.13 (+20.0%)** | +5.5% | 0.61 / 0.72, 0.82 / 0.87, 0.68 / 0.78 |
| 0.8 | 18.27-18.41 | 21.35 (+16.0%) | **22.06 (+20.7%)** | +3.3% | 0.65 / 0.73, 0.80 / 0.90, 0.71 / 0.75 |
| 1.0 | 18.36-18.41 | 20.77 (+13.1%) | **22.01 (+19.5%)** | +6.0% | 0.61 / 0.78, 0.78 / 0.85, 0.67 / 0.75 |

Dropping unsure drafts with the two-rate policy added 5.6% to greedy chat replies (21.89 -> 23.11; simulated +5.8%),
and the whole branch 6.6% over main (21.68 -> 23.11): 13% of the drafts were dropped (14% at T = 1), and 83% of the
verified ones were accepted. A k = 0 round cost 54.19 ms against 54.05 ms per plain token (main: 54.60 against 53.62);
draft steps 4.95-5.21 ms against main's 5.84-7.31. Every run gave plain decoding's tokens. The server (`tools/q4serve-test.sh`):
every `scripts/test-server.py` check passes; up after 27 s (parallel setup); `q4-turns.py` 57 tokens in 2.64 s, then 64
in 2.50 s.

**Session 3** (`chain3/`, this branch rebased on main `2293550`: small-batch experts, connect retries, load time,
prefill; `tools/mtp2-chain3.sh`): validation, then one alternated pair on the chat replies (greedy, `--runs 1`):

| run | plain tok/s | MTP tok/s | gain | water / csv / transformer | accepted/draft | rounds k = 0/1/2/3 | ms per round k = 0/1/2/3 | draft step |
|---|---|---|---|---|---|---|---|---|
| chat, main `2293550` | 19.54 | 23.28 | +19.2% | 21.97 / 26.23 / 21.66 | 0.71 / 0.86 / 0.71 | 18/3/97/93 | 53.1-53.3 / 84.3-85.0 / 99.7-101.0 / 118.9-119.5 | 5.75-5.98 ms |
| **chat, this branch** | 19.48 | **24.72** | **+26.9%** | 23.62 / 26.92 / 23.62 | 0.80 / 0.88 / 0.81 | 33/21/28/122 | 52.0-52.6 / 83.4-85.5 / 100.2-102.3 / 117.7-118.1 | 5.06-5.17 ms |

The branch adds 6.2% to main's MTP (23.28 -> 24.72 tok/s; 62 drafts dropped, 9 of them matching the next token), with
the same tokens as plain decoding. Validation (`tools/valcmp.sh mtp2s3`, `chain3/validation.txt`): the branch's p100
dump is byte-identical to main's (586 of 586 files); on the 14-token reference prompt the top token is the CPU
reference's (9419, logits rel 2.8e-4), and p100's logits are as far from the CPU reference as main's (rel 3.5e-2).

**Sessions 4 and 5** (`chain4/`, `chain5/`, this branch rebased on main `eb4dbeb`: the 1-4-token verification kernels;
binary built from `16ad0dd9`, the engine as committed; `tools/mtp2-chain4.sh`, `tools/mtp2-chain5.sh`). Session 4:
p100 validation, then the pair; invite's GPU0 ran at 450 MHz through it (latched after overheating; every other GPU at
850 MHz), so session 5 repeated the pair after invite's reboot with every GPU at 850 MHz while working (its first
attempt stopped before decoding: the chain's Thunderbolt names were not back yet). Greedy chat replies, `--runs 1`:

| run | plain tok/s | MTP tok/s | gain | water / csv / transformer | accepted/draft | rounds k = 0/1/2/3 | ms per round k = 0/1/2/3 | draft step |
|---|---|---|---|---|---|---|---|---|
| session 5, main `eb4dbeb` | 20.24 | 27.29 | +34.9% | 24.97 / 30.21 / 26.69 | 0.71 / 0.90 / 0.76 | 3/40/192/1 | 49.7-51.4 / 66.7-67.1 / 81.5-82.6 / 107.2 | 5.75-5.85 ms |
| **session 5, this branch** | 20.32 | **28.49** | **+40.2%** | 27.64 / 29.97 / 27.86 | 0.84 / 0.85 / 0.84 | 24/25/75/85 | 50.1-50.2 / 66.7-67.4 / 82.3-83.9 / 101.1-102.0 | 5.02-5.07 ms |
| session 4 (invite GPU0 at 450 MHz), main | 19.27 | 26.03 | +35.1% | 24.20 / 28.71 / 25.18 | 0.72 / 0.91 / 0.76 | 3/34/185/9 | 52.9-53.9 / 70.3-70.8 / 86.9-87.3 / 106.7-111.9 | 5.76 ms |
| session 4, this branch | 19.30 | 27.41 | +42.0% | 26.14 / 29.44 / 26.65 | 0.83 / 0.91 / 0.85 | 28/25/96/66 | 52.1-52.7 / 70.3-70.9 / 86.6-87.1 / 106.4-113.1 | 5.00-5.06 ms |

With the cheaper batches the branch adds 4.4% to main's MTP (27.29 -> 28.49 tok/s; 5.3% in session 4), against 6.2%
on `2293550` and the replays' 5.5%: water +10.7%, transformer +4.4%, csv -0.8% (main's acceptance makes 2-draft rounds
pay there, 68 of 68; the branch's policy, with its cheaper draft steps, chose 3 drafts in 42 of 58 rounds). 55 drafts
were dropped (7 matching the next token). Every run gave plain decoding's tokens; the branch's p100 dump is
byte-identical to main `eb4dbeb`'s (586 of 586 files, `chain4/validation.txt`).

**Session 6** (`chain6/`, this branch rebased on main `9f54fb8`, engine `e9f34e6`: QSA kernels and prefill chunks queued
in GpuStage's submission slots, which the verification batches and the rollback also go through;
`tools/mtp2-chain6.sh`; every GPU at 850 MHz while working): p100 validation, then one pass:

| run | plain tok/s | MTP tok/s | gain | water / csv / transformer | accepted/draft | rounds k = 0/1/2/3 | ms per round k = 0/1/2/3 | draft step |
|---|---|---|---|---|---|---|---|---|
| main `e9f34e6` | 20.38 | 27.72 | +36.0% | 26.61 / 30.29 / 26.24 | 0.73 / 0.87 / 0.72 | 1/10/168/38 | 53.6 / 66.6-66.9 / 80.4-82.1 / 99.3-103.9 | 5.76-5.86 ms |
| **this branch** | 20.26 | **29.14** | **+43.8%** | 27.30 / 31.42 / 28.70 | 0.83 / 0.95 / 0.92 | 31/24/176/1 | 49.5-51.2 / 65.9-68.2 / 80.5-82.2 / 109.6 | 5.01-5.04 ms |

+5.1% over main's MTP (27.72 -> 29.14), the same tokens as plain decoding, and the p100 dump byte-identical to main
`e9f34e6`'s (598 of 598 files, `chain6/validation.txt`).

## Reproduction

```sh
# Mac (CPU reference): sampled runs, the drafter replayed along them, acceptance by temperature and method
sampling/study.sh                                   # accept --temperature T --depth 0; replay --teacher (64 candidates)
python3 sampling/sampling.py DIR [--top-k 20] [--draft-scale 0.9] [--csv steps.csv]
python3 gate/confsim.py DIR; python3 gate/confsim2.py DIR 1.0     # unsure drafts: fixed rules on the traces
python3 gate/policysim.py DIR --t 0 [--verify 40,52,60,70]          # the engine's policy variants on the traces
tools/policysim-eb4dbeb.sh DIR      # eb4dbeb's round costs: thresholds, cost-aware gates (gate/policysim2.py)
# Mac: two-process CPU chain (identity, greedy and sampled); the drafter's candidates against main's
tools/cpu2.sh 0 1; tools/same.sh
# incite (under a lease): draft-step timing, main and this branch alternating; the instrumented profile
tools/mtp2-db-ab2.sh; python3 tools/instrument2.py WORKTREE   # temporary timers, then draft-bench
# x86 unit tests (the SSSE3/AVX drafter kernels) built in the kbuild VM and run on incite
tools/xtest.sh; ./trashcan-mtp2-tests qwen4::draft qwen4::spec
# six Macs (sessions 1-4; DRY=1 prints the commands of sessions 3 and 4)
chain6.sh mtp2-chain1 50 OWNER -- tools/mtp2-chain1.sh; tools/fetch_session.sh chain1 NAMES...
chain6.sh mtp2-chain2 35 OWNER -- tools/mtp2-chain2.sh; chain6.sh mtp2-chain3 8 OWNER -- tools/mtp2-chain3.sh
chain6.sh mtp2-chain4 4 OWNER -- tools/mtp2-chain4.sh
python3 summary.py chain1/*.json; tools/valcmp.sh mtp2s4     # validation dumps against the CPU references and main's
```

## What is left

- Cheaper 2-4-token verification batches remain the largest lever (another agent's kernels): a round's verification is
  most of its time (k = 3 on main `eb4dbeb`: 104-108 ms of verification against 15 ms of drafting), and the policy
  picks up cheaper batches by itself (`gate/policysim.py` with 40 / 52 / 60 / 70 ms rounds: 39 tok/s on the chat
  replies).
- Drafting the next round while a batch is verified (the CPU is idle then) does not pay as such. Teacher-forced chains
  of 7 drafts along the greedy chat runs (`gate/rep7.sh`, `gate/pipesim.py`, `gate/pipesim-eb4dbeb.txt`): a round's
  chain, run on during its verification, predicts the bonus token and the next round's drafts, usable when every draft
  and the bonus match; but drafts 5-7 of a chain are accepted 0.65-0.88 against 0.75-0.95 for drafts 1-3 from the
  trunk's row, so with eb4dbeb's costs 3-draft rounds lose 2-6% and 2-draft rounds gain -1 to +5%. Using such drafts
  only when the drafter is sure of them, or rounds of 1-2 pre-drafted tokens, might still pay a few percent.
- The block's context: it reads only the residual rows of a prompt's last prefill chunk (`Speculator::prefill`), so a
  prompt whose length is just past a multiple of the chunk size leaves it 1-4 rows, and from a conversation's second
  turn on it no longer sees the earlier turns. In a second turn (`turns/`: water's prompt and greedy reply, then a
  follow-up question, 312 tokens, reply decoded on the CPU reference) the drafter's 3-draft rounds gave 2.46 tokens per
  round with 2 prompt tokens of context (what a 1-token last chunk leaves), 2.56-2.67 with 8 to 128 (no trend past a few
  rows). Merging a remainder of a few tokens into the previous prefill chunk (`Chain::extend`, prefill code) would avoid
  the short windows; the earlier turns did not matter for this reply (the 3K-token document lost 11% of its tokens per
  round to a 128-token window in the previous README).
- The draft step's remaining time below the bandwidth: the mixers (0.27 ms each for 4 MB) and the input projection
  (the four streams decode each weight block four times, 0.46 ms for 8 MB); at long contexts each of the 24 attention
  heads reads its KV head's 2,051 positions from L3 (100 MB per step at 3K positions, 0.73 ms), where the 12 heads
  sharing a KV head could read each row once (needs a different summation order than the scalar reference).
- Degenerate text (the reference prompt) still costs ~2%: the rounds that measure each batch size at the start and
  the drafts tried before the acceptance estimate falls.
- Standard speculative sampling would add ~1% more acceptance than the coupled drafts at T = 1 (CPU study) but makes
  seeded output depend on the round sizes, which follow measured costs; not done.
