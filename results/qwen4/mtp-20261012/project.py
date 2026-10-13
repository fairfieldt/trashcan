#!/usr/bin/env python3
"""Projected six-Mac decode rate with MTP speculative decoding.

A round drafts k tokens on the driver's CPU (k draft steps), verifies the pending token and the drafts as one
(k + 1)-token batch on the chain and keeps the matching prefix:
tok/s = tokens_per_round(k) / (k * draft_ms + verify_ms(k + 1)).

usage: project.py
"""
# Tokens per round, the final drafter (rows below 64K, 4-bit dense matrices; `qwen4 mtp replay --rounds` without
# rewriting accepted rows: 1.84 / 2.50 / 2.95; with the rewrite 1.84 / 2.53 / 3.05), pooled over the four prompts.
ROUNDS = {1: 1.84, 2: 2.50, 3: 2.95}
SCENARIOS = {
    # Measured on the chain (mtp spec, session 2): 67.3 ms per plain token, 112.6 ms for 3 tokens, 133.3 ms for 4;
    # 2 tokens from stage-bench's 1.38x on the GPU share.
    'session 2': {1: 67.3, 2: 90.0, 3: 112.6, 4: 133.3},
    # Session 4, main with the decode kernels: 53.9 ms per plain token; verification with rollback 91 / 108 / 128 ms.
    'session 4': {1: 53.9, 2: 91.0, 3: 108.0, 4: 128.0},
    # The target the coordinator set: 35 ms per token, 3 tokens at 1.4x and 4 at ~1.65x.
    '35 ms decode, 3 tokens 1.4x': {1: 35.0, 2: 42.0, 3: 49.0, 4: 58.0},
}
DRAFT_MS = [10.0, 7.0, 6.0, 4.0]

for name, v in SCENARIOS.items():
    print(f"\n{name}: plain {1000 / v[1]:.1f} tok/s")
    print('draft ms\t' + '\t'.join(f'k={k}' for k in ROUNDS))
    for dm in DRAFT_MS:
        print(f'{dm}\t\t' + '\t'.join(f'{1000 * ROUNDS[k] / (k * dm + v[k + 1]):.1f}' for k in ROUNDS))

# The engine's choice of drafts per round (`spec::Policy`): the k with the most expected tokens per ms, a round of k
# drafts yielding 1 + a + ... + a^k tokens at per-draft acceptance a (k = 0: a plain step).
print('\nbest k by per-draft acceptance (6 ms draft steps)')
print('a\t' + '\t\t'.join(SCENARIOS))
for a in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    cells = []
    for v in SCENARIOS.values():
        rate = {k: sum(a ** i for i in range(k + 1)) / (k * 6.0 + v[k + 1]) for k in range(4)}
        k = max(rate, key=rate.get)
        cells.append(f'k={k} {1000 * rate[k]:.1f} tok/s')
    print(f'{a}\t' + '\t\t'.join(cells))
