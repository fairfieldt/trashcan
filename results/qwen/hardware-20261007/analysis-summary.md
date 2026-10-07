# Hardware comparison

Full-clock summaries require at least two in-window samples per host, 90% clock coverage and 99% of observed time at 850/1370 MHz (or configured thresholds) on both cards of both hosts. Shorter windows retain measured latency but cannot independently establish sustained clocks. Excluded repetitions are explicit in the manifest. Numerical tolerances are explicit per comparison.

## base3-pp2048-a → gdn3-pp2048-b

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp2048 | 0 | yes | 0.0 | -0.95% | -0.12% | yes/yes |

- pp: 1/1 equal-work, correct, full-clock pairs; median gain -0.95%.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -0.12%.

## base3-pp2048-a → sum3-pp2048-c

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp2048 | 0 | yes | 0.0 | -4.07% | +0.39% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain +0.39%.

## base3-pp2048-a → both3-pp2048-d

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp2048 | 0 | yes | 0.0 | -3.13% | +0.51% | yes/yes |

- pp: 1/1 equal-work, correct, full-clock pairs; median gain -3.13%.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain +0.51%.

## base3-pp2048-a → both3-pp2048-e

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp2048 | 0 | yes | 0.0 | -4.97% | -4.16% | yes/yes |

- pp: 1/1 equal-work, correct, full-clock pairs; median gain -4.97%.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -4.16%.

## base3-pp2048-a → sum3-pp2048-f

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp2048 | 0 | yes | 0.0 | -5.65% | -0.82% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -0.82%.

## base3-pp2048-a → gdn3-pp2048-g

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp2048 | 0 | yes | 0.0 | -5.04% | +0.11% | yes/yes |

- pp: 1/1 equal-work, correct, full-clock pairs; median gain -5.04%.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain +0.11%.

## base3-pp2048-a → base3-pp2048-h

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp2048 | 0 | yes | 0.0 | -6.42% | +0.36% | yes/yes |

- pp: 1/1 equal-work, correct, full-clock pairs; median gain -6.42%.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain +0.36%.

## gdn3-pp2048-b → sum3-pp2048-c

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp2048 | 0 | yes | 0.0 | -3.16% | +0.52% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain +0.52%.

## gdn3-pp2048-b → both3-pp2048-d

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp2048 | 0 | yes | 0.0 | -2.21% | +0.64% | yes/yes |

- pp: 1/1 equal-work, correct, full-clock pairs; median gain -2.21%.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain +0.64%.

## gdn3-pp2048-b → both3-pp2048-e

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp2048 | 0 | yes | 0.0 | -4.06% | -4.04% | yes/yes |

- pp: 1/1 equal-work, correct, full-clock pairs; median gain -4.06%.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -4.04%.

## gdn3-pp2048-b → sum3-pp2048-f

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp2048 | 0 | yes | 0.0 | -4.75% | -0.69% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -0.69%.

## gdn3-pp2048-b → gdn3-pp2048-g

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp2048 | 0 | yes | 0.0 | -4.14% | +0.23% | yes/yes |

- pp: 1/1 equal-work, correct, full-clock pairs; median gain -4.14%.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain +0.23%.

## gdn3-pp2048-b → base3-pp2048-h

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp2048 | 0 | yes | 0.0 | -5.53% | +0.48% | yes/yes |

- pp: 1/1 equal-work, correct, full-clock pairs; median gain -5.53%.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain +0.48%.

## sum3-pp2048-c → both3-pp2048-d

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp2048 | 0 | yes | 0.0 | +0.98% | +0.12% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain +0.12%.

## sum3-pp2048-c → both3-pp2048-e

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp2048 | 0 | yes | 0.0 | -0.94% | -4.53% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -4.53%.

## sum3-pp2048-c → sum3-pp2048-f

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp2048 | 0 | yes | 0.0 | -1.65% | -1.20% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -1.20%.

## sum3-pp2048-c → gdn3-pp2048-g

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp2048 | 0 | yes | 0.0 | -1.01% | -0.28% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -0.28%.

## sum3-pp2048-c → base3-pp2048-h

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp2048 | 0 | yes | 0.0 | -2.45% | -0.03% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -0.03%.

## both3-pp2048-d → both3-pp2048-e

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp2048 | 0 | yes | 0.0 | -1.90% | -4.65% | yes/yes |

- pp: 1/1 equal-work, correct, full-clock pairs; median gain -1.90%.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -4.65%.

## both3-pp2048-d → sum3-pp2048-f

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp2048 | 0 | yes | 0.0 | -2.60% | -1.32% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -1.32%.

## both3-pp2048-d → gdn3-pp2048-g

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp2048 | 0 | yes | 0.0 | -1.97% | -0.40% | yes/yes |

- pp: 1/1 equal-work, correct, full-clock pairs; median gain -1.97%.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -0.40%.

## both3-pp2048-d → base3-pp2048-h

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp2048 | 0 | yes | 0.0 | -3.40% | -0.15% | yes/yes |

- pp: 1/1 equal-work, correct, full-clock pairs; median gain -3.40%.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -0.15%.

## both3-pp2048-e → sum3-pp2048-f

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp2048 | 0 | yes | 0.0 | -0.72% | +3.48% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain +3.48%.

## both3-pp2048-e → gdn3-pp2048-g

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp2048 | 0 | yes | 0.0 | -0.07% | +4.45% | yes/yes |

- pp: 1/1 equal-work, correct, full-clock pairs; median gain -0.07%.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain +4.45%.

## both3-pp2048-e → base3-pp2048-h

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp2048 | 0 | yes | 0.0 | -1.53% | +4.71% | yes/yes |

- pp: 1/1 equal-work, correct, full-clock pairs; median gain -1.53%.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain +4.71%.

## sum3-pp2048-f → gdn3-pp2048-g

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp2048 | 0 | yes | 0.0 | +0.65% | +0.93% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain +0.93%.

## sum3-pp2048-f → base3-pp2048-h

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp2048 | 0 | yes | 0.0 | -0.81% | +1.18% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain +1.18%.

## gdn3-pp2048-g → base3-pp2048-h

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp2048 | 0 | yes | 0.0 | -1.45% | +0.25% | yes/yes |

- pp: 1/1 equal-work, correct, full-clock pairs; median gain -1.45%.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain +0.25%.

## short-budget-plain → short-budget-old-mtp

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| count-1 | 0 | yes | 0.0 | +6.72% | -34.68% | no/no |
| count-2 | 0 | yes | 0.0 | +5.84% | -37.55% | yes/no |
| count-3 | 0 | yes | 0.0 | -1.14% | +16.80% | yes/no |
| count-4 | 0 | yes | 0.0 | -1.33% | +70.17% | yes/no |
| count-5 | 0 | yes | 0.0 | -4.52% | +14.33% | yes/no |
| count-6 | 0 | yes | 0.0 | -1.38% | +31.25% | yes/no |
| count-7 | 0 | yes | 0.0 | -3.37% | +71.96% | yes/no |
| count-8 | 0 | yes | 0.0 | -5.28% | +28.34% | yes/no |
| stop-1 | 0 | yes | 0.0 | +6.77% | -12.01% | yes/no |
| stop-2 | 0 | yes | 0.0 | -0.87% | -44.73% | yes/no |
| stop-3 | 0 | yes | 0.0 | -0.85% | -44.44% | yes/no |
| stop-4 | 0 | yes | 0.0 | -2.62% | -51.63% | yes/no |
| stop-5 | 0 | yes | 0.0 | -2.76% | -43.92% | yes/no |
| stop-6 | 0 | yes | 0.0 | -4.77% | -47.75% | yes/no |
| stop-7 | 0 | yes | 0.0 | -8.17% | -44.53% | yes/no |
| stop-8 | 0 | yes | 0.0 | -0.79% | -43.92% | yes/no |

- pp: 15/16 equal-work, correct, full-clock pairs; median gain -1.38%.
- tg: 0/16 equal-work, correct, full-clock pairs; median gain unavailable.

## short-budget-plain → short-budget-new-mtp

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| count-1 | 0 | yes | 0.0 | -0.15% | -1.42% | no/no |
| count-2 | 0 | yes | 0.0 | +2.00% | -16.91% | yes/no |
| count-3 | 0 | yes | 0.0 | -5.92% | +40.13% | yes/no |
| count-4 | 0 | yes | 0.0 | -1.12% | +60.28% | yes/no |
| count-5 | 0 | yes | 0.0 | -1.02% | +44.58% | yes/no |
| count-6 | 0 | yes | 0.0 | +0.48% | +62.55% | yes/no |
| count-7 | 0 | yes | 0.0 | -0.90% | +72.20% | yes/no |
| count-8 | 0 | yes | 0.0 | -1.45% | +55.70% | yes/no |
| stop-1 | 0 | yes | 0.0 | +7.79% | -10.08% | yes/no |
| stop-2 | 0 | yes | 0.0 | -6.02% | -5.87% | yes/no |
| stop-3 | 0 | yes | 0.0 | -0.86% | -28.00% | yes/no |
| stop-4 | 0 | yes | 0.0 | -0.75% | -44.52% | yes/no |
| stop-5 | 0 | yes | 0.0 | -0.42% | -44.36% | yes/no |
| stop-6 | 0 | yes | 0.0 | -1.45% | -50.03% | yes/no |
| stop-7 | 0 | yes | 0.0 | -0.88% | -44.49% | yes/no |
| stop-8 | 0 | yes | 0.0 | -0.81% | -44.20% | yes/no |

- pp: 15/16 equal-work, correct, full-clock pairs; median gain -0.88%.
- tg: 0/16 equal-work, correct, full-clock pairs; median gain unavailable.

## short-budget-plain → short-budget-adaptive

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| count-1 | 0 | yes | 0.0 | +7.17% | +4.90% | no/no |
| count-2 | 0 | yes | 0.0 | +5.69% | +7.95% | yes/no |
| count-3 | 0 | yes | 0.0 | -1.10% | +36.10% | yes/no |
| count-4 | 0 | yes | 0.0 | -1.26% | +27.46% | yes/no |
| count-5 | 0 | yes | 0.0 | -1.19% | +7.11% | yes/no |
| count-6 | 0 | yes | 0.0 | +0.24% | +14.73% | yes/no |
| count-7 | 0 | yes | 0.0 | -1.24% | +10.10% | yes/no |
| count-8 | 0 | yes | 0.0 | -1.34% | +18.69% | yes/no |
| stop-1 | 0 | yes | 0.0 | +7.23% | -5.42% | yes/no |
| stop-2 | 0 | yes | 0.0 | -0.99% | -15.30% | yes/no |
| stop-3 | 0 | yes | 0.0 | -1.25% | -29.64% | yes/no |
| stop-4 | 0 | yes | 0.0 | -0.64% | -2.22% | yes/no |
| stop-5 | 0 | yes | 0.0 | -0.59% | -1.36% | yes/no |
| stop-6 | 0 | yes | 0.0 | -1.04% | -15.66% | yes/no |
| stop-7 | 0 | yes | 0.0 | -1.29% | -1.94% | yes/no |
| stop-8 | 0 | yes | 0.0 | -0.88% | -1.45% | yes/no |

- pp: 15/16 equal-work, correct, full-clock pairs; median gain -1.04%.
- tg: 0/16 equal-work, correct, full-clock pairs; median gain unavailable.

## short-budget-old-mtp → short-budget-new-mtp

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| count-1 | 0 | yes | 0.0 | -6.44% | +50.92% | no/no |
| count-2 | 0 | yes | 0.0 | -3.63% | +33.05% | yes/no |
| count-3 | 0 | yes | 0.0 | -4.84% | +19.97% | yes/no |
| count-4 | 0 | yes | 0.0 | +0.21% | -5.81% | yes/no |
| count-5 | 0 | yes | 0.0 | +3.67% | +26.45% | yes/no |
| count-6 | 0 | yes | 0.0 | +1.89% | +23.84% | yes/no |
| count-7 | 0 | yes | 0.0 | +2.55% | +0.14% | yes/no |
| count-8 | 0 | yes | 0.0 | +4.05% | +21.32% | yes/no |
| stop-1 | 0 | yes | 0.0 | +0.95% | +2.20% | yes/no |
| stop-2 | 0 | yes | 0.0 | -5.20% | +70.30% | yes/no |
| stop-3 | 0 | yes | 0.0 | -0.01% | +29.58% | yes/no |
| stop-4 | 0 | yes | 0.0 | +1.92% | +14.68% | yes/no |
| stop-5 | 0 | yes | 0.0 | +2.41% | -0.77% | yes/no |
| stop-6 | 0 | yes | 0.0 | +3.48% | -4.36% | yes/no |
| stop-7 | 0 | yes | 0.0 | +7.94% | +0.07% | yes/no |
| stop-8 | 0 | yes | 0.0 | -0.02% | -0.50% | yes/no |

- pp: 15/16 equal-work, correct, full-clock pairs; median gain +1.89%.
- tg: 0/16 equal-work, correct, full-clock pairs; median gain unavailable.

## short-budget-old-mtp → short-budget-adaptive

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| count-1 | 0 | yes | 0.0 | +0.42% | +60.59% | no/no |
| count-2 | 0 | yes | 0.0 | -0.14% | +72.87% | yes/no |
| count-3 | 0 | yes | 0.0 | +0.03% | +16.53% | yes/no |
| count-4 | 0 | yes | 0.0 | +0.07% | -25.10% | yes/no |
| count-5 | 0 | yes | 0.0 | +3.49% | -6.32% | yes/no |
| count-6 | 0 | yes | 0.0 | +1.64% | -12.59% | yes/no |
| count-7 | 0 | yes | 0.0 | +2.21% | -35.97% | yes/no |
| count-8 | 0 | yes | 0.0 | +4.17% | -7.51% | yes/no |
| stop-1 | 0 | yes | 0.0 | +0.43% | +7.49% | yes/no |
| stop-2 | 0 | yes | 0.0 | -0.12% | +53.25% | yes/no |
| stop-3 | 0 | yes | 0.0 | -0.40% | +26.63% | yes/no |
| stop-4 | 0 | yes | 0.0 | +2.04% | +102.14% | yes/no |
| stop-5 | 0 | yes | 0.0 | +2.23% | +75.91% | yes/no |
| stop-6 | 0 | yes | 0.0 | +3.91% | +61.43% | yes/no |
| stop-7 | 0 | yes | 0.0 | +7.49% | +76.80% | yes/no |
| stop-8 | 0 | yes | 0.0 | -0.09% | +75.73% | yes/no |

- pp: 15/16 equal-work, correct, full-clock pairs; median gain +1.64%.
- tg: 0/16 equal-work, correct, full-clock pairs; median gain unavailable.

## short-budget-new-mtp → short-budget-adaptive

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| count-1 | 0 | yes | 0.0 | +7.33% | +6.41% | no/no |
| count-2 | 0 | yes | 0.0 | +3.62% | +29.93% | yes/no |
| count-3 | 0 | yes | 0.0 | +5.12% | -2.87% | yes/no |
| count-4 | 0 | yes | 0.0 | -0.14% | -20.48% | yes/no |
| count-5 | 0 | yes | 0.0 | -0.17% | -25.92% | yes/no |
| count-6 | 0 | yes | 0.0 | -0.24% | -29.42% | yes/no |
| count-7 | 0 | yes | 0.0 | -0.34% | -36.06% | yes/no |
| count-8 | 0 | yes | 0.0 | +0.11% | -23.77% | yes/no |
| stop-1 | 0 | yes | 0.0 | -0.52% | +5.18% | yes/no |
| stop-2 | 0 | yes | 0.0 | +5.35% | -10.01% | yes/no |
| stop-3 | 0 | yes | 0.0 | -0.39% | -2.27% | yes/no |
| stop-4 | 0 | yes | 0.0 | +0.12% | +76.25% | yes/no |
| stop-5 | 0 | yes | 0.0 | -0.17% | +77.27% | yes/no |
| stop-6 | 0 | yes | 0.0 | +0.42% | +68.79% | yes/no |
| stop-7 | 0 | yes | 0.0 | -0.42% | +76.67% | yes/no |
| stop-8 | 0 | yes | 0.0 | -0.07% | +76.62% | yes/no |

- pp: 15/16 equal-work, correct, full-clock pairs; median gain -0.14%.
- tg: 0/16 equal-work, correct, full-clock pairs; median gain unavailable.

## decode3-old-plain-a → decode3-new-plain-b

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp512 | 0 | yes | 0.0 | -0.10% | +1.28% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain +1.28%.

## decode3-old-plain-a → decode3-new-mtp-c

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp512 | 0 | yes | 0.0 | -1.75% | +33.11% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain +33.11%.

## decode3-new-plain-b → decode3-new-mtp-c

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp512 | 0 | yes | 0.0 | -1.65% | +31.43% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain +31.43%.

## maxfan-base3-pp2048-a → maxfan-gdn3-pp2048-b

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp2048 | 0 | yes | 0.0 | +0.96% | -0.23% | yes/yes |

- pp: 1/1 equal-work, correct, full-clock pairs; median gain +0.96%.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -0.23%.

## maxfan-base3-pp2048-a → maxfan-sum3-pp2048-c

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp2048 | 0 | yes | 0.0 | -1.55% | +0.21% | yes/yes |

- pp: 1/1 equal-work, correct, full-clock pairs; median gain -1.55%.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain +0.21%.

## maxfan-base3-pp2048-a → maxfan-both3-pp2048-d

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp2048 | 0 | yes | 0.0 | -0.47% | -0.01% | yes/yes |

- pp: 1/1 equal-work, correct, full-clock pairs; median gain -0.47%.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -0.01%.

## maxfan-base3-pp2048-a → maxfan-both3-pp2048-e

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp2048 | 0 | yes | 0.0 | -1.23% | +0.11% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain +0.11%.

## maxfan-base3-pp2048-a → maxfan-sum3-pp2048-f

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp2048 | 0 | yes | 0.0 | -2.84% | +0.05% | yes/yes |

- pp: 1/1 equal-work, correct, full-clock pairs; median gain -2.84%.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain +0.05%.

## maxfan-base3-pp2048-a → maxfan-gdn3-pp2048-g

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp2048 | 0 | yes | 0.0 | -1.42% | -0.97% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -0.97%.

## maxfan-base3-pp2048-a → maxfan-base3-pp2048-h

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp2048 | 0 | yes | 0.0 | -2.42% | -0.39% | yes/yes |

- pp: 1/1 equal-work, correct, full-clock pairs; median gain -2.42%.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -0.39%.

## maxfan-gdn3-pp2048-b → maxfan-sum3-pp2048-c

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp2048 | 0 | yes | 0.0 | -2.49% | +0.44% | yes/yes |

- pp: 1/1 equal-work, correct, full-clock pairs; median gain -2.49%.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain +0.44%.

## maxfan-gdn3-pp2048-b → maxfan-both3-pp2048-d

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp2048 | 0 | yes | 0.0 | -1.42% | +0.22% | yes/yes |

- pp: 1/1 equal-work, correct, full-clock pairs; median gain -1.42%.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain +0.22%.

## maxfan-gdn3-pp2048-b → maxfan-both3-pp2048-e

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp2048 | 0 | yes | 0.0 | -2.17% | +0.34% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain +0.34%.

## maxfan-gdn3-pp2048-b → maxfan-sum3-pp2048-f

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp2048 | 0 | yes | 0.0 | -3.77% | +0.28% | yes/yes |

- pp: 1/1 equal-work, correct, full-clock pairs; median gain -3.77%.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain +0.28%.

## maxfan-gdn3-pp2048-b → maxfan-gdn3-pp2048-g

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp2048 | 0 | yes | 0.0 | -2.36% | -0.74% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -0.74%.

## maxfan-gdn3-pp2048-b → maxfan-base3-pp2048-h

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp2048 | 0 | yes | 0.0 | -3.35% | -0.16% | yes/yes |

- pp: 1/1 equal-work, correct, full-clock pairs; median gain -3.35%.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -0.16%.

## maxfan-sum3-pp2048-c → maxfan-both3-pp2048-d

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp2048 | 0 | yes | 0.0 | +1.09% | -0.22% | yes/yes |

- pp: 1/1 equal-work, correct, full-clock pairs; median gain +1.09%.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -0.22%.

## maxfan-sum3-pp2048-c → maxfan-both3-pp2048-e

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp2048 | 0 | yes | 0.0 | +0.33% | -0.10% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -0.10%.

## maxfan-sum3-pp2048-c → maxfan-sum3-pp2048-f

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp2048 | 0 | yes | 0.0 | -1.31% | -0.16% | yes/yes |

- pp: 1/1 equal-work, correct, full-clock pairs; median gain -1.31%.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -0.16%.

## maxfan-sum3-pp2048-c → maxfan-gdn3-pp2048-g

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp2048 | 0 | yes | 0.0 | +0.13% | -1.17% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -1.17%.

## maxfan-sum3-pp2048-c → maxfan-base3-pp2048-h

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp2048 | 0 | yes | 0.0 | -0.89% | -0.59% | yes/yes |

- pp: 1/1 equal-work, correct, full-clock pairs; median gain -0.89%.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -0.59%.

## maxfan-both3-pp2048-d → maxfan-both3-pp2048-e

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp2048 | 0 | yes | 0.0 | -0.76% | +0.12% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain +0.12%.

## maxfan-both3-pp2048-d → maxfan-sum3-pp2048-f

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp2048 | 0 | yes | 0.0 | -2.38% | +0.06% | yes/yes |

- pp: 1/1 equal-work, correct, full-clock pairs; median gain -2.38%.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain +0.06%.

## maxfan-both3-pp2048-d → maxfan-gdn3-pp2048-g

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp2048 | 0 | yes | 0.0 | -0.95% | -0.96% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -0.96%.

## maxfan-both3-pp2048-d → maxfan-base3-pp2048-h

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp2048 | 0 | yes | 0.0 | -1.96% | -0.38% | yes/yes |

- pp: 1/1 equal-work, correct, full-clock pairs; median gain -1.96%.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -0.38%.

## maxfan-both3-pp2048-e → maxfan-sum3-pp2048-f

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp2048 | 0 | yes | 0.0 | -1.63% | -0.06% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -0.06%.

## maxfan-both3-pp2048-e → maxfan-gdn3-pp2048-g

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp2048 | 0 | yes | 0.0 | -0.20% | -1.07% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -1.07%.

## maxfan-both3-pp2048-e → maxfan-base3-pp2048-h

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp2048 | 0 | yes | 0.0 | -1.21% | -0.50% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -0.50%.

## maxfan-sum3-pp2048-f → maxfan-gdn3-pp2048-g

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp2048 | 0 | yes | 0.0 | +1.46% | -1.02% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -1.02%.

## maxfan-sum3-pp2048-f → maxfan-base3-pp2048-h

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp2048 | 0 | yes | 0.0 | +0.43% | -0.44% | yes/yes |

- pp: 1/1 equal-work, correct, full-clock pairs; median gain +0.43%.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -0.44%.

## maxfan-gdn3-pp2048-g → maxfan-base3-pp2048-h

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp2048 | 0 | yes | 0.0 | -1.01% | +0.59% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain +0.59%.

## maxfan-decode3-old-plain-a → maxfan-decode3-new-plain-b

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp512 | 0 | yes | 0.0 | +0.80% | +0.42% | yes/yes |

- pp: 1/1 equal-work, correct, full-clock pairs; median gain +0.80%.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain +0.42%.

## maxfan-decode3-old-plain-a → maxfan-decode3-new-mtp-c

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp512 | 0 | yes | 0.0 | -1.44% | +32.11% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain +32.11%.

## maxfan-decode3-old-plain-a → maxfan-decode3-old-mtp-d

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp512 | 0 | yes | 0.0 | -0.83% | +31.87% | yes/yes |

- pp: 1/1 equal-work, correct, full-clock pairs; median gain -0.83%.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain +31.87%.

## maxfan-decode3-old-plain-a → maxfan-decode3-old-mtp-e

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp512 | 0 | yes | 0.0 | -0.78% | +31.75% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain +31.75%.

## maxfan-decode3-old-plain-a → maxfan-decode3-new-mtp-f

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp512 | 0 | yes | 0.0 | -0.65% | +32.10% | yes/yes |

- pp: 1/1 equal-work, correct, full-clock pairs; median gain -0.65%.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain +32.10%.

## maxfan-decode3-old-plain-a → maxfan-decode3-new-plain-g

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp512 | 0 | yes | 0.0 | +0.79% | -0.53% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -0.53%.

## maxfan-decode3-old-plain-a → maxfan-decode3-old-plain-h

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp512 | 0 | yes | 0.0 | +0.46% | +0.20% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain +0.20%.

## maxfan-decode3-new-plain-b → maxfan-decode3-new-mtp-c

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp512 | 0 | yes | 0.0 | -2.23% | +31.55% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain +31.55%.

## maxfan-decode3-new-plain-b → maxfan-decode3-old-mtp-d

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp512 | 0 | yes | 0.0 | -1.62% | +31.32% | yes/yes |

- pp: 1/1 equal-work, correct, full-clock pairs; median gain -1.62%.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain +31.32%.

## maxfan-decode3-new-plain-b → maxfan-decode3-old-mtp-e

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp512 | 0 | yes | 0.0 | -1.57% | +31.20% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain +31.20%.

## maxfan-decode3-new-plain-b → maxfan-decode3-new-mtp-f

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp512 | 0 | yes | 0.0 | -1.44% | +31.54% | yes/yes |

- pp: 1/1 equal-work, correct, full-clock pairs; median gain -1.44%.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain +31.54%.

## maxfan-decode3-new-plain-b → maxfan-decode3-new-plain-g

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp512 | 0 | yes | 0.0 | -0.01% | -0.95% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -0.95%.

## maxfan-decode3-new-plain-b → maxfan-decode3-old-plain-h

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp512 | 0 | yes | 0.0 | -0.34% | -0.22% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -0.22%.

## maxfan-decode3-new-mtp-c → maxfan-decode3-old-mtp-d

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp512 | 0 | yes | 0.0 | +0.62% | -0.18% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -0.18%.

## maxfan-decode3-new-mtp-c → maxfan-decode3-old-mtp-e

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp512 | 0 | yes | 0.0 | +0.67% | -0.27% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -0.27%.

## maxfan-decode3-new-mtp-c → maxfan-decode3-new-mtp-f

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp512 | 0 | yes | 0.0 | +0.80% | -0.01% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -0.01%.

## maxfan-decode3-new-mtp-c → maxfan-decode3-new-plain-g

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp512 | 0 | yes | 0.0 | +2.26% | -24.71% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -24.71%.

## maxfan-decode3-new-mtp-c → maxfan-decode3-old-plain-h

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp512 | 0 | yes | 0.0 | +1.93% | -24.15% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -24.15%.

## maxfan-decode3-old-mtp-d → maxfan-decode3-old-mtp-e

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp512 | 0 | yes | 0.0 | +0.05% | -0.09% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -0.09%.

## maxfan-decode3-old-mtp-d → maxfan-decode3-new-mtp-f

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp512 | 0 | yes | 0.0 | +0.18% | +0.17% | yes/yes |

- pp: 1/1 equal-work, correct, full-clock pairs; median gain +0.18%.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain +0.17%.

## maxfan-decode3-old-mtp-d → maxfan-decode3-new-plain-g

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp512 | 0 | yes | 0.0 | +1.63% | -24.57% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -24.57%.

## maxfan-decode3-old-mtp-d → maxfan-decode3-old-plain-h

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp512 | 0 | yes | 0.0 | +1.30% | -24.02% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -24.02%.

## maxfan-decode3-old-mtp-e → maxfan-decode3-new-mtp-f

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp512 | 0 | yes | 0.0 | +0.13% | +0.26% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain +0.26%.

## maxfan-decode3-old-mtp-e → maxfan-decode3-new-plain-g

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp512 | 0 | yes | 0.0 | +1.58% | -24.51% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -24.51%.

## maxfan-decode3-old-mtp-e → maxfan-decode3-old-plain-h

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp512 | 0 | yes | 0.0 | +1.25% | -23.95% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -23.95%.

## maxfan-decode3-new-mtp-f → maxfan-decode3-new-plain-g

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp512 | 0 | yes | 0.0 | +1.45% | -24.70% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -24.70%.

## maxfan-decode3-new-mtp-f → maxfan-decode3-old-plain-h

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp512 | 0 | yes | 0.0 | +1.12% | -24.15% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -24.15%.

## maxfan-decode3-new-plain-g → maxfan-decode3-old-plain-h

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp512 | 0 | yes | 0.0 | -0.33% | +0.74% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain +0.74%.

## maxfan-code-plain-1 → maxfan-code-fixed-2

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| code | 0 | yes | 0.0 | +0.64% | +62.86% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain +62.86%.

## maxfan-code-plain-1 → maxfan-code-adaptive-3

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| code | 0 | yes | 0.0 | -1.02% | +57.08% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain +57.08%.

## maxfan-code-plain-1 → maxfan-code-adaptive-4

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| code | 0 | yes | 0.0 | -7.46% | +51.09% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain +51.09%.

## maxfan-code-plain-1 → maxfan-code-fixed-5

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| code | 0 | yes | 0.0 | -3.25% | +49.69% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain +49.69%.

## maxfan-code-plain-1 → maxfan-code-plain-6

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| code | 0 | yes | 0.0 | -3.09% | +0.72% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain +0.72%.

## maxfan-code-fixed-2 → maxfan-code-adaptive-3

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| code | 0 | yes | 0.0 | -1.65% | -3.55% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -3.55%.

## maxfan-code-fixed-2 → maxfan-code-adaptive-4

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| code | 0 | yes | 0.0 | -8.04% | -7.23% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -7.23%.

## maxfan-code-fixed-2 → maxfan-code-fixed-5

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| code | 0 | yes | 0.0 | -3.87% | -8.09% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -8.09%.

## maxfan-code-fixed-2 → maxfan-code-plain-6

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| code | 0 | yes | 0.0 | -3.70% | -38.16% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -38.16%.

## maxfan-code-adaptive-3 → maxfan-code-adaptive-4

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| code | 0 | yes | 0.0 | -6.50% | -3.82% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -3.82%.

## maxfan-code-adaptive-3 → maxfan-code-fixed-5

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| code | 0 | yes | 0.0 | -2.25% | -4.71% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -4.71%.

## maxfan-code-adaptive-3 → maxfan-code-plain-6

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| code | 0 | yes | 0.0 | -2.08% | -35.88% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -35.88%.

## maxfan-code-adaptive-4 → maxfan-code-fixed-5

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| code | 0 | yes | 0.0 | +4.54% | -0.93% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -0.93%.

## maxfan-code-adaptive-4 → maxfan-code-plain-6

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| code | 0 | yes | 0.0 | +4.72% | -33.34% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -33.34%.

## maxfan-code-fixed-5 → maxfan-code-plain-6

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| code | 0 | yes | 0.0 | +0.17% | -32.71% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -32.71%.

## maxfan-chinese-plain-1 → maxfan-chinese-fixed-2

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| chinese | 0 | yes | 0.0 | +7.33% | +6.49% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain +6.49%.

## maxfan-chinese-plain-1 → maxfan-chinese-adaptive-3

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| chinese | 0 | yes | 0.0 | -4.41% | +10.42% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain +10.42%.

## maxfan-chinese-plain-1 → maxfan-chinese-adaptive-4

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| chinese | 0 | yes | 0.0 | +3.87% | +11.19% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain +11.19%.

## maxfan-chinese-plain-1 → maxfan-chinese-fixed-5

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| chinese | 0 | yes | 0.0 | -0.57% | +1.86% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain +1.86%.

## maxfan-chinese-plain-1 → maxfan-chinese-plain-6

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| chinese | 0 | yes | 0.0 | +7.94% | +6.66% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain +6.66%.

## maxfan-chinese-fixed-2 → maxfan-chinese-adaptive-3

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| chinese | 0 | yes | 0.0 | -10.94% | +3.69% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain +3.69%.

## maxfan-chinese-fixed-2 → maxfan-chinese-adaptive-4

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| chinese | 0 | yes | 0.0 | -3.22% | +4.41% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain +4.41%.

## maxfan-chinese-fixed-2 → maxfan-chinese-fixed-5

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| chinese | 0 | yes | 0.0 | -7.36% | -4.35% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -4.35%.

## maxfan-chinese-fixed-2 → maxfan-chinese-plain-6

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| chinese | 0 | yes | 0.0 | +0.57% | +0.16% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain +0.16%.

## maxfan-chinese-adaptive-3 → maxfan-chinese-adaptive-4

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| chinese | 0 | yes | 0.0 | +8.67% | +0.70% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain +0.70%.

## maxfan-chinese-adaptive-3 → maxfan-chinese-fixed-5

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| chinese | 0 | yes | 0.0 | +4.02% | -7.76% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -7.76%.

## maxfan-chinese-adaptive-3 → maxfan-chinese-plain-6

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| chinese | 0 | yes | 0.0 | +12.93% | -3.41% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -3.41%.

## maxfan-chinese-adaptive-4 → maxfan-chinese-fixed-5

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| chinese | 0 | yes | 0.0 | -4.28% | -8.39% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -8.39%.

## maxfan-chinese-adaptive-4 → maxfan-chinese-plain-6

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| chinese | 0 | yes | 0.0 | +3.92% | -4.08% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain -4.08%.

## maxfan-chinese-fixed-5 → maxfan-chinese-plain-6

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| chinese | 0 | yes | 0.0 | +8.56% | +4.71% | no/yes |

- pp: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
- tg: 1/1 equal-work, correct, full-clock pairs; median gain +4.71%.

## maxfan-profile3-gdn-base-pp2048 → maxfan-profile3-gdn-prepared-pp2048

| Case | Rep | Greedy IDs | Top max error | Prefill gain | Decode gain | Full clocks pp/tg |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pp2048 | 0 | yes | 0.0 | +0.95% | +60.82% | yes/no |

- pp: 1/1 equal-work, correct, full-clock pairs; median gain +0.95%.
- tg: 0/1 equal-work, correct, full-clock pairs; median gain unavailable.
