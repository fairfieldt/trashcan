# Batch-default and graph-cache audit

Run `20261003T112418Z-380364` completed all four configurations: 72 measured requests, 338 harness checks, 16 exact repeated-output pairs, and both existing protocol suites passed. All 108 cross-configuration request/output/token-count comparisons are exact and eligible; no output-drift exclusions apply to this run. All four scoped cleanup reports contain an empty `remaining` list.

All runs use binary SHA-256 `325b087f565ea4d2b0394892735ab06f9b9e1ffe3b8a6f190bbd2f9e81ea719e`, matching checkpoint `cc3f50f` and all 41 declared source hashes in the shared-KV build manifest. See the [complete invocation](batch-defaults/20261003T112418Z-380364-summary.json) and [build identity](shared-kv/build.json) for exact source maps and commands. This checkpoint follows core checkpoint `63290a5`; the binary hash identifies this test. The 512 and 512-MTP commands omit `--batch`, exercising the new default. All configurations use context 8192.

## Correctness and cleanup

| Configuration | Measured requests | Harness checks passed | Exact repeat pairs | Existing protocol suite | Cleanup |
|---|---:|---:|---:|---|---|
| 32 | 18/18 | 84/84 | 4/4 | not scheduled | no tracked processes remain |
| 128 | 18/18 | 84/84 | 4/4 | not scheduled | no tracked processes remain |
| 512 | 18/18 | 85/85 | 4/4 | passed, exit 0 | no tracked processes remain |
| 512-mtp | 18/18 | 85/85 | 4/4 | passed, exit 0 | no tracked processes remain |

The 512 protocol logs confirm discovery, JSON/SSE parity, persistent HTTP, greedy/stochastic graph switching, seeded repeats, token limits/reasoning separation, invalid input/overflow handling, function calls, tool continuation, namespaces and custom tools. The wrappers exit with `-15` after the driver deliberately sends SIGTERM; this is expected cleanup, not a test failure.

Stochastic ordinary-versus-MTP runs can consume random draws differently, so a general claim of same-seed cross-mode text equality would be too strong. This list fixture produced identical outputs across all four configurations, including its stochastic requests. There are therefore **zero** stochastic cross-mode exclusions here. The summary script excludes latency comparisons whenever request, normalized output or token count differs.

## Request latency and retained prefixes

All values below are measured HTTP wall seconds, rounded to milliseconds; they include prefill and protocol work. The medium input has 512 tokens and one output token. The long input has 1763 tokens and the default 256-token output budget. The history continuation has 2041 input tokens, returns two tokens (`Red`), and is identical in every configuration.

| Configuration | Medium s | Long first s | Long repeat s | Repeat cached / 1763 | History s | History cached / 2041 |
|---|---:|---:|---:|---:|---:|---:|
| 32 | 7.865 | 39.766 | 46.462 | 0 | 41.198 | 0 |
| 128 | 3.954 | 26.302 | 29.441 | 0 | 20.442 | 0 |
| 512 | 3.194 | 22.639 | 11.989 | 1763 | 2.783 | 1759 |
| 512-mtp | 3.156 | 17.636 | 7.104 | 1762 | 2.846 | 1759 |

Every first long request reuses 467 tokens from the preceding medium prompt. After 256 output tokens, batch 32/128 correctly falls back to zero cached tokens, explaining why its repeat is slower than its first request. Batch 512 retains the prompt; MTP replays the final prompt token to reconstruct the assistant input and reports 1762 cached tokens. History continuation safely reuses 1759 tokens with batch 512.

Against batch 128, ordinary default 512 is 1.238× faster on medium, 1.162× on the first long request, 2.456× on its repeated long request, and 7.345× on history continuation. These are output-matched request-latency gains, with cache benefits included. MTP reduces the repeated-long completion from 11.989 s to 7.104 s (1.688×).

## Short requests after long histories

| Configuration | Stochastic repeat s | Greedy repeat s | Greedy output tok/s | Greedy first delta ms | Greedy cached / 47 |
|---|---:|---:|---:|---:|---:|
| 32 | 1.200 | 1.211 | 26.41 | 4.662 | 47 |
| 128 | 1.230 | 1.203 | 26.60 | 2.465 | 47 |
| 512 | 1.251 | 1.183 | 27.05 | 2.468 | 47 |
| 512-mtp | 0.792 | 0.777 | 41.16 | 130.435 | 46 |

These requests emit the same 32 output tokens. The ordinary greedy completions remain near 1.2 s after long histories, rather than the earlier 4–6 s resident-server stalls. The listed rates are output tokens per HTTP wall second, not sustained GPU decode rates. MTP completes the greedy repeat 1.521× faster than ordinary 512, with a larger first-delta delay (130.435 ms versus 2.468 ms).

## Observed memory headroom

MiB uses 2²⁰ bytes. Each pair is GPU0 / GPU1. Visible VRAM has a 256 MiB capacity on each card; full VRAM has 6144 MiB.

| Configuration | Peak visible VRAM MiB | Remaining visible headroom MiB | Peak full VRAM MiB | Peak GTT MiB |
|---|---:|---:|---:|---:|
| 32 | 87.250 / 94.574 | 168.750 / 161.426 | 4291.656 / 4275.801 | 75.633 / 51.570 |
| 128 | 222.250 / 229.574 | 33.750 / 26.426 | 4465.000 / 4449.145 | 214.852 / 190.789 |
| 512 | 237.250 / 244.574 | 18.750 / 11.426 | 4633.375 / 4617.520 | 246.727 / 222.664 |
| 512-mtp | 241.594 / 248.824 | 14.406 / 7.176 | 5054.277 / 5038.328 | 246.812 / 222.750 |

| Configuration | Tracked RSS at readiness MiB | Peak tracked RSS MiB | Growth from readiness MiB |
|---|---:|---:|---:|
| 32 | 6928.656 | 6948.098 | 19.441 |
| 128 | 6929.109 | 6950.551 | 21.441 |
| 512 | 6929.387 | 6964.133 | 34.746 |
| 512-mtp | 6951.945 | 6963.520 | 11.574 |

Ordinary visible-VRAM usage reaches its maximum by the medium request and remains flat through long generation, sampler changes, all eight varied prompt shapes, and the final snapshot. MTP reaches its maximum in the final snapshot after the existing protocol suite. The smallest observed headroom is 7.176 MiB on GPU1 with MTP. The four-graph bound prevents the earlier 256 MiB residency cliff in this tested request mix.

These are 20 snapshots per configuration, not continuous allocation maxima. Device counters include other clients; RSS sums this trial’s tracked processes. Readiness precedes lazy peer mappings and command recording, so readiness growth includes that initialization. The supported claims are the observed bounds, exact output checks and measured completion times; this audit does not establish a universal memory or performance maximum.
