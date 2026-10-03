# Fixed-dimension benchmark clock analysis — 2026-10-03

**Both native GPUs report 850 MHz core / 1370 MHz memory in every sampled row of all six reconstructed decode windows**, including their one-second-trimmed interiors. This supports comparing the recorded decode rates at the same sampled clock state; it does not establish continuous residency or exact inference boundaries.

The two repetitions per build and the long comparison use the build identities in [fixed-dim/build.json](fixed-dim/build.json). Full captured metadata, checksums, clock histograms and shifted-anchor sensitivity results are in [fixed-dim-clock-analysis.json](fixed-dim-clock-analysis.json).

| Benchmark | Decode seconds | Full decode: GPU0 / GPU1 at 850/1370 | Interior: GPU0 / GPU1 at 850/1370 | Full-decode max °C GPU0 / GPU1 |
| --- | ---: | --- | --- | --- |
| decode-shared-1 | 11.207 | 42/42 / 42/42 | 34/34 / 34/34 | 71 / 69 |
| decode-fixed-1 | 11.022 | 40/40 / 40/40 | 33/33 / 33/33 | 73 / 71 |
| decode-shared-2 | 11.271 | 42/42 / 42/42 | 34/34 / 34/34 | 74 / 72 |
| decode-fixed-2 | 10.845 | 40/40 / 40/40 | 33/33 / 33/33 | 76 / 74 |
| long-shared | 6.228 | 23/23 / 23/23 | 16/16 / 16/16 | 79 / 77 |
| long-fixed | 5.993 | 22/22 / 22/22 | 15/15 / 15/15 | 83 / 81 |

Counts are qualifying samples / all samples, separately for each GPU. Native GPU0 is PCI 02:00.0 / DRM card0; GPU1 is PCI 06:00.0 / DRM card1. The historical llama.cpp/ggml Vulkan ordering differs. Interior windows remove one second at both ends; the trim is an explicit sensitivity margin, not a measured uncertainty bound.

## Window reconstruction and uncertainty

The benchmark stops its decode timer, prints its JSON result, writes the final profile TSV, and only then drops engine resources. The whole-command end therefore includes cleanup and is unsuitable as an inference-end anchor. The prefill profile is written after prefill timing stops and before decode timing begins. This ordering is verified in the exact tested [main.rs](../../../engine/src/main.rs), [generation.rs](../../../engine/src/generation.rs) and [model.rs](../../../engine/src/model.rs) hashes retained in the JSON artifact.

Local rsync copies retain only whole-second mtimes. Read-only incite stat/hash checks recovered subsecond JSON and profile timestamps, with every inspected remote file matching its local SHA-256 and size. Nominal decode end is the remote JSON mtime; decode start subtracts the reported `tg_seconds`. Combined start additionally subtracts `pp_seconds`. The latter ignores the small untimed prefill-profile write.

| Benchmark | Remote JSON mtime UTC | Inferred decode start minus prefill TSV, ms | Final TSV minus JSON, ms | Wrapper end minus JSON, s |
| --- | --- | ---: | ---: | ---: |
| decode-shared-1 | 2026-10-03T11:38:29.813830+00:00 | +0.719 | 0.000 | 0.707 |
| decode-fixed-1 | 2026-10-03T11:39:12.027501+00:00 | +1.865 | 0.000 | 0.399 |
| decode-shared-2 | 2026-10-03T11:39:53.169233+00:00 | -3.289 | 4.000 | 0.480 |
| decode-fixed-2 | 2026-10-03T11:40:33.155030+00:00 | +2.612 | 0.000 | 0.516 |
| long-shared | 2026-10-03T11:48:38.952317+00:00 | -0.441 | 0.000 | 0.543 |
| long-fixed | 2026-10-03T11:49:46.992582+00:00 | -0.992 | 0.000 | 0.421 |

The phase-boundary checks agree within 3.3 ms, while cleanup extends wrapper completion by 0.399–0.707 s. That agreement supports the reconstruction but does not prove a timing-error bound: file mtimes, stdout flushing, filesystem timestamp granularity and sensor read timing remain indirect observations. The JSON includes endpoint shifts of ±0.25 s and ±1 s, and the original whole-second local-anchor results for comparison.

## Combined prefill and decode samples

These approximate combined windows provide context for the prefill measurements. Clock changes here are sampled states; a lower fraction is not automatically thermal throttling or lower GPU utilization.

| Benchmark | Full combined: GPU0 / GPU1 at 850/1370 | Interior combined: GPU0 / GPU1 at 850/1370 | Full combined max °C GPU0 / GPU1 |
| --- | --- | --- | --- |
| decode-shared-1 | 53/53 / 53/53 | 46/46 / 46/46 | 71 / 69 |
| decode-fixed-1 | 51/52 / 52/52 | 45/45 / 45/45 | 73 / 71 |
| decode-shared-2 | 53/53 / 53/53 | 45/45 / 45/45 | 74 / 72 |
| decode-fixed-2 | 52/52 / 52/52 | 44/44 / 44/44 | 76 / 74 |
| long-shared | 151/152 / 152/152 | 144/144 / 144/144 | 79 / 77 |
| long-fixed | 150/150 / 150/150 | 144/144 / 144/144 | 83 / 81 |

## Clock-log continuity

The captured growing log contains **30,484 rows**, from 2026-10-03T09:37:08.967000+00:00 through 2026-10-03T11:54:17.678000+00:00, with a median interval of 0.267 s. The sampler timestamps first, reads the six sensors sequentially, then sleeps 0.25 s; the actual interval includes those reads. Brief changes can be missed, and sample counts are not a time-weighted duty cycle.

The historical [measurement-windows.json](measurement-windows.json) records a 537,088-byte snapshot. SHA-256 of exactly that prefix of the current log is `18aeda79abc2e7bb6ec483d53006abf5a85101b0981d8996b6e549bf45a2991d`, **matching the historical hash**. Historical window metadata was not modified. The new JSON also records this analysis's full clock snapshot size/hash so a later append can be checked the same way.

The later check/server workloads are outside these six benchmark windows.
