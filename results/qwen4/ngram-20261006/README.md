# N-gram table placement on incite (2026-10-06)

`scripts/qwen4-ngram-bench.sh` (one `locked-run.sh` run per mode), binary from the commit "Add qwen4exp GPU stages,
the machine chain and an n-gram table benchmark" (`b34d1ef` before the rebase onto `a40bde3`). The table is
`per_layer_token_embd` of the Q8_0 GGUF: 320,001,536 rows of 160 values, 170 bytes per row, in its own 54.4 GB shard
on incite's SSD (Intel 660p 1 TB QLC NVMe, linked at PCIe 2.0 x4). Each token needs 16 rows. `corpus` hashes the
first 8192 tokens of the quality corpus (real n-gram repetition), `random` uniformly random tokens. Decode gathers
one token's 16 rows (4096 tokens timed); prefill gathers 512-token batches (16 readers).

| storage | resident | decode, 1 reader, p50 | decode, 16 readers, p50 / p99 | prefill (16 readers) |
|---|---|---|---|---|
| RAM, Q4_0 rows (90 B, requantized at load in 52-57 s) | 26.8 GiB | 0.07 ms | 0.39 / 0.62 ms (thread start-up) | 825-950k tok/s |
| SSD, page cache, cold start, corpus | page cache | 0.20 ms | 0.40 / 0.60 ms | 10.1k tok/s |
| SSD, page cache, cold start, random | page cache | 2.30 ms | 0.40 / 0.60 ms | 2.76k tok/s |
| SSD, O_DIRECT 4 KiB pages, corpus | 0 | 2.40 ms | 0.74 / 1.29 ms | 2.68k tok/s |
| SSD, O_DIRECT 4 KiB pages, random | 0 | 2.26 ms | 0.72 / 1.25 ms | 2.74k tok/s |

A random 4 KiB read costs about 140 us on this drive; 16 parallel readers fetch a token's rows in 0.4-0.7 ms
and sustain about 44k reads per second (2.7k tokens per second) without help from the cache. Real text repeats
n-grams, so through the page cache prefill reaches 10k tokens per second on the corpus. A resident Q8_0 copy
(50.7 GiB) was not tried: with 62 GiB of RAM it would leave too little for the OS and risks the OOM killer.

For scale: a decode step through 48 GPU layers takes tens of milliseconds and the measured GPU prefill is well
under 1k tokens per second per stage, so either SSD path costs about 1% of decode and is not a prefill limit.
