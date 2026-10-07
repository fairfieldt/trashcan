# Qwen3.8 27B Thunderbolt verification, 2026-10-07

Verification completed on incite and invite after installing additional system
crash capture. Every inference run used three Thunderbolt USB4STREAM lanes,
`/dev/tbstream0,/dev/tbstream1,/dev/tbstream2`; no LAN inference fallback was used.
Both nodes stayed in the same boot throughout these checks. Their final kernel
journals show no compute timeout, GPU reset, PCIe completion timeout, or panic.
All peer sessions finished GPU teardown before the harness terminated the idle
listener. Leases were released and all 24 CPU governors on each host restored.
Crash recorders remain enabled.

## Tested source and settings

The binary `verify-capture` on both nodes has SHA-256
`b52ecdeb0403a3678ea7863b5157a813cdc546c1584c0444d0093d85d0554ddb`, identical to the
previously archived `reviewable` build. Engine source is unchanged from `6d650c4`.
The binary was rebuilt in the kbuild VM using `scripts/cross-build.sh`, then copied
to `/home/fairfieldt/trashcan-exp/improvements-20261007/verify-capture` on both Macs.
No kernel or engine source changes were needed for this verification.

Models on both nodes:

```
/home/fairfieldt/models/qwen3.8-27b/Qwen3.8-27B-trashcan-Q4_0.gguf
/home/fairfieldt/models/qwen3.8-27b/mtp-Qwen3.8-27B-trashcan-Q4_0.gguf
```

Both hosts report performance CPU governors, CPU latency requests of 0 us, and
affinity covering CPUs 0–23 during each run. Each command was wrapped with
`scripts/with-clocks.sh` on both hosts. Experimental flags are recorded in every
benchmark row. Tiled attention and KV-only updates remain opt-in.

The cable replug recovered Linux peer discovery. It also exposed reordered
Thunderbolt service indices: net was `.1` and stream `.0` on incite, while the
other end used the opposite order. `tblink` and the new `scripts/tbstream-up.sh`
now discover the relevant service instead of assuming its number. Three stream
endpoints were recreated. Physical lane counts were 2, 1, 2 on both ends. Both
configurations in each comparison used this same link layout; absolute results
should not be pooled with earlier six-lane sessions.

## Correctness results

`tests/fixtures/qwen-bench-corpus.jsonl` was copied as `corpus-verify.jsonl` and run
twice per case with context 4096 and default maximum generation 128. The sustained
case overrides that maximum to 2048. The corpus includes code, Spanish, Chinese,
native tool request/result messages, thinking, seeded sampling, and sustained
generation. The tool cases ended naturally at 26 and 22 tokens.

- Plain scalar/full projection: **8 cases × 2 repetitions completed**.
- MTP with tiled attention and KV-only updates: **8 cases × 2 repetitions completed**.
- **All 14 greedy MTP/plain token-ID comparisons matched exactly**, including
  both 2048-token sustained repetitions.
- Every case, including seeded sampling, repeated with identical token IDs within
  each configuration. Sampled IDs differ between plain and MTP, as expected from
  their different sampling draw sequences; no cross-mode sampled equality claim
  is made.
- A third sustained MTP repetition after cooling also matched the plain IDs.
- At context 12288 with 8192 prompt tokens and 32 generated tokens, scalar and
  tiled attention produced identical generated IDs and identical prefill top-8
  IDs. Maximum top-score difference was **0.0008277893**, with finite scores.

## Invite clock reduction

The second D700 (`06:00.0`) is the main observed limit during the warmed MTP
corpus. Its memory stayed at 1370 MHz, but shader clocks increasingly alternated
between 450, 650 and 850 MHz as its temperature rose. In timed generation samples,
clock reductions first appear at 87–88 C; they dominate at 89–95 C. Earlier MTP
cases held 850 MHz through 86–87 C. This correlation is preserved in
`throttle-summary.json`, with complete raw clock logs and external telemetry.

| Sustained case, same prompt/output/settings | tok/s | Invite peak GPU temperature during generation | Both invite GPUs at 850/1370 MHz |
| --- | ---: | ---: | ---: |
| Plain, two repetitions | 27.93 / 28.24 | up to 87 C | 551/551 samples |
| MTP after the mixed corpus, two repetitions | 29.98 / 30.22 | up to 95 C | 39/512 samples |
| MTP repeated after cooling | 39.52 | 79 C | 197/197 samples |

The cooler MTP repeat also held both incite GPUs at full clocks in all 197
generation samples. It started with invite at approximately 62/67 C and generated
the same 2048 IDs. Its rate is about **31% higher** than the warmed MTP average.
This supports temperature-related power/thermal limiting as the leading cause of
the observed speed loss, rather than workload shape alone. It is one roughly
52-second generation from a cooler start, not proof that 39.52 tok/s can be held
indefinitely.

The shared fan reached about 1900 RPM, matching its reported maximum. GPU power
management remained `auto`; no clock, voltage, fan limit, power limit or protection
override was applied. The SI driver does not expose a usable busy-percentage
counter here, and these captures do not identify an explicit firmware clamp reason.
The exact thermal-versus-power mechanism therefore remains unresolved.

This does **not** establish the prior shutdown cause: this morning's reset began
at roughly 54/61 C, while the current higher-temperature runs completed without a
reset or system stop. SMC power values in the telemetry are sensor readings, not
calibrated wall-power measurements.

## Long-context measurement

| 8192-token prefill, same binary/link layout | Prefill tok/s | Generation tok/s |
| --- | ---: | ---: |
| Scalar | 110.34 | 25.46 |
| Tiled | 134.43 | 25.82 |

Observed prefill gain is **21.83%** in this single paired check, consistent with
the earlier measured gain. Incite held both GPUs at full clocks in every timed
sample. Invite scalar prefill held full clocks in 274/274 samples; tiled prefill
held full clocks in 220/221 samples, with one initial 300/150 MHz sample on GPU0.
Both generation windows held full clocks throughout. Full counts, temperatures
and numerical checks are in `long-context-summary.json`.

The warmed MTP corpus is retained as a correctness/stability result, not a clean
comparison of optimization throughput, because clocks changed during it.

## Reproduction and evidence

GPU commands require a lease on each machine. The first corpus pair used
incite/invite tickets 58/49; the cooled MTP and 8K pair used 59/50. With freshly
granted tickets:

```sh
python3 scripts/verify-qwen-tp.py --incite INCITE_ID --invite INVITE_ID \
  --name resumed-plain --env 'TRASHCAN_QWEN_ATTN=scalar TRASHCAN_TP_KV_ONLY=0' \
  --args '--corpus corpus-verify.jsonl --tg 128 --reps 2 --context 4096'
python3 scripts/verify-qwen-tp.py --incite INCITE_ID --invite INVITE_ID \
  --name resumed-mtp --env 'TRASHCAN_QWEN_ATTN=tiled TRASHCAN_TP_KV_ONLY=1' \
  --args '--corpus corpus-verify.jsonl --tg 128 --reps 2 --context 4096' --mtp
```

The cooled repeat uses `--corpus corpus-sustained-only.jsonl --reps 1` with the
same context and flags; its fixture is included here. Long-context runs use
`--pp 8192 --tg 32 --reps 1 --context 12288`, with scalar/tiled attention and
KV-only disabled in both.

This directory contains exact benchmark JSONL, driver and peer logs, clock logs,
host setup/build hashes, final kernel journals, the corpus clock summary,
long-context checks, and 2036 external telemetry records in `telemetry.jsonl.gz`.
Telemetry remained enabled throughout verification, so all new timings include
recording overhead. Additional crash-capture installation and synthetic validation
are documented in `results/kernel/invite-crash-20261007/CAPTURE.md`.
