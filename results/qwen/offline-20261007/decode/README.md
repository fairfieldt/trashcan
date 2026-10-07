# Decode work while the Macs are down

Source base: `b0a2e6884d06ac8798df70f7057910ff1ddff3fb`, with the accompanying
working-tree changes in `engine/src/qwen/spec.rs`, `qwen/chat.rs`, and `server.rs`.
All work ran on the local Mac. No remote Mac, GPU, model, lease, or sustained-clock
measurement was used. These results establish host logic, not D700 throughput.

## Generation boundary

Speculative rounds now cap their draft count by `remaining_output - 2`: one slot
is the already sampled pending token, and one is the final lookahead that can be
emitted without a target forward. Both CLI generation and the Qwen server pass
their remaining output budget. The target history contains only processed tokens;
the final lookahead remains unprocessed, as on ordinary decoding.

With two drafts configured and two output tokens remaining, the previous code
ran two MTP steps and verified three target columns. It could then discard part
of the accepted tail. The new path runs one target column, samples the last token,
and emits it without evaluating it. With three tokens left, it drafts one token
and verifies two columns. These are reductions in evaluated work, not measured
latency gains. Greedy equivalence at these existing verification widths still
needs confirmation on D700.

Shorter rounds consume a different number of random draws. Sampled output with
a fixed seed can therefore change under this boundary fix; the target sampling
law is preserved. Stop-token handling, rollback, and checkpoint interfaces are
unchanged.

## Optional adaptive MTP

`TRASHCAN_QWEN_ADAPTIVE_MTP=1` enables a driver-side controller selecting either a
serial target step or the configured MTP draft depth. It compares measured elapsed
time per committed token, including drafting, verification and rollback:

- Start each generation with four serial steps and eight full MTP rounds.
- Retain an exponentially weighted estimate of time and committed tokens for each
  mode. Compare the ratio of these estimates, rather than averaging round rates.
- Require a 5% advantage to switch. After 32 rounds in the selected mode, probe the
  other mode for four serial steps or eight MTP rounds.
- Exclude output/context-limited rounds from MTP cost samples. Include early draft
  stops: rejected stop proposals are real work and must not stall a probe.
- Reset the controller per request; cumulative benchmark statistics remain available.

The flag is off by default. It requires no node flag or protocol change, because
the driver sends ordinary forward/draft requests. Mode decisions only use completed
rounds, so the rejection-sampling proposal law is unchanged. Runtime timings can
change the chosen mode and hence fixed-seed sampled output. `mtp.plain_steps` and
`plain_ms_per_step` expose serial fallback work; benchmark experimental metadata
records the flag. No throughput improvement is claimed for the controller.

## Offline replay

`replay.py` extracts and compiles the actual Rust controller, then models episodes
from the seven completed plain/MTP pairs in
`results/qwen/improvements-20261007/incite/final-corpus-{plain,mtp}.json`.
The failed sustained MTP case is excluded. Run:

```sh
python3 results/qwen/offline-20261007/decode/replay.py > results/qwen/offline-20261007/decode/replay.csv
```

The model samples possible accepted-prefix lengths from recorded counts, using
1,024 seeds per case and constant recorded mean round costs. The original order
of outcomes, per-round costs, and later workload/thermal changes are unavailable.
Serial cost comes from plain runs without the loaded MTP cache's maintenance, and
short tails use serial cost because one-draft timings are absent. Do not treat
these modeled rates as hardware measurements or predicted throughput.

For 128-token modeled episodes, the controller changes the low-acceptance Chinese
case from 26.78 to 28.11 modeled tokens/s. It costs 1.5–3.2% in the other six cases
because it probes serial steps and sometimes follows noisy short acceptance
windows. At 2,048 modeled tokens, Chinese changes from 26.54 to 28.19 while the
other modeled costs are 1.2–2.3%. This is a tradeoff to measure on hardware, and
the evidence does **not** support enabling the controller by default.

## Local checks and pending hardware checks

`cargo test --release qwen::spec::tests` covers budgets across all supported draft
depths; exact greedy tokens and processed histories on a toy target across every
rejection position; controller warmup/probes/hysteresis/recovery; and target
sampling mass while the adaptive controller changes modes. An independent agent
review found no blocking boundary or bookkeeping issue. Full integration checks
are recorded in the enclosing offline results.

After recovery, run `qwen-check`, greedy CLI/server comparisons at output budgets
1 through 8 (and stop tokens at each position), checkpoint reuse after bounded
generation, then paired corpus runs with the adaptive flag off/on. Include low
acceptance cases and long generation. Use paired leases, governor/CPU-latency
settings, `scripts/with-clocks.sh`, and the existing per-machine experiment
workspaces. Keep the controller opt-in pending those measurements.
