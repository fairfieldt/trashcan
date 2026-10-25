# Preparation checks, 2026-10-25

Inspected source at `fc22c1fb`, dirty mixer changes in `engine/src/qwen4/gpu.rs` and
`tools/qwen4_budget.py`, archived results, existing worktrees, cleanup task #1361, kernel/lights task #1127,
mixer task #1291 and parent #1327. Main cleanup was in progress; no files in its worktree were edited here.

Local checks:

- `python3 scripts/test-tblink.py`: 6 passed (fake sysfs/configfs only).
- `python3 scripts/test-fancurve.py`: 30 passed (local mocked controller checks).
- `python3 tools/qwen4_budget.py --check`: all 21 historical allocation checks match within 0.2 MiB.
  This validates bookkeeping against old measurements; it does not validate the dirty candidate's GPU allocation.
- `cargo test --manifest-path ~/dev/homelab/portlights/Cargo.toml --quiet`: 6 passed; no device access.
- `bash -n` on A1's `deploy.sh`, `mc-s1.sh`, `mc-chain.sh`: passed.
- `cmpdump.py`: changed the report-only comparator to return failure for differences and explicitly handle
  unequal lengths. Five temporary synthetic fixtures: identical exit 0; changed, truncated, missing and empty reference exit 1.
  Reproduce: `python3 results/release-readiness-20261025/verify-cmpdump.py`.
  No real GPU dump was produced. The baseline/candidate source patch is archived, not applied to lab's runtime.

Read-only SSH probes to incite, invite, indite, enlight, inflight and alight all timed out with
`ConnectTimeout=4`, `ConnectionAttempts=1`, `BatchMode=yes`. No remote command ran; current booted kernel,
portlights deployment and node service states could not be confirmed. No wake/reboot/recovery attempted.
The historical local `~/dev/claude-work/mixcopies-20261023/` binary directory was absent. No newer run output was
found in `results/qwen4/mixer-copies-20261023/` (README and tools only before this snapshot).

Parent #1327 checked To-dos: A1 mixer hardware tests and A14 blog review only; no existing release/kernel/lights
items. #1361 confirmed it owns main cleanup and that `lab@fc22c1fb` preserves the removed evidence/series/scripts.
It is raising the license question and revising kernel docs to separate optional 0009/0010 from tested kernels.

Archive verification: every local Markdown file link resolves against the lab tree plus this snapshot;
`git apply --cached --check candidate.patch` passes against `lab@fc22c1fb` using a temporary index.
Only documentation, prepared test helpers and an unapplied source patch are committed; active runtime files stay unchanged.
