# Live maximum fan policy — 2026-10-07

The first deployment was commit `a1e90ea204d1d038da3797dcdeedeb10c5482f6f` on incite and invite.
The interval validation follow-up from `53c228d` is now deployed too; see the final readings below. The user requested maximum
fan speed at all temperatures. Both services were active and enabled after deployment, with
`NRestarts=0` and `ExecMainStatus=0` through the observation below. No GPU work was run during
the fan probe/deployment; subsequent GPU measurements are recorded separately.

The old script read `fan1_max` once at startup. Both boot journals report a cached 1900 RPM
maximum, while the probe's initial snapshots show live maxima of 2363 RPM (incite) and
2357 RPM (invite). The old service still requested 1900 RPM. Restarting that service refreshed
its cached limit, but its temperature curve could still reduce speed: incite's pre-deployment
journal later shows 2048–2205 RPM at 48–49 °C.

The new [script](../../../../scripts/fancurve.sh) reads the live SMC maximum every two seconds,
follows increases and decreases, and checks manual mode and target readback each poll. It
repairs either setting if changed. It sets no fixed RPM above the reported limit. Small RPM
jitter is applied without logging every change. Two transient maximum-read errors may retain
the last valid target; a third consecutive failure exits with an error. Startup has no cached fallback. Invalid
limits, other control reads/writes, or failed readback cause an error exit. Normal exit and
TERM/INT attempt to return `fan1_manual` to 0; a failed restoration write is logged and exits
with an error. The service retries after two seconds. An explicit service stop returns
automatic SMC control without a restart. These failure paths were tested with local mocks,
not by injecting faults into the running machines.

The bounded probe stopped the old service, set manual mode and the then-current maximum,
took 30 one-second samples, and restarted the old service in `finally`.

| Host | Initial target / tachometer | Requested maximum | Tachometer after 30 s | Tachometer increase |
| --- | ---: | ---: | ---: | ---: |
| incite | 1900 / 1898 RPM | 2363 RPM | 2362 RPM | 24.45% |
| invite | 1900 / 1900 RPM | 2357 RPM | 2356 RPM | 24.00% |

Those percentages compare the final tachometer reading with the initial one. The separate
`rpm_gain_percent` in [probe-summary.json](probe-summary.json) uses the last-ten-sample median
and is 24.32% / 23.92%. This measures fan RPM only: airflow, physical fan maximum, cooling
capacity, and model throughput were not established by this probe.

After deployment, 30 one-second samples covered approximately 21:25:45–21:26:14 UTC:

| Host | Live maximum range | Target range | Tachometer range | Manual mode |
| --- | ---: | ---: | ---: | ---: |
| incite | 2362–2365 RPM | 2362–2365 RPM | 2359–2364 RPM | 1 in all samples |
| invite | 2370–2373 RPM | 2370–2373 RPM | 2370–2372 RPM | 1 in all samples |

The one-second sampler may observe a target from the previous two-second control poll;
the largest observed target/limit difference was 2 RPM on either host. Raw observations
are in [incite-after.jsonl](incite-after.jsonl) and [invite-after.jsonl](invite-after.jsonl),
with service state and privileged journals in the corresponding `*-after-service.txt` and
`*-after-journal.txt`. The `-- No entries --` line in the service captures is not the journal
evidence; use the separate journal files.

`*-during-probe-context.*` contains exact read-only SSH commands, installed old code, and boot
journals captured at 21:21:09 UTC, during the 21:20:44–21:21:14 probe. The inactive service in
that snapshot is intentional. Use the `phase: before` row in `*-live-max-probe.jsonl` for the
1900 RPM baseline. Deployment backups were retained on each host in
`/home/fairfieldt/trashcan-exp/tp4-offline-20261007/fan-max-20261007/backup/`.

All commands were launched from `/Users/fairfieldt/dev/trashcan`. These exact command forms
were run once with `HOST=incite` and once with `HOST=invite`, concurrently within each phase;
both finished before the next dependent phase. `HOST` below is a textual placeholder.

```sh
ssh HOST 'sudo -n python3 -' < results/qwen/hardware-20261007/fans/test-live-max.py > results/qwen/hardware-20261007/fans/HOST-live-max-probe.jsonl 2> results/qwen/hardware-20261007/fans/HOST-live-max-probe.err

ssh HOST 'mkdir /home/fairfieldt/trashcan-exp/tp4-offline-20261007/fan-max-20261007' && rsync -a scripts/fancurve.sh scripts/fancurve.service HOST:/home/fairfieldt/trashcan-exp/tp4-offline-20261007/fan-max-20261007/

ssh HOST 'bash -s' < results/qwen/hardware-20261007/fans/deploy.sh > results/qwen/hardware-20261007/fans/HOST-deploy.log 2>&1

ssh HOST 'python3 -' < results/qwen/hardware-20261007/fans/sample.py > results/qwen/hardware-20261007/fans/HOST-after.jsonl 2> results/qwen/hardware-20261007/fans/HOST-after.err
```

[deploy.sh](deploy.sh) records the exact backup, install, daemon-reload, enable, restart,
state, and checksum commands. It changes only the fan script and unit. It is a historical
deployment script, not an idempotent rerun: the stage and backup directories already exist.
[test-live-max.py](test-live-max.py) and [sample.py](sample.py) retain the probe and sampler.

The deployed hashes matched on both hosts, as recorded in `*-deploy.log`:

| File | SHA-256 |
| --- | --- |
| New script | `02a62a95ab37e39c21fc3b1efffc815b7f52916fb9d0af00cc4b55e721e6ec50` |
| New service | `ad1d7e35f7bf8e0053f10329ea008ff842a8ff748f0d18ed7ea4a775b9451de8` |
| Backed-up old script | `75f76a94c099015f07f55d592b2f2a89118cad739dec6faac6937e1433819233` |
| Backed-up old service | `4b125c83edddc73e32d36f9e31128a020c8eafd167e7eb485fd353936cc86723` |
| Probe script | `448e314ca82d002e63085d8dc2b73594b604aa6d10f44986b467796486d4425b` |
| Deployment script | `ab7bb23a163aca6534e4993015b8395a93f17add23dbeb1ce8854ec62b8bc2a9` |
| Sampler | `0aac8de66067b061d30a3204de91f3817e1fdd066677af97a05da07c4be8a7bc` |

Local validation passed all 11 mock tests, Bash syntax, and whitespace checks. The tests
cover live-limit increases/decreases, independent mode/target drift, bounded stale-limit
fallback, startup and malformed-limit failures, control readback, stop restoration, and
quiet jitter handling. Exact source/test hashes and results are in
[local-validation.json](local-validation.json); root's independent rerun is in
[local-tests.log](local-tests.log).

```sh
python3 scripts/test-fancurve.py
bash -n scripts/fancurve.sh
git diff --check -- scripts/fancurve.sh scripts/fancurve.service docs/hardware.md
```

Follow-up source review found that the original numeric `INTERVAL <= 10` check accepted
`18446744073709551617` after Bash integer overflow. The default two-second interval was
unaffected. Validation now accepts only the literal integers 1–10, and a local mock regression
checks rejection and return to automatic control before any polling. This follow-up does not
change the fan policy. The deployment hashes above remain the historical first deployment;
the follow-up's local checks and hashes are recorded separately in
[interval-validation.json](interval-validation.json), with host redeployment recorded by the
coordinating run.

## Final interval-guard deployment

After all GPU tests completed and both queues were empty, the config-validation
follow-up from `53c228d` was installed on both hosts. The maximum-fan policy and
default two-second polling interval did not change. Only the script was replaced;
the unit hash is unchanged. [deploy-interval.sh](deploy-interval.sh) verifies the
old/new script and unit hashes, preserves the previous script, installs atomically
and restarts the service. Both services remain enabled and active with zero
automatic restarts and exit status 0.

The final installed script SHA-256 is
`b175a2bb41002a52a133a1d6ed7eb2988a421a4d2bac9a8005efed3fd6b0b3c3`.
Thirty one-second samples after restart show:

| Host | Live maximum / target range | Tachometer range | Manual mode |
| --- | ---: | ---: | ---: |
| incite | 2403–2408 RPM | 2404–2409 RPM | 1 in all 30 samples |
| invite | 2418–2421 RPM | 2417–2421 RPM | 1 in all 30 samples |

The largest sampled target/live-limit differences are 3 and 2 RPM respectively,
consistent with the two-second control interval and asynchronous readings. These
remain RPM observations only. Backups are in each host's
`/home/fairfieldt/trashcan-exp/tp4-offline-20261007/fan-interval-20261007/backup/`.
All 24 CPU governors per host are restored to schedutil, no experiment model
process remains, and all three Thunderbolt endpoints remain accessible.

Exact command forms (each run with HOST=incite and HOST=invite):

```sh
ssh HOST 'mkdir /home/fairfieldt/trashcan-exp/tp4-offline-20261007/fan-interval-20261007'
rsync -a scripts/fancurve.sh HOST:/home/fairfieldt/trashcan-exp/tp4-offline-20261007/fan-interval-20261007/
ssh HOST 'bash -s' < results/qwen/hardware-20261007/fans/deploy-interval.sh
ssh HOST 'python3 -' < results/qwen/hardware-20261007/fans/sample.py
ssh HOST 'python3 -' < results/qwen/hardware-20261007/scripts/final-host-state.py
```

`*-interval-deploy.log`, `*-interval-after.jsonl`,
[interval-deployment-summary.json](interval-deployment-summary.json), and
`../incite/final-state.json` / `../invite/final-state.json` retain the results.
These final deployment records are separate from the original fan-script hashes
captured by every GPU benchmark scope.
