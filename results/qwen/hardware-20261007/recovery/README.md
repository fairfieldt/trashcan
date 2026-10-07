# Post-power-loss recovery, October 7, 2026

Read-only recovery collected at approximately 13:47–13:49 PDT, before paired model
benchmarks resumed. No GPU jobs, governor changes, driver/module changes, service
changes, reboots, or writes to either remote workspace were performed by this
recovery task. The parent task separately coordinates current benchmark work.

## Readiness snapshot

| Host | Current boot ID | Kernel | CPU governors |
|---|---|---|---|
| incite | `43167ad0-6779-4307-a7ec-1a600edd52df` | `7.2.9-p2p` | all 24 `schedutil` |
| invite | `96d9c5b9-f34a-49a6-9713-f945ea225320` | `7.2.9-p2p` | all 24 `schedutil` |

All saved values in both `slab-governors.before` files equal the current values.
No governor restoration was needed. Both crash-capture and platform-trace services
were active, enabled, and running with zero restarts and successful main status.
The workstation LaunchAgent was running and receiving fresh telemetry from both
new boots. See `external-receiver/`.

Both hosts had **two** Thunderbolt links: bus 0 ↔ bus 0 and bus 2 ↔ bus 2, each
with carrier, 10 Gb/s speed per lane, and two lanes in each direction. Bus 1 had
no peer. `thunderbolt_stream` was not loaded and `/dev/tbstream*` endpoints did not
exist. Thus this snapshot was **not ready for the prior three-stream TP4 setup**.
These are read-only observations, not a transport recovery attempt. See each
host's `state.txt`, `thunderbolt-readiness.txt`, and `tblink-readiness.txt`.

## Interrupted slab run recovered

The old binary SHA-256 on both machines is
`ff199d8eb204f952c9865853862aa972b65ff28ab9ab7f60e6f9a2819fd23e74`.
Source and original commands are documented in `../../prefill-slab-20261007/`.
Recovered workspace files include both hosts' logs, clocks, exact benchmark
commands, scoped latency wrapper, and saved CPU governors.

One complete candidate repetition survived, followed by an interrupted second
repetition. Every complete repetition has exactly equal prefill top scores and
all eight generated token IDs. The comparison used TP4 Qwen3.8 27B Q4_0, pp4096,
tg8, context8192, all three streams, performance CPU governors and scoped 0-us
CPU latency requests.

| Slab | Repetition | Prefill tok/s | Decode tok/s | Incite full-clock samples | Invite full-clock samples | Maximum sampled prefill temperature, incite / invite |
|---|---:|---:|---:|---:|---:|---:|
| 64 baseline | 0 | 145.573 | 29.270 | 106/106 | 106/106 | 59 / 80 °C |
| 64 baseline | 1 | 139.386 | 30.199 | 110/110 | 108/108 | 66 / 88 °C |
| 256 candidate | 0 | 143.654 | 28.973 | 108/108 | 104/104 | 69 / 92 °C |

Full clock means **both local GPUs at 850 MHz core / 1370 MHz memory**. The single
candidate result falls inside the two baseline prefill results. It does not
establish an end-to-end speedup. The candidate clock files end in NUL-filled
partial data after the complete repetition; raw files are unchanged, and the
summary excludes invalid lines. See `summary.json` and `summarize.py`.

## Recovered final records

Disk telemetry continued beyond the external receiver's last records:

| Host | Last prior-boot disk telemetry (PDT) | Final recorded GPU temperatures | External last telemetry (PDT) |
|---|---|---|---|
| incite | 12:11:12.334944 | 72 / 70 °C | 12:10:54 |
| invite | 12:11:11.462753 | 96 / 74 °C | 12:10:53 |

The user reported a circuit-breaker trip. These logs do not independently establish
its cause or settle earlier shutdown incidents. They show that the external log
stopped before the final records persisted on the nodes. Both last disk samples
still report active GPUs, 8 GT/s PCIe ×16 GPU links, and fans near 1900 RPM.

The final incite platform record at 12:11:12.374661 is:

```
pcie_link_event: 0000:00:01.0 type:4, reason:3, cur_bus_speed:20, max_bus_speed:22, width:8, flit_mode:0, status:LT|DLLLA
```

This raw PCIe event was absent from the earlier external tail. No captured software
poweroff call stack or GPU configuration-reset event appears in the prior-boot
platform records. Invite's last platform event is the known harmless slot-power
status-read self-check at 12:08:17. No device dumps for either interrupted boot
were found. The incite recorder archive also contains two older dump files from
boot `38f2ea0a-69e7-4c43-98ce-df86d1f225ec`; they are not from this interruption.
The pstore directories are empty. Prior kernel and system journal captures are
preserved per host.

## Evidence and collection limitations

- Each host's `commands.jsonl` records exact SSH commands, timestamps, return codes,
  and output sizes. `collect.py` reproduces the principal read-only collection.
- `interrupted-workspace.tar.gz` preserves the requested old workspace data;
  `interrupted-workspace/` exposes the recovered benchmark files for review.
- `recorders.tar.gz` preserves crash telemetry, platform events, and existing dumps.
  `summarize.py` reads those archives directly and writes compact relevant extracts.
- The invite workspace tar returned 2 because incite-only standalone microbenchmark
  files were absent; its results, scripts, and saved governors were recovered.
- The invite recorder tar returned 1 because its current telemetry file grew while
  being read. The archive extracted successfully; all recovered prior-boot JSON
  records parsed successfully. This is a live snapshot, not an atomic archive.
- Initial prior-boot journal calls used hyphenated IDs, which journalctl rejected.
  Successful retries used the same IDs without hyphens. Initial remote `rg` calls
  failed because ripgrep is not installed there; `tblink-readiness.txt` uses grep
  and supplies the missing service/journal observations.
