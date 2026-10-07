# Additional crash capture installed 2026-10-07

Installed and enabled on **incite and invite** before resumed verification:

- Persistent systemd journal with `SyncIntervalSec=1s`, 256 MiB retention, and
  journal rate limiting disabled. Kernel netconsole remains enabled.
- `system-crash-capture.service` starts on boot and restarts on failure. Each
  second it saves and externally sends GPU temperatures, PCIe link speed/width,
  upstream AER counters, fan speed, and raw SMC diagnostic keys. SMC samples include
  GPU/CPU/ambient and additional SMC temperatures, and CPU/GPU/total power readings. Raw key type
  and bytes are retained alongside decoded values. MSSD is still an unmapped
  shutdown cause; the recorder assigns no new meaning to it.
- Device-coredump watcher polls every 200 ms, copies each kernel-created dump to
  a root-only file, fsyncs it and its directory, and announces its size and SHA-256
  to the workstation. Reading the installed kernel's dump callback copies an
  already formatted buffer; it does not invoke another GPU job or reset.
- The workstation receiver is now a macOS LaunchAgent (`KeepAlive`, `RunAtLoad`).
  It fsyncs received messages and automatically copies announced dumps to this
  workstation through SSH, checking size and SHA-256 before finalizing each file.
  Dump copying uses established tailnet addresses and each node's already trusted
  LAN host-key alias; strict host-key verification remains enabled. Inference
  verification uses Thunderbolt streams exclusively.

## Files and services

On each Mac Pro:

```
/usr/local/sbin/system-crash-capture.py
/etc/systemd/system/system-crash-capture.service
/etc/systemd/journald.conf.d/90-trashcan-crash.conf
/var/log/trashcan-crash/telemetry.jsonl
/var/log/trashcan-crash/<boot-id>/devcd*-<timestamp>.dump
```

The service reads its receiver address from the existing
`/etc/default/netconsole-target`. Telemetry rotates at 16 MiB with three backups
per machine. Dumps are capped at 256 MiB each and retained for investigation.
Kernel/journal limits, dump readability, or abrupt loss of power can still prevent
capture. UDP records can be lost; timestamps, boot IDs and sample uptime expose
gaps. The external workstation must remain awake and connected. This adds no
whole-system kdump reservation and requires no kernel change or reboot.

On this workstation:

```
~/Library/LaunchAgents/local.trashcan.netconsole.plist
~/.local/libexec/trashcan-netconsole-listen.py
~/dev/claude-work/trashcan-kbuild/netconsole-incite.log
~/dev/claude-work/trashcan-kbuild/crash-dumps/
~/.local/libexec/netconsole-errors.log
```

The external log rotates at 64 MiB with five backups. Dump files are mode 0600;
local Mac Pro directories are root-only. Reinstall/update the workstation receiver
with `python3 scripts/install-netconsole-receiver.py`. The node's standard
`nodes/macpro/provision.sh HOST system` workflow now installs the recorder too.
Do not rerun the whole system provisioner just to inspect logs: it also restarts
other node services. To inspect the installed recorder:

```sh
ssh invite 'systemctl status system-crash-capture --no-pager'
ssh invite 'sudo tail -n 2 /var/log/trashcan-crash/telemetry.jsonl'
```

SMC collection writes only the sysfs read-selection index under a shared lock and
restores it afterward. No actual SMC control key, GPU reset setting, audio power
setting, or watchdog setting was changed. Idle sample duration was approximately
22 ms, mostly spent reading sysfs/SMC sensors. Telemetry remains enabled during
the benchmark; new timings therefore include recording overhead.

## Validation

- Both systemd services active/enabled; live telemetry received from both nodes.
- Test `/dev/kmsg` markers from both nodes received externally.
- Synthetic device-dump fixture exercised the watcher, local persistence,
  announcement, workstation copy and hash verification. No GPU reset or system
  crash was induced. See `capture-validation.txt`.
- Python syntax and shell syntax checks passed.

The cable replug exposed a separate recovery defect: service discovery order had
swapped net and stream indices on incite. `tblink` previously assumed net service
`.0`, while the stream setup assumed `.1`. Both now discover the relevant service
instead of assuming its number. All three stream endpoints were recreated and
both nodes report their links up. The middle physical link negotiated one lane;
the others negotiated two. Correctness verification can proceed, but absolute
throughput is not directly comparable to the earlier six-lane measurements.
