# System power-off investigation: incite, October 7

**Confirmed:** incite's first D700 PCIe branch lost its link at 11:31:30, followed
by the whole machine powering off. The user found both its power LED and fan off.
No final panic or orderly OS shutdown was captured. This is a system-level failure;
the initial GPU/transport hang and the later machine power-off are separate events.

The cause is not yet established. GPU/board power delivery and platform firmware
remain possible, as does Linux's handling of PCIe errors on this Mac. No component
replacement is justified by the current evidence alone.

## Timeline and direct observations

- 11:14:13: both incite GPUs timed out during the first post-repaste TP4 run.
  amdgpu performed legacy PCI CONFIG resets, reported VRAM loss and recovery.
  Incite remained reachable. Both GPU device dumps were captured automatically
  to disk and copied to the external workstation; see the compressed dumps in
  `results/kernel/repaste-20261007/failure/`.
- 11:31:30.021: final incite telemetry: GPU edges **64/63 C**, CPU sensor TC0P
  **42.625 C**, PSU-associated sensor Tp0F **49.355 C**, fan **1900 RPM**. SMC
  reported CPU power **84.75 W**, GPU power **127.94/129.63 W**, and total **403.22 W**.
  These are SMC sensor readings, not calibrated wall-power measurements.
- 11:31:30.292: root port `0000:00:02.0` logs `Slot(3-3): Power fault`.
- 11:31:30.340: the same port logs fatal AER **SDES / Surprise Down**, a Data
  Link Layer failure; graphics/audio functions `02:00.0/.1` become inaccessible.
- 11:31:30.341: kernel logs `Slot(3-3): Link Down`.
- 11:31:30.409–.937: invite observes all three Thunderbolt peers disconnect.
- User manually powers incite on. New boot starts at 11:33:21 with ID
  `c7692794-b9bd-44fe-bfe8-2401ddd8f04d`.

The final kernel events survived only in external netconsole. The previous disk
journal's last entry is 11:30:29; no more messages occurred before the abrupt
failure, and the last fraction of a second was not persisted by journald.

The GPUs were not at their high-temperature limits in the final sample. This does
not rule out an unsampled electrical fault or other component temperature, but the
evidence does not support a GPU die overheating explanation for this power-off.

## The power-fault message is not a PSU diagnosis

PCIe Slot Status `PowerFlt+` is present on incite's root port `00:02.0` after
reboot, on healthy invite's same port, and in incite's October 5 snapshot
`results/kernel/p2pdma-20261005/suspended-run/lspci-vvv-rootports.txt`.
Port `00:03.0` has `PowerFlt-` on both Macs. Therefore the printed power-fault
message may report a pre-existing or platform-specific status bit. It cannot
establish that the PSU failed or that the total-power reading exceeded a limit.

The **new fatal Surprise Down and link loss** remain meaningful evidence. Root
`00:02.0` feeds the first D700. Thunderbolt is under `00:01.0`; Ethernet is under
other PCIe branches. Loss of all connectivity and the observed fan/LED state
establish a wider platform stop following the GPU-branch failure.

SMC MSSD was **5 before and after** incite's power-off. This unchanged, untimestamped
value does not prove an orderly shutdown. Current pstore is empty; archived pstore
predates this event. No captured CPU machine-check, ECC error, critical thermal
event, or OOM explains the stop.

Both machines share a power strip, but the user reports other attached devices
appeared unaffected. That weakens a shared-strip/circuit outage explanation;
it does not rule out a bad socket, cable, transient input drop, or internal supply.

## Linux hot-plug path under investigation

The Mac firmware advertises its internal GPU root ports as hot-plug slots with
power controllers. Linux assumes native PCIe hot-plug/AER control at boot. In the
installed 7.2.9 source, the power-fault interrupt handler changes indicator state;
on link loss, the hot-plug path removes the downstream device and calls
`pciehp_power_off_slot()`. That writes the slot power-control bit.

This identifies a plausible **reaction to the link failure**, not proof of what
first caused the link to fail or why the entire machine stopped. No hot-plug,
AER, GPU recovery or power-management protection was disabled during this audit.
Source references: [pciehp controller](https://github.com/torvalds/linux/blob/master/drivers/pci/hotplug/pciehp_ctrl.c),
[hot-plug hardware handling](https://github.com/torvalds/linux/blob/master/drivers/pci/hotplug/pciehp_hpc.c).
The exact installed source was also inspected in the kbuild VM.

## Additional observation installed

`system-platform-trace.service` is enabled on both Macs. An isolated tracefs
instance records:

- PCIe hot-plug/link events, AER, memory-failure and machine-check tracepoints.
- GPU PCI CONFIG reset entry.
- Slot on/off/disable calls, indicator changes, and slot-control command/mask values.
- Kernel, machine and ACPI software power-off entry.
- Call stacks for important slot-power, GPU-reset and software-power-off calls.

The service saves bounded, fsynced local JSONL and transmits each record to the
existing external receiver. A harmless read of each Mac's slot power status
generated `slot_read` events that were verified on the workstation. A temporary
stack trigger on that same read also delivered full call stacks from both Macs
to the external receiver. The startup trigger detection was corrected to ignore
comment lines listing available triggers; active stack triggers were checked explicitly. No slot power
was toggled to test the recorder. Trace hooks observe control flow without
changing power state or recovery policy. User-space relay latency can still lose
the very last events in an instantaneous power cut.

Files/services:
`/usr/local/sbin/system-platform-trace.py`,
`/etc/systemd/system/system-platform-trace.service`,
`/var/log/trashcan-platform/events.jsonl`, and
`/sys/kernel/tracing/instances/trashcan_platform`.

## Next evidence needed

GPU workload testing was stopped after the power-off. At the user’s request, controlled
prefill A/B measurements resumed after a fresh boot with the platform tracer active;
see `results/qwen/prefill-slab-20261007/`. Inference cancellation work and the
post-repaste temperature comparison remain unfinished and uncommitted.

The user ran Apple Diagnostics on both incite and invite and reported that neither
found issues. No reference codes were supplied. This is an independent check outside
Linux; it does not rule out an intermittent failure under load. The user will notify
us when the Macs are back after the circuit change. GPU testing remains stopped.
[Apple's instructions](https://support.apple.com/en-gb/102550).

Any subsequent reproduction should change one variable, use a fresh boot rather
than a machine that already recovered from a GPU reset, and keep external traces
active. Candidate distinctions include hardware failure outside Linux versus
Linux PCIe hot-plug/recovery behavior. No deliberate power-off or slot-control
experiment has been performed.
