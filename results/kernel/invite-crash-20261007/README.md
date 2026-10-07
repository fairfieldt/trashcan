# Invite system shutdown investigation, 2026-10-07

Both incidents have kernel logs. Neither has a captured final explanation for the
whole-machine stop. The repeated observation is a D700 compute timeout followed
by amdgpu's legacy PCI CONFIG reset. This morning's last messages additionally
identify a PCIe completion timeout on the second D700's upstream root port and an
inaccessible audio function on that card. This localizes the observed failure
path; it does not establish whether the underlying cause is the kernel reset
sequence, GPU/card hardware, platform firmware, or power delivery.

No new GPU workload or deliberate reset was run for this investigation. Engine
changes prototyped during recovery were reverted. This report concerns the system
stop, independently of what caused the initial compute job to time out.

## Timeline

Times below are October 7, local Pacific time. Netconsole times are receiver wall
times; bracketed kernel times are the sending host's uptime. Files include both
hosts, so `host disconnected` messages from 192.168.0.83 are incite's observations
of invite disappearing. Invite sends as 192.168.0.100 in the night capture and
169.254.153.67 in the morning capture.

| Incident | Captured events |
| --- | --- |
| Overnight, boot `2298c7801b75431c9ed8c16c9695d078` | 02:27:48 vblank wait warning on GPU `06:00.0`; 02:27:49 compute ring timeout and reset begins; 02:27:59 PCI CONFIG reset, VRAM lost, recovery reports success; 02:27:59–02:28:00 incite observes all three Thunderbolt peers disconnect. |
| Morning, boot `6e9eee7131b849d1bb45517b39ec0c3f` | 08:59:50 both GPUs (`02:00.0`, `06:00.0`) time out; 08:59:51 both undergo PCI CONFIG reset, lose VRAM, and report successful recovery; 09:01:13 root port `00:03.0` reports uncorrectable non-fatal AER `CmpltTO`, and `snd_hda_intel 06:00.1` cannot transition D3hot to D0 because the device is inaccessible; immediately afterward all three Thunderbolt peers disconnect. |

The disk journal ends at 02:27:51 overnight and 09:00:37 in the morning. External
netconsole therefore preserves important events absent from the disk journal,
including the morning PCIe error. A stop can prevent both subsequent journal
writes and further network messages; absence of a final message is not evidence
that nothing happened.

The morning compute timeouts occurred while the earlier LAN run was being
cancelled. The subsequent Thunderbolt verification began after those resets.
That verification is not a clean independent reproduction from a healthy boot.

The user reported the power LED definitely off after the morning failure; the
fan may have remained on, but that observation is uncertain. The user manually
powered the machine on. This does not establish complete loss of electrical
power or distinguish a firmware stop from a hard platform hang.

## What the evidence supports

- Current PCI topology maps `00:03.0` to D700 graphics `06:00.0` and its HDMI audio
  function `06:00.1`. Thunderbolt and Ethernet are on other branches. The audio
  error corroborates loss of access to this GPU branch, rather than merely a
  userspace inference failure. It may be a consequence, not the initiating fault.
- Both captures contain GPU recovery immediately before eventual platform loss.
  This makes the GPU/PCIe recovery path the strongest lead. Correlation alone
  cannot distinguish a software recovery bug from failing hardware or power.
- No captured panic, critical thermal shutdown, OOM kill, or machine-check error
  explains either system stop. Morning GPU temperature samples immediately
  before the reset were about 54/61 C; they do not support GPU die overheating
  then. They cannot exclude unsampled VRM, PSU, or other thermal/electrical faults.
- Post-reboot SMC `MSSD` is signed **-108** (raw `94`, type `si8`). Its specific
  meaning is not established by the cited sources, and it has no timestamp tying
  it uniquely to either incident. It is not a justified RAM diagnosis.

The message `device wedged, but no recovery needed` is emitted in the successful
recovery path in this exact kernel. It is a DRM telemetry notification, not proof
that the device remained wedged. See the saved source excerpt and
[Linux DRM documentation](https://www.kernel.org/doc/html/next/gpu/drm-uapi.html#no-recovery).

## Missing evidence and limits

Current pstore is empty. Archived pstore is from an earlier October 6 Thunderbolt
oops, not either of these incidents. GPU device coredumps were announced but were
not persisted before reboot. There is no whole-system crash dump for either stop.
The kernel's panic/lockup settings are enabled, but the Intel hardware watchdog
reports BIOS `NO_REBOOT`; these settings do not guarantee capture or recovery of
a platform or power failure. Wtmp's `crash` labels establish unclean boots only.

The next useful system experiment would capture GPU coredumps and SMC power,
temperature, and shutdown-state values externally across a failure, then isolate
the GPU reset path and audio runtime power management one variable at a time.
That experiment could again stop invite and has not been performed. No reset
method, audio power setting, PCIe bridge state, or SMC control was changed here.

## Preserved evidence

- `night-netconsole.log`, `morning-netconsole.log`: copied external receiver
  captures from `/Users/fairfieldt/dev/claude-work/trashcan-kbuild/netconsole-incite.log`.
- `night-kernel-journal.log`, `morning-kernel-journal.log`: prior-boot kernel journals.
- `journal-endings.log`: system journal endings, including disk-capture cutoffs.
- `pci-current.txt`: post-reboot PCI topology and device details.
- `boot-recovery.txt`: boot history, panic settings, and pstore inventory.
- `kernel-reset-path.txt`: excerpts from the installed 7.2.9-p2p source's SI reset
  and recovery path; PCI CONFIG reset writes the GPU's config offset `0x7c`.
- `smc.json`: read-only SMC diagnostic/power values and key names. The sysfs
  read-selection index was restored; no SMC control key was written. Power values
  describe post-reboot idle conditions, not conditions at the failure.

[FreeBSD's applesmc implementation](https://cgit.freebsd.org/src/commit/?id=4fb3fae62393f02b21e499b2c0cbb28b5a7089ac)
identifies MSSD as the last shutdown cause but does not map -108.
[VirtualSMC's reverse-engineered key reference](https://raw.githubusercontent.com/acidanthera/VirtualSMC/master/Docs/SMCKeys.txt)
also labels it UNKNOWN_108; its speculative comment is not a confirmed cause.
