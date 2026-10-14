# Linux 7.3 for the Mac Pros (2026-10-14)

7.3 brings USB4STREAM busy polling (`thunderbolt: stream: Add support for busy polling`, e120d14d03b3, and
`thunderbolt: Make interrupt optional for rings`, f5cb175ed728): a stream can poll its rings instead of waiting for
interrupts, which should fix the lone frame that waits for 0-3 more frames before delivery. This directory has the
first 7.3 build (v7.3-rc7), the status of our patches on it, and the checks against 7.2.9-p2p.

## Build

`v7.3-rc7` (commit 24a19b953e2a, 2026-10-12; the latest mainline tag that day, with 7.2.10 the latest stable), from
kernel.org's snapshot tarball (sha256 `f789fc0d…2523`, its pax header names that commit). Kernel release
`7.3.0-rc7-p2p`, packages `7.3.0~rc7-1p2p` (the tilde sorts it before a later `7.3.0-1p2p`). Built with
`REV=1 JOBS=24 kernel/cross-build-7.3.sh` in the kbuild VM in 5 min 19 s. `build/` has the start and end of the
build log and the packages' checksums.

For 7.3 final: add its sha256 to the table in `kernel/build-7.3.sh` and run `VER=7.3 REV=1 kernel/cross-build-7.3.sh`.
The patches and the config should carry over unchanged.

## Patches

| 7.2 patch | On 7.3-rc7 | `kernel/patches-7.3/` |
|---|---|---|
| 0001 amdgpu VFCT exact match | Still needed: 7.3's VFCT lookup still takes the first image whose device and slot match (only the ATRM path changed) | 0001, unchanged |
| 0002 SI `GPIO_DC` only on mobile parts | Upstream: cd883d0a3937 is in v7.3-rc7, as is its companion 08c9100f6529 ("Fix updating clock limits on AC/DC") | dropped |
| 0003 P2PDMA allowlist, Ivy Bridge-EP | Still needed: no `8086:0e00` (7.3 added only Nvidia entries) | 0003, unchanged (applies at offset -1) |
| 0004 `thunderbolt_net` `throttling` parameter | Still needed: 7.3 still uses the fixed `TBNET_THROTTLING` (128 us) | 0004, unchanged |
| 0005 USB4STREAM without Tx E2E | Still needed: 7.3's `tbstream_dev_start()` puts `RING_FLAG_E2E` in the flags both rings get, next to busy polling's `RING_FLAG_NO_INTERRUPT` | 0005, rebased |

0005 is now the upstream draft `kernel/upstream/usb4stream-falcon-ridge/series/0001` rebased onto 7.3: the Tx ring gets
the shared flags without E2E and the Rx ring gets `flags | RING_FLAG_E2E`, so `NO_INTERRUPT` still reaches both rings.
The draft itself doesn't apply to 7.3, because its linux-next context also had receive polling (`start_poll`), which 7.3
doesn't. The draft's 0002 (HopID release) applies to 7.3 but isn't carried: it isn't one of our patches and no check
needs it.

Checked against the tag on GitHub's mirror: e120d14d03b3, f5cb175ed728, cd883d0a3937 and 08c9100f6529 are ancestors of
v7.3-rc7. The packaged `thunderbolt_stream.ko` has the `busy_poll` attribute and `thunderbolt_net.ko` the `throttling`
parameter.

## Config

`kernel/configs/7.2.9-p2p.config` (the Macs' `/boot/config-7.2.9-p2p`, sha256 `f8973974…fcd4`) through `make
olddefconfig`, with `build-7.3.sh` reasserting the 7.2 settings (P2PDMA, USB4STREAM, no debug info, `-p2p`). The result
is `kernel/configs/7.3.0-rc7-p2p.config`; `build/diffconfig-7.2.9-p2p-to-7.3.0-rc7-p2p.txt` lists every difference:
89 new symbols, 22 gone, 3 changed.

- New symbols: almost all new drivers, off. Defaults worth knowing: `SWIOTLB_DEFAULT_SIZE_MB=64` (the old fixed size,
  now a knob), `SYSCALL_USER_DISPATCH=y`, `USERFAULTFD_RWP=y`, `SECURITY_LANDLOCK_LOG=y`,
  `SECURITY_APPARMOR_COMPRESSED_POLICY=y`, `VDSO_DATASTORE=y`, built-in AES library modes `CRYPTO_LIB_AES_{CBC,CCM,CTR,
  ECB,GCM,XTS}` (`CRYPTO_LIB_AESGCM` renamed), and `CC_OPT_INLINE_MEMSET` (GCC 14's `-finline-stringops=memset`).
- Changed: `BLK_INLINE_ENCRYPTION` n -> y (with `_FALLBACK` and `FS_ENCRYPTION_INLINE_CRYPT`, now default y with
  fscrypt), `CRYPTO_AEAD`/`CRYPTO_AEAD2` m -> y.
- Gone: `PROC_SYSCTL` (`/proc/sys` now comes with `SYSCTL`), removed drivers and filesystems (`EFS_FS`, `VXFS_FS`,
  `IBM_ASM`, `IPWIRELESS`, `PCMCIA_XIRCOM`), `OPENVSWITCH_{GRE,VXLAN,GENEVE}`, `XEN_PVHVM*`, `DMA_FENCE_TRACE` and a few
  internal symbols.
- Unchanged, and checked by `build-7.3.sh` (it stops if one is missing): `DRM_AMDGPU=m` with `DRM_AMDGPU_SI=y` and
  `DRM_AMD_DC_SI=y`, `DRM_VGEM=m`, `NETCONSOLE=m` (`_DYNAMIC`), `USB4`, `USB4_NET`, `USB4_CONFIGFS`, `USB4_STREAM` as
  modules, `PCI_P2PDMA=y`, `DEBUG_INFO_NONE`. Also unchanged: lazy preemption with `PREEMPT_DYNAMIC`, 250 Hz, no
  simpledrm.

## Checks on inflight: 7.2.9-p2p, then 7.3.0-rc7-p2p (2026-10-14)

Under gpuq lease #98: `run.sh` on 7.2.9-p2p, `kernel-boot.sh once 7.3.0-rc7-p2p` (install, `grub-reboot`; GRUB_DEFAULT
stays on 7.2.9-p2p), reboot, `run.sh` on 7.3. `tb.sh` ran on 7.2.9 before the reboot and on 7.3 after the
coordinator's Thunderbolt reset of all six Macs. Same binaries on both kernels: `xgpu` 8e0fa782…, test-engine's
`trashcan` 09227c85…, Flash-Next `trashcan-15e5457` 9a379fa3…. The fan ran at 2350 RPM both times (`fancurve` on
7.2.9; on 7.3 set by hand, then by the fixed `fancurve`, see below). Raw outputs: `inflight-7.2.9-p2p/` and
`inflight-7.3.0-rc7-p2p/`; `sbsum.py` and `p2psum.py` make the table rows.

| | 7.2.9-p2p | 7.3.0-rc7-p2p |
|---|---|---|
| Boot | | one-time GRUB entry, ssh after about 70 s |
| VBIOS 02:00.0 / 06:00.0 | 113-C3861XA-028 / 113-C3861XB-028 | same |
| PCIe, both cards | 8 GT/s x16 | same |
| `xgpu p2p-verify` | PASS, export in VRAM (one run; two more were stopped to shorten the lease) | 3/3 PASS, 0 bad words, export in VRAM |
| P2P write, submit-to-fence, 1 / 8 / 32 MiB | 7.94 / 7.77 / 7.96 GB/s | 7.87 / 7.91 / 8.00 GB/s |
| P2P read, 32 MiB | 3.76 GB/s | 3.72 GB/s |
| Stage-bench: 128-token chunk, wall | 194.05 ms (193.79-194.33) | 194.30 ms (194.28-194.34) |
| Stage-bench: decode step, wall | 7.89 ms | 7.88 ms |
| GPU ms per chunk, stage 0 / 1 | 95.63 / 91.25 | 95.62 / 91.39 |
| GPU ms per decode step, stage 0 / 1 | 3.89 / 3.64 | 3.88 / 3.64 |
| Clocks while computing | 850/1370 on both cards | same |
| DPM `high`, then `auto` | rc 0 on both (the cards were already at `high`) | rc 0 on both: 300/150 at `auto`, 850/1370 at `high` |
| `amdgpu.lockup_timeout` | unset (2000 ms default) | unset (2000 ms default) |
| `scripts/test-engine.sh` | 58 unit tests, 6/6 validations (inflight's provisioning run on 2026-10-07, same binary; this run's was stopped to shorten the lease) | 58 unit tests, 6/6 validations; worst relative errors 9.95e-4, 9.76e-4, 9.85e-4, 9.82e-4, 9.95e-4, 9.85e-4, the same as 7.2.9's |
| netconsole | token received | token received; the whole boot log arrived too (from a new DHCP address, .153) |
| Thunderbolt links | incite and alight, 2 lanes each | same, after the reset |
| Streams from `tblink`, udev rule | `/dev/tbstream0`/`1`, root:render 0660 | same; the groups also show `busy_poll=0` |
| `thunderbolt_net throttling` | 32000 | 32000 |
| ping over each link, 100 x 10 ms, avg | 0.154 / 0.155 ms | 0.154 / 0.150 ms |

Stage-bench is `qwen4 stage-bench G-iq4 --ranges 0:4,4:8 --gpus 0,1 --context 32768 --head-rows 0:21504,21504:43008
--batch 128 --prefill 8 --decode 32 --start 64` under `latency-run.py` and `with-clocks.sh`: a warm-up, then three
runs. The two kernels are within 0.15% on every row.

dmesg (`dmesg-warn.txt`): the only WARNING on both is `apple-gmux`; both have the VCE firmware note, the `pcspkr`,
`b43` and `bpf-restrict-fs` messages, and the Thunderbolt bridge-window and device-link messages. New on 7.3:
`thunderbolt 0-3`/`1-3: failed to send properties changed notification`, once each at boot, while the links to the
7.2.9 neighbours were still stale (before the reset); 7.2.9 had 188 `failed request link state change` from the same
state over its uptime. The coordinator's two reset rounds (`dmesg-tb-end.txt`, `dmesg-warn-end.txt`) unloaded and
reloaded the Thunderbolt modules on 7.3 without a warning; the first round left both links at `error waiting for link
width to become 14`, the second brought them up on two lanes, as on 7.2.9.

## amdgpu job timeout

Still 2 s. `amdgpu_device_get_job_timeout_settings()` is identical in 7.2.9 and 7.3-rc7: every queue defaults to
2000 ms unless `amdgpu.lockup_timeout` is set, and the Macs don't set it.

## 7.3 sysfs changes

Compared on inflight (7.3) and alight (7.2.9):

| Path | 7.2.9 | 7.3 | Our users |
|---|---|---|---|
| applesmc fan | `/sys/devices/platform/applesmc.768/`: `fan1_input`, `fan1_min`, `fan1_max`, `fan1_manual` (1/0), `fan1_output` | moved to hwmon attributes in `applesmc.768/hwmon/hwmon0/`: `fan1_input`, `fan1_min`, `fan1_max`, `pwm1_enable` (1 manual, 2 automatic), `fan1_target` | `fancurve.sh` and its unit, `system-crash-capture.py` (`fan_rpm`): fixed for both layouts |
| applesmc temperatures | `applesmc.768/temp*_input` | `applesmc.768/hwmon/hwmon0/temp*_input` | none (the telemetry reads SMC keys) |
| applesmc keys | `applesmc.768/key_at_index*`, `key_count` | unchanged | `system-crash-capture.py` |
| amdgpu hwmon | `freq1_input`, `freq2_input`, `temp1_input`, `pwm1*`, `fan1*`; no `power1_average` | unchanged | `clockmon.sh`, telemetry |
| `power_dpm_force_performance_level` | `high`/`auto` | unchanged: `echo high` raises both cards from 300/150 to 850/1370 MHz | engine `clocks.rs`, `dpm-run.sh`, `q4run.py` |
| debugfs `dri/` | `0000:02:00.0`, `0000:06:00.0`, with `0`, `1`, `128`, `129` linking to them | unchanged | |
| `amdgpu_pm_info` | `power level N    sclk: … mclk: … vddc: … vddci: … pcie gen: 3` | unchanged | thermal-latch checks |
| Thunderbolt sysfs | `unique_id`, `vendor`, `maxhopid`, `rx_lanes`, services `0-3.0`/`0-3.1` with `key` | unchanged | `tblink`, `tbstream-up.sh` |
| Stream configfs group | `index`, `in_hopid`, `out_hopid`, `ring_size`, `throttling` | adds `busy_poll` (0 by default) | `tblink`, `tb-reset.sh` |

Without the fix, 7.3 skips `fancurve.service` (its condition was `fan1_manual`) and the SMC runs the fan at about
790 RPM instead of 2350. On inflight the fan was set by hand (`pwm1_enable=1`, `fan1_target=2350`) before the GPU
checks; at 12:36 the fixed `fancurve.sh`, its unit and `system-crash-capture.py` were installed and took over (`fan
target 2350 rpm (live SMC maximum 1900)`). `systemctl stop fancurve` set `pwm1_enable` back to 2 and a start set it to
1 again. inflight's telemetry had `fan_rpm` null from boot until then and 2348-2349 since. `scripts/test-fancurve.py`
runs every case on both layouts (30 pass; 13 of the hwmon cases fail on the old script).

## Other 7.3 changes on paths we use

- Thunderbolt: busy polling and optional ring interrupts; control-request cancellation reworked (`ctl.c`); HopID
  allocations hold a reference on the router; XDomain property responses are length-checked; on shutdown or reboot
  (new `nhi_pci_shutdown()`), `tb_stop()` resets downstream ports that lead to non-USB4 device routers. Links between
  Macs are XDomain connections, not device routers, so that reset doesn't touch them.
- amdgpu SI DPM: 08c9100f6529 takes the AC clock limits from all power states, not only the performance ones.
- netconsole: reworked internally (per-target skb pools, addresses moved out of netpoll); same module parameter.

Not checked here: data through the streams and busy polling itself (the USB4STREAM agent's tests).

## Files

- `kernel-boot.sh once|default KREL`: install a kernel from `~/kernel/7.3/debs` and boot it once, or make it GRUB's
  default.
- `run.sh [TAG]`: the GPU-side checks, under the Mac's lease; `tb.sh [TAG]`: Thunderbolt and netconsole.
- `sbsum.py`, `p2psum.py`: summaries of `run.sh`'s stage-bench and xgpu outputs.
- `inflight-7.2.9-p2p/`, `inflight-7.3.0-rc7-p2p/`: outputs (`clocks/` from `clockmon.sh`; full dmesg not kept, it
  has MAC addresses). `test-engine-provision-20261007.log` is inflight's 7.2.9 test-engine run from provisioning.
- `build/`: build log excerpt, package checksums, config differences.
