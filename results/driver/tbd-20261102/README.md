# tbd, first measurements (2026-11-02)

tbd at main f92fcd8c (`driver/tbd`), built with `cargo build --release --target x86_64-unknown-linux-gnu -p tbd`
in the kbuild VM. Hosts: **enlight** runs tbd on NHI 0000:16:00.0 (domain 0), booted once with `tbd-custom.cfg`
(`intel_iommu=on iommu=pt vfio_pci.disable_idle_d3=1`). **invite** runs Linux 7.3.0-rc7-p2p (patch set 9c9a5449)
with thunderbolt_net (`throttling=32000`), thunderbolt_stream and tblink, unchanged. The cable joins enlight 0-1
and invite 0-1, both lanes bonded. Both have `/etc/modprobe.d/tbd-debug.conf` (dyndbg on for the three modules).

## Linux reference (`linux-oracle/`)

Linux on both ends, captured before tbd took the NHI: `dmesg-*.txt` (full dynamic debug of a simultaneous
`tblink reload`: XDomain handshake, lane bonding, ThunderboltIP login, hop writes) and `debugfs-*.txt` (every file
under `/sys/kernel/debug/thunderbolt`: router and port registers, hop tables). tbd's unit tests use values from
here (the DMA RX hop `0x801c2802 0x017ff501` on 0:1 hop 8).

## Bring-up

- `tbd probe`: 12 paths; router 8086:156d, upstream port 5, 12 ports; host UUID read from the link controller
  fuses `00000000-0000-0008-0048-1a1dad52f062`, the same as Linux's `unique_id`; port types, HopID limits and
  credits as in Linux's log.
- `tbd run --stream trashcan:15:15`: link bounced, remote UUID ...8041-c5264ae46855, lanes bonded (enlight has
  the lower UUID and asks), properties read (`Linux invite`, services network and stream). invite logged
  `new host found, vendor=0x1d6b device=0x4` / `tbd enlight`, `we have higher UUID, other side bonds the lanes`,
  `lane bonding enabled`, ThunderboltIP login both ways and `network traffic started`; invite's tblink listed the
  link as `enlight bus 0, 2 lane(s) ... stream /dev/tbstream0`.
- Restarting tbd (SIGINT: logout, then a fresh start with a bounce) brought everything back each time.

## Numbers (same cable, same tools)

Linux both ends: enlight's NHI given back (`vfio.sh give`) and both hosts' modules reloaded together.

| | Linux ↔ Linux | tbd ↔ Linux |
|---|---|---|
| `iperf3 -t 8`, enlight → invite | 8.88 Gb/s (5089 retr) | 1.68 / 1.72 Gb/s (191 / 259 retr) |
| `iperf3 -t 8 -R`, invite → enlight | 8.33 Gb/s | 8.26 / 7.59 Gb/s (4245 retr, RX ring overflows) |
| `ping -i 0.01`, 200 | 0.153 ms avg | 0.302 / 0.357 ms |
| `tbd bench --bytes 1e9` (window 512 KiB) through `tbd echo /dev/tbstream0` on invite | 289.4 MB/s each way | 175.3 MB/s |
| `tbd ping --size 64`, 20000: min / median / p99 | 26.6 / 45.4 / 49.1 µs | 48.8 / 77.8 / 88.8 µs |
| `tbd ping --size 4096`, 5000 | 35.8 / 46.8 / 50.5 µs | 54.5 / 72.3 / 112.9 µs |

tbd's two runs: the first per-frame interrupts, the second with ring interrupts unmasked only while a thread
sleeps. Every bench byte came back intact. A plain `tbd cat` of 100 MiB without a window lost 1.5 MB at invite's
receive ring (no transmit flow control on streams, as with Linux on both ends).

## enlight reset at 17:21

`netconsole-enlight-reset.log` (from the netconsole receiver here). `vfio.sh take` unbound the NHI while
invite's link was up with Linux's thunderbolt_net running on enlight: `TX ring 2 is still active`, `RX ring 2 is
still active` and `remove_proc_entry` warnings, and enlight's `thunderbolt0` stayed with 10.99.36.2. tbd started
and ThunderboltIP came up on tbd0 (same address). During the ping/iperf that followed, enlight reset with nothing
more on netconsole. Probable cause: the leftover kernel interface sending on ring 2, whose registers belonged to
tbd's ring 2. `vfio.sh` now unbinds the domain's services first; the same tests with the same tbd binary then ran
without trouble (second tbd column).

Also seen: closing the VFIO device made vfio-pci reset the NHI (the only device on its bus, so a bus reset);
the next start worked. After tbd exited with the lanes still bonded, Linux on enlight looped on `failed request link
state change`; tbd now undoes bonding and bounces the link on exit (not yet retested).

## Later the same evening (main ae9d8643, 91c6fcf3)

`target/tbd-suite.sh`-style runs: tbd restarted with `--stream trashcan:15:15` (plus `--busy-poll` where marked),
invite running `tbd echo /dev/tbstream0` and `iperf3 -s`, all tests from enlight. Linux both ends re-measured with
64 KiB writes for the stream bench (enlight's NHI given back, tbd's exit left the link so that Linux took it back
without a module reload: two lanes, carrier, stream).

| | Linux ↔ Linux | tbd, MSI | tbd, MSI-X | tbd, busy poll |
|---|---|---|---|---|
| iperf3 enlight → invite | 8.88 Gb/s | 9.06 Gb/s (3452 retr) | 8.34 (2450) | 8.77, 4.20, 4.03 * |
| iperf3 invite → enlight | 8.33 Gb/s | 7.95 (90) | 7.26 (45) | 7.25, 6.96, 6.89 |
| ping, avg | 0.153-0.156 ms | 0.343 ms | 0.317 ms | 0.141-0.145 ms |
| `tbd bench --write 65536` | 330.9, 453.3 MB/s | 291.8 MB/s | 294.3 MB/s | 320.6, 339.4, 384.7, 342.4 MB/s |
| `tbd ping --size 64` median (min, p99) | 45.4 (26.6, 49.1) µs | 85.0 (48.3, 96.5) | 73.5 (42.8, 83.6) | 41.1-42.8 (26.6-28.1, 45.9-66.4) |
| `tbd ping --size 64 --spin` median | 33.8 µs | | | 32.0 µs |
| `tbd ping --size 4096` median | 46.8 µs | 103.9 | 64.1 | 42.9-51.5 |

\* The 4.2 and 4.0 Gb/s runs were before the checksum went from 16-bit adds to 8-byte words (ae9d8643): TSO
packets of 22 KB on average at about 40 µs each. TCP's sending side is the TAP device with TSO and checksum offload
(the kernel hands over packets of up to 64 KiB). Remaining with tbd receiving: a few RX ring overflows per 8 s
run (`RX overflow on ring 2`, 1-16 bad frames).

The second takeover with Linux's services running (`vfio.sh take`: "unbinding 0-1.0 from thunderbolt-net",
"unbinding 0-1.1 from thunderbolt_stream") went without trouble, and the tests ran on it.

At the end: tbd stopped, NHI back on `thunderbolt`, `tbd-debug.conf` removed and dynamic debug off on both hosts.
enlight stays on the one-time IOMMU boot until it reboots.
