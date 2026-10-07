# Thunderbolt 2 between two Mac Pros: incite and invite (2026-10-06)

Both on `7.2.9-p2p` (package `7.2.9-2p2p`), default CPU settings (schedutil, turbo), `thunderbolt_net` at its
default 128 us throttle. Three cables, each joining the same controller on both machines:

| Link | incite | invite | Subnet | Lanes |
|---|---|---|---|---|
| 0 | `0-3` (Bus 0, bottom row) | `0-3` | 10.99.0.0/30 | 2 |
| 1 | `1-3` (Bus 1, top left) | `1-3` | 10.99.1.0/30 | 2 (1 until replugged, see below) |
| 2 | `2-1` (Bus 2, right column) | `2-1` | 10.99.2.0/30 | 2 |

incite is `.1`, invite `.2` on each subnet. These measurements used hand-set addresses; `tblink.service`
(`nodes/macpro/README.md`) now assigns per-link addresses automatically (10.99.18.1/.2, .17/.18, .33/.34).

## Throughput and latency (iperf3 TCP, 8-10 s)

| | incite -> invite | invite -> incite | ping avg |
|---|---|---|---|
| Link 0 alone | 8.59 Gb/s | 9.29 Gb/s | 0.18 ms |
| Link 1 alone | 9.03 Gb/s | 9.34 Gb/s | 0.18 ms |
| Link 2 alone | 9.06 Gb/s | 9.22 Gb/s | 0.18 ms |
| All three at once | 8.92 + 8.02 + 8.99 = **25.9 Gb/s** | | |
| All three, both directions | 5.2-5.7 Gb/s per stream, **32.6 Gb/s** total | | |

The three links scale almost linearly: each controller has its own PCIe Gen2 x4 uplink and the shared PEX 8723 Gen3
x8 isn't a limit at this rate. Both directions on one link give about 11 Gb/s combined, as on the single-cable
test earlier the same day; not investigated (default CPU settings).

sockperf UDP ping-pong, 64 B, link 1 while it was single-lane: 63.8 us median, 68.1 us p99.

## Copying files

| Method (link 1) | Rate |
|---|---|
| Gigabit LAN, through the Mac (`tar` over ssh, `nodes/macpro/provision.sh models`) | 112 MB/s |
| `tar | nc` (OpenBSD netcat), qwen3.8-27b, 18.8 GB | 399 MB/s (47.2 s) |
| `nc` from `/dev/zero`, to `/dev/null` | 443 MB/s |
| `socat -b 1048576` from `/dev/zero`, to `/dev/null` | 665 MB/s |
| `tar | socat -b 1048576`, qwen3.5-9b, 7.9 GB, to invite's disk | **505 MB/s** (15.7 s) |

`nc` copies in small chunks and caps a single stream near 440 MB/s; `socat` with a 1 MiB buffer does better. Real
copies end up near invite's buffered disk write rate (682 MB/s, 8 GiB `dd` with `fdatasync`; reading the source
with `tar` runs at 1.0 GB/s from cache), so splitting a copy across links wouldn't help while the destination is an
Apple SSD.

Receiving end, then sending end:

```bash
socat -b 1048576 -u TCP-LISTEN:5102,bind=10.99.1.2,reuseaddr STDOUT | tar -C ~/models -xf -
tar -C ~/models -cf - qwen3.5-9b | socat -b 1048576 -u STDIN TCP:10.99.1.2:5102
```

## Bring-up problems

- **Stale peer after the other Mac reboots.** While invite rebooted, its firmware answered on the link
  (`Apple, Inc. Macintosh`, vendor `0xa27`), and incite kept that peer after invite's Linux took over because the
  link never dropped: `failed read XDomain properties`, no network service. Reloading `thunderbolt` on incite, then on
  invite, fixed it (incite alone retried `failed request link state change`).
- **One lane after the reloads.** Link 1 then came up with one lane (`error waiting for link width to become 14`);
  unplugging and replugging it on both ends gave two lanes. The other two links, plugged in with Linux running on
  both machines, bonded two lanes at once.

## Peer reboot with tblink (19:13-19:21)

invite was rebooted with all three cables attached. incite kept the firmware peer on every link. The first tblink
version reset each of incite's controllers after 90 s (driver unbind/bind); on bus 1 the rebind failed with
`sysfs: cannot create duplicate filename '/bus/thunderbolt/devices/1-3'` and `failed to add XDomain device`, and
about a minute later invite reset abruptly (its journal stops at 93 s, nothing on netconsole). tblink no longer
resets controllers. Reloading the modules on incite then invite brought back only link 1, on one lane (and another
duplicate-device WARN for 0-3). Unplugging and replugging only incite's end of each cable restored all three links
on two lanes, re-addressed by tblink within seconds, including one cable moved to bus 1's other port.
