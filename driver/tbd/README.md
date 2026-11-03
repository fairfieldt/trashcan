# tbd: a userspace Thunderbolt 2 driver for the Mac Pro 6,1

`tbd` owns one of the Mac Pro's three Falcon Ridge controllers through VFIO and talks to the hosts on its cables:
ThunderboltIP networking (a TAP device) and USB4STREAM streams, both compatible with Linux's `thunderbolt_net`
and `thunderbolt_stream` on the other end. Like `gpud`, it runs as an ordinary process on `gpud-platform` (PCI
config, MMIO, DMA through the IOMMU, MSI), so another OS can host it by replacing the platform backend.

Only host-to-host links over Falcon Ridge are supported. No PCIe, DisplayPort or USB tunnels, no devices, no
firmware upgrades.

## Status

Measured on two Mac Pros joined by one cable (both lanes): `tbd` on one end, Linux 7.3's driver on the other.

- It finds the other host and bonds the two lanes, whichever side has the lower UUID. Linux lists it as a host
  with `network` and `stream` services.
- ThunderboltIP: Linux logs in and the link carries traffic both ways (ping, TCP).
- USB4STREAM: an echo through the kernel's `/dev/tbstreamN` gets every byte back intact.
- On exit it logs out, undoes bonding and takes the link down and up once, so the other host starts afresh.

Compared with Linux on both ends, on the same cable and with the same tools (two runs or more each; "busy" is
`tbd run --busy-poll`):

| | Linux ↔ Linux | tbd ↔ Linux | tbd busy ↔ Linux |
|---|---|---|---|
| TCP, tbd sending (iperf3) | 8.9 Gb/s | 8.3-9.1 Gb/s | 8.8 Gb/s |
| TCP, tbd receiving | 8.3 Gb/s | 7.3-8.0 Gb/s | 6.9-7.3 Gb/s |
| ping | 0.15 ms | 0.32 ms | 0.14 ms |
| stream echo, 64 KiB writes, 512 KiB window (`tbd bench`) | 331-453 MB/s | 292-294 MB/s | 321-385 MB/s |
| stream round trip, 64 B, median (`tbd ping`) | 45 µs | 74 µs | 42 µs |

TCP goes through the TAP device as 64 KiB TSO packets, as Linux's driver gets them from its stack. The checksums are
filled in in software, as Linux does. Each ring has an MSI-X vector of its own. Without busy polling a stream round
trip still wakes three threads on tbd's side (its TX thread for the client's message, its RX thread for the
interrupt, then the client), where Linux wakes one. Putting the stream rings in the client's own process would
remove the other two.

## Layout

| File | What | Linux reference (7.3) |
|---|---|---|
| `nhi.rs` | NHI registers, descriptor rings, interrupts (an MSI-X vector per ring, or one MSI; a ring's interrupt is unmasked only while its thread sleeps) | `nhi.c`, `nhi_regs.h`, `pci.c` |
| `ctl.rs` | Control channel: frames, CRC-32C, config reads/writes, matching replies to requests | `ctl.c`, `tb_msgs.h` |
| `switch.rs` | Host router: header, capabilities, ports, lane control, link controller UUID, hop entries | `switch.c`, `cap.c`, `lc.c`, `path.c`, `tb_regs.h` |
| `property.rs` | XDomain property blocks | `property.c` |
| `xdomain.rs` | XDomain discovery packets | `xdomain.c` |
| `domain.rs` | Bring-up, hot plug, per-peer discovery and lane bonding, our property block, DMA paths | `tb.c`, `xdomain.c`, `tunnel.c` |
| `net.rs`, `tap.rs` | ThunderboltIP login and framing; the TAP device | `drivers/net/thunderbolt/main.c` |
| `stream.rs` | USB4STREAM: DATA/CLOSE frames, HopIDs in the `stream` service directory, a socket per stream | `stream.c` |
| `bench.rs` | `echo`, `bench`, `ping` on a tbd socket or a Linux `/dev/tbstreamN` | |

The Rust is written from the Linux code's protocol and register definitions, not translated from it. Tests check
the packet and register layouts against values captured from a running Linux driver.

## Running

The NHI must be on vfio-pci, with the IOMMU on (`intel_iommu=on iommu=pt vfio_pci.disable_idle_d3=1`; the Mac Pro
has no interrupt remapping, so `allow_unsafe_interrupts` too, see `../docs/vfio.md`). `vfio.sh take` unbinds the
kernel's Thunderbolt services on that controller first, then hands the NHI over. Unbinding the NHI under a running
`thunderbolt_net` leaves that driver's rings behind, writing to an NHI it no longer owns.

```bash
sudo ./vfio.sh take 0000:16:00.0
sudo ./tbd probe 0000:16:00.0                        # read-only: router, ports, link states, UUID, hops
sudo ./tbd run 0000:16:00.0 --stream trashcan:15:15  # TAP tbd0 plus one stream per peer
sudo ip addr add 10.0.0.2/30 dev tbd0
sudo ./tbd bench /run/tbd/PEER.trashcan             # against `tbd echo /dev/tbstreamN` on the other host
sudo ./vfio.sh give 0000:16:00.0                     # back to the kernel driver
```

Each stream is offered on two sockets (`SOCK_SEQPACKET`, group `render`), one client at a time between them; a
second client is told the stream is in use. tbd also writes `PEER.NAME.info` (ring size, HopIDs, both UUIDs) and,
for each TAP interface, `net/IFNAME` (the peer it leads to), so a program can find the stream of a link.

- `PEER.NAME`: tbd moves the frames. A message of up to 64 KiB goes out as 4 KiB frames, and consecutive received
  frames come back as one message. A CLOSE from the other host arrives as an empty message. A client that shuts
  down its writing side sends a CLOSE and keeps reading.
- `PEER.NAME.ring`: tbd hands the stream's two rings to the client (`../tbd-client`): the memory holding their
  descriptors and buffers (a memfd, mapped for the NHI), the NHI's register BAR (a duplicate of the VFIO device
  descriptor) and the RX ring's MSI-X eventfd (throttled to one per 8 µs, as Linux's streams are). The client posts
  and completes frames itself, without system calls, and sleeps in `poll()` on the eventfd. When it goes, tbd takes
  the TX ring back from the NHI's indexes, sends the CLOSE and takes the paths down. The client is trusted: the
  BAR reaches every ring.

Paths are routed only while a client is connected, as Linux routes them only while the device is open.

Falcon Ridge's TX engine sometimes stops with frames posted: the NHI's consumer index stays behind the producer
until the next doorbell, even one that writes the same index again. Nothing is lost. It showed at the end of
bursts, a few times per hundred 1 GB benches through a ring client, and does not depend on end-to-end flow control
or the descriptors' interrupt flag. The cause is not known. tbd-client writes the producer index again when frames
have been pending for 50 µs without progress, on every read, write and wait. Only the client writes that register,
because a stale value from tbd could move the index back.

On the Mac Pro nodes, `tblink` knows only the kernel's links: while tbd owns a controller, the `PEER.tbN` names of
that controller's links drop out of `/etc/hosts` (connect by address), and `tbd0` gets no address of its own (add the
one `tblink` gave the link's `thunderboltN`).

`TBD_TRACE=1` prints every control frame. `TBD_KMSG=1` copies tbd's output to the kernel log (with
`sysctl kernel.printk_devkmsg=on`), so netconsole has it if the machine goes down.

`driver/` is one Cargo workspace. `cargo test -p tbd` runs anywhere. The driver builds on Linux only.
