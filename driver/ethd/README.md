# ethd: a userspace Ethernet driver for the Mac Pro 6,1

`ethd` owns one of the Mac Pro's two Broadcom BCM57762 gigabit ports (Linux's `tg3` family, chip revision
57766000) through VFIO and gives the host a TAP interface on it. Like `gpud` and `tbd`, it runs as an ordinary
process on `gpud-platform` (PCI config, MMIO, DMA through the IOMMU, MSI), so another OS can host it by replacing
the platform backend and the TAP.

On incite the ports are 0b:00.0 (enp11s0, the management port: SSH, default route, netconsole) and 0c:00.0
(enp12s0); each is alone in its IOMMU group. ethd is meant for 0c:00.0.

## Status

Measured on incite against its own enp11s0 through the LAN switch (`results/driver/ethd-20261102/`):

- Bring-up: chip reset, rings, MAC and PHY as tg3 does them; link at 1000 Mb/s full duplex with flow control in
  about 3.5 s. Every register ethd writes ends up with tg3's value, except where tg3 uses MSI-X and RSS.
- Self-test echo: 20000 frames of 1514 B and 20000 of 60 B, all intact, interrupt-driven (one MSI, tagged status).
- TAP: 20000 frames through the host's stack and back, all intact.
- A SIGKILL mid-traffic leaves nothing behind: the next run comes up and passes.
- `vfio.sh give` returns the port to tg3, which works as before.

The echo rates (49k frames/s at 1514 B, 86k at 60 B) are the Python peer's limit, not the driver's. Line rate
hasn't been measured.

Not supported: checksum or segmentation offload, jumbo frames, more than one queue, VLAN offload, the 57766's
optional EEE patch firmware (EEE stays off, as under tg3 here), ASF/IPMI firmware (ethd refuses a chip with it
enabled), Wake-on-LAN, PTP. Link changes are followed (MAC attention and a once-a-second PHY check) but have only
been tested in simulation; no cable was pulled.

## Why Rust from tg3, not tg3.c

`gpud` imports Linux's amdgpu files behind a shim; `tg3.c` is a poor fit for that. It is 18,000 lines written
against the network core (`net_device`, NAPI, `sk_buff` with fragments and GSO, ethtool, PTP, hwmon), which the
shim doesn't have and would have to grow, and most of it is workarounds for chips other than this one. ethd
follows tg3's sequences for the 57766 alone, in the configuration tg3 uses with a single MSI, and cites the tg3
function each step comes from. As with tbd, Linux 7.3 is the reference and the oracle: the tests check the
register values against a dump from tg3 on the same port.

## Layout

| File | What | Linux reference (7.3, drivers/net/ethernet/broadcom/) |
|---|---|---|
| `regs.rs` | Registers, bits and tg3's defaults for this chip | `tg3.h` |
| `hw.rs` | Register, config space and NIC SRAM access over any gpud-platform `PciDevice` | `tg3_read_mem`, `tg3_write_mem` |
| `phy.rs` | MII access, PHY reset, link result, autonegotiation, pause resolution | `__tg3_readphy`, `tg3_phy_reset`, `tg3_setup_copper_phy` |
| `nic.rs` | Identification, bring-up, send/receive rings, interrupt service, halt | `tg3_get_invariants`, `tg3_chip_reset`, `tg3_reset_hw`, `tg3_rx`, `tg3_tx`, `tg3_halt` |
| `sim.rs`, `tests.rs` | A simulated chip that follows the programmed rings, and the tests against it | |
| `tap.rs` | The TAP device (tbd's) | |
| `vfio.sh` | Hands a port to vfio-pci and back, with guards | |
| `peer.py` | Bounded test traffic: an echo peer, and a sender for the TAP | |

Frames are copied: each ring entry owns a 2 KiB buffer for the driver's lifetime, so the chip only reaches memory
allocated at start, and dropping the driver halts the chip (blocks stopped, core reset) before that memory is
freed.

## Running

The IOMMU must be on (`intel_iommu=on iommu=pt vfio_pci.disable_idle_d3=1`; the Mac Pro has no interrupt
remapping, so `allow_unsafe_interrupts` too, see `../docs/vfio.md`). `vfio.sh take` refuses a port whose address
isn't the one given, whose interface has an IPv4 or global IPv6 address, a route or netconsole, or that shares
its IOMMU group; ethd itself refuses a port whose station address isn't `--expect-mac`.

```bash
sudo ./vfio.sh take 0000:0c:00.0 00:3e:e1:ce:a4:17
sudo ./ethd probe 0000:0c:00.0                     # chip, station address, PHY and link; no reset
sudo ./ethd regs 0000:0c:00.0                      # the registers tg3 shows for ethtool -d
# bounded self-test against an echoing peer (here incite's enp11s0)
sudo ./peer.py echo enp11s0 00:3e:e1:ce:a4:17 --count 1000 &
sudo ./ethd selftest 0000:0c:00.0 --expect-mac 00:3e:e1:ce:a4:17 --peer 00:3e:e1:ce:a4:16 --count 1000
# the driver, on a TAP (an existing persistent TAP is reused as configured)
sudo ./ethd run 0000:0c:00.0 --expect-mac 00:3e:e1:ce:a4:17 --tap ethd0
sudo ./vfio.sh give 0000:0c:00.0                   # back to tg3
```

vfio-pci bus-resets the port (root port 00:1c.1, which has nothing else below it) when ethd opens and closes it,
so every run starts from the chip's reset state and the boot code; the PHY comes up again after about 3 s.

`cargo test` runs the tests anywhere; the binary builds for Linux (`cargo build --release --target
x86_64-unknown-linux-gnu -p ethd` in the kbuild VM).
