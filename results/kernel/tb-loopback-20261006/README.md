# Thunderbolt 2 controller-to-controller loopback on incite (2026-10-06)

One Thunderbolt 2 cable between two ports of the same Mac Pro 6,1, tested with Linux
`thunderbolt_net` (kernel `6.12.111-p2p`, software connection manager). One end
(`thunderbolt1`) is moved into network namespace `tbB` so traffic crosses the cable
instead of loopback. Script: `run.sh`. Raw output: `out-cross/`, `out-veth/`.

## Port to controller mapping (measured)

| Rear port | Controller | NHI | PCIe switch port |
|---|---|---|---|
| top-left | domain1 | 5d:00.0 | 11:09.0 |
| middle-left | domain1 | 5d:00.0 | 11:09.0 |
| bottom-left | domain0 | 16:00.0 | 11:02.0 |
| right column | one domain0 port, two domain2 ports (a4:00.0, 11:0a.0); not yet identified | | |

Rows are not controller pairs. Top-left to middle-left loops a single controller (both
XDomain peers appear in domain1 with domain1's own UUID).

## Link and upper bounds

- Cross-controller link (top-left to bottom-left): 2 lanes x 10 Gb/s bonded each way,
  20 Gb/s per direction on the cable.
- Same-controller loop (top-left to middle-left): 1 lane, 10 Gb/s.
- Each controller's PCIe uplink: Gen2 x4 (5 GT/s, 8b/10b) = 16 Gb/s per direction
  before TLP overhead. This is the tighter bound.
- All three controllers share a PLX PEX 8723 with a Gen3 x8 uplink to root port 00:01.0
  (~63 Gb/s); not a bound for one pair.
- CPU governor `schedutil`; clocks varied 1.2-2.6 GHz during env capture.

## Results (cross-controller, domain0 <-> domain1)

TCP, iperf3 3.18, 15 s each. Gb/s at receiver; both ends run on incite, so CPU is shared.

| MTU | domain0 -> 1 | domain1 -> 0 | 4 streams | bidir (each way) |
|---|---|---|---|---|
| 1500 | **8.11** | 6.89 | 7.49 | 6.33 + 5.93 |
| 9000 | 7.71 | 6.69 | 7.06 | 5.77 + 5.41 |
| 65522 | 6.16 | 6.28 | 7.19 | 5.29 + 4.33 |

Retransmits are high (2k-51k per run); dmesg warns `thunderbolt1: Driver has suspect GRO
implementation`. Larger MTU did not help.

Latency (round trip, idle link):

| Test | median / avg | p99 |
|---|---|---|
| sockperf UDP ping-pong 64 B, MTU 1500 | 127.6 / 127.9 us | 143 us |
| sockperf UDP ping-pong 1024 B, MTU 1500 | 127.7 / 127.9 us | 139 us |
| ping 56 B, MTU 1500 (2000 x, 2 ms interval) | avg 161 us (min 80) | max 187 us |
| ping 56 B over veth between namespaces (software only) | avg 16 us | max 44 us |

sockperf RTT is the same within 1 us across message size and MTU. The ~128 us value
matches the NHI interrupt throttling interval the Linux driver programs
(`nhi_enable_int_throttling`, 128 us), so interrupt moderation, not the link, likely sets
latency. Not yet confirmed by changing the throttle.

Rough same-controller number (top-left to middle-left, 1 lane, single 8 s run, MTU 1500):
6.79 Gb/s.

## Not done

- Right-column port mapping.
- Throttle-interval experiment (needs a patched `thunderbolt.ko`).
- Pinning IRQs / iperf processes, or fixed CPU clocks.
