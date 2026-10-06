# Thunderbolt 2 controller-to-controller loopback on incite (2026-10-06)

One Thunderbolt 2 cable between two ports of the same Mac Pro 6,1, tested with Linux
`thunderbolt_net` (kernel `6.12.111-p2p`, software connection manager). One end
(`thunderbolt1`) is moved into network namespace `tbB` so traffic crosses the cable
instead of loopback.

Files: `run.sh` (first MTU survey, `out-cross/`), `out-veth/` (software-only baseline),
`bench.sh` + `summarize.py` (one configuration, `runs/`), `sweep-throttle.sh` and
`nhi-int-throttle-param.patch` (interrupt throttle sweep, `runs/d-throttle-*`).

## Port to controller mapping

Apple KB HT5918 (archived) diagram, facing the rear panel, matched against Linux:

| | Left | Right |
|---|---|---|
| Top | Port 1, Bus 1 | Port 2, Bus 2 |
| Middle | Port 3, Bus 1 | Port 4, Bus 2 |
| Bottom | Port 5, Bus 0 | Port 6, Bus 0 (shared with HDMI) |

| Apple bus | Linux | NHI | PCIe switch port | Checked |
|---|---|---|---|---|
| Bus 0 | domain0 | 16:00.0 | 11:02.0 | bottom-left |
| Bus 1 | domain1 | 5d:00.0 | 11:09.0 | top-left, middle-left |
| Bus 2 | domain2 (inferred) | a4:00.0 | 11:0a.0 | not yet |

Top-left to middle-left loops a single controller: both XDomain peers appear in domain1
with domain1's own UUID.

## Link and upper bounds

- Cross-controller link (top-left to bottom-left): 2 lanes x 10 Gb/s bonded each way,
  20 Gb/s per direction on the cable.
- Same-controller loop (top-left to middle-left): 1 lane, 10 Gb/s.
- Each controller's PCIe uplink: Gen2 x4 (5 GT/s, 8b/10b) = 16 Gb/s per direction
  before TLP overhead.
- All three controllers share a PLX PEX 8723 with a Gen3 x8 uplink to root port 00:01.0
  (~63 Gb/s); not a bound for one pair.

## First survey (`out-cross/`, schedutil, no pinning, stock module)

TCP, iperf3 3.18, 15 s each. Gb/s at receiver; both ends run on incite, so CPU is shared.

| MTU | domain0 -> 1 | domain1 -> 0 | 4 streams | bidir (each way) |
|---|---|---|---|---|
| 1500 | **8.11** | 6.89 | 7.49 | 6.33 + 5.93 |
| 9000 | 7.71 | 6.69 | 7.06 | 5.77 + 5.41 |
| 65522 | 6.16 | 6.28 | 7.19 | 5.29 + 4.33 |

Larger MTU did not help. sockperf UDP ping-pong RTT was 127.6-128.0 us median at every
MTU and message size. Ping over a veth pair between namespaces (software only) averages
16 us.

## Clocks and pinning (`runs/a-c`, MTU 1500)

`bench.sh`: ping 2000 x 2 ms, sockperf UDP ping-pong 64 B for 10 s, iperf3 TCP 10 s
(forward and reverse x3, bidir, 4 streams). Pinned layout: domain0 IRQs on CPU 4, client
on CPU 6; domain1 IRQs on CPU 8, server (namespace) on CPU 10. CPU 2k and 2k+1 are HT
siblings.

| config | sockperf p50 / p99 us | ping avg / min us | fwd Gb/s | rev Gb/s | bidir Gb/s | 4 streams Gb/s | fwd retr |
|---|---|---|---|---|---|---|---|
| a: schedutil, turbo | 127.7 / 146.7 | 131 / 46 | 8.19 (6.47-8.35) | 6.97 | 6.05 + 5.88 | 7.67 | 2969, 1280, 1636 |
| b: performance, no turbo (2.7 GHz) | 127.8 / 131.5 | 81 / 50 | 8.76 (8.58-8.77) | 8.78 | 7.46 + 7.48 | 8.77 | 0, 50, 164 |
| c: b + pinned | 127.8 / 131.1 | 70 / 34 | 8.78 | 8.76 | 6.86 + 7.61 | 8.77 | 0, 0, 0 |

Fixed 2.7 GHz clocks (all 24 CPUs read 2700 MHz mid-run) remove the throughput variance
and almost all retransmits. Pinning removes the rest but doesn't raise throughput. In
configuration a, one CPU sat at 95% busy during the forward run.

## NHI interrupt throttle sweep (`runs/d-throttle-*`)

`thunderbolt.ko` rebuilt from `~/kernel/linux-source-6.12` with
`nhi-int-throttle-param.patch`, which turns the 128 us constant in
`nhi_enable_int_throttling()` into the `int_throttle_ns` module parameter. Loaded with
`insmod`, never installed. Configuration c (performance, no turbo, pinned) for every row.

| int_throttle_ns | sockperf p50 / p99 us | ping avg / min us | fwd Gb/s | rev Gb/s | bidir Gb/s | 4 streams Gb/s | IRQ CPUs busy (fwd) |
|---|---|---|---|---|---|---|---|
| 128000 (stock) | 127.8 / 132.4 | 76 / 45 | 8.78 | 8.75 | 6.97 + 7.59 | 8.78 | 49%, 73% |
| 32000 | 44.3 / 51.1 | 76 / 45 | 8.76 | 8.73 | 6.79 + 7.60 | 8.76 | 61%, 90% |
| 8000 | 44.3 / 51.1 | 73 / 44 | 8.75 | 8.72 | 5.13 + 7.86 | 8.76 | 73%, 98% |
| 2000 | 44.2 / 51.0 | 76 / 47 | 8.69 | 8.66 | 4.28 + 6.79 | 8.58 | 83%, 94% |
| 512 | 44.4 / 50.9 | 76 / 70 | 8.68 | 8.66 | 4.18 + 6.76 | 8.58 | 83%, 95% |
| 256 | 44.6 / 51.4 | 76 / 49 | 8.68 | 8.66 | 4.22 + 6.79 | 8.58 | 83%, 95% |
| 0 | 44.1 / 50.9 | 72 / 50 | 8.68 | 8.66 | 4.12 + 6.90 | 8.58 | 83%, 94% |

- The throttle sets back-to-back latency: 127.8 us round trip at the stock 128 us,
  44 us at 32 us or less. Below 32 us it no longer matters; something else sets the
  44 us floor.
- Ping at 2 ms intervals doesn't change (70-76 us avg). The throttle is a minimum gap
  between interrupts, so isolated packets aren't delayed; ping's floor comes from idle
  wakeups instead.
- Throughput doesn't depend on the throttle. Below 8 us, single-stream drops by 1% and
  bidirectional drops sharply while the IRQ CPUs climb to 83-95% busy.
- 32 us gives the low latency without losing throughput.

## Open questions

- The ~8.77 Gb/s ceiling is not CPU (no CPU above 73% at stock throttle) and not the
  throttle. It is close to what one 10 Gb/s lane carries after Thunderbolt and TCP/IP
  overhead, although sysfs reports both lanes bonded. Unverified.
- Bus 2 (right column) mapping.
- The 44 us latency floor versus 16 us over veth.
