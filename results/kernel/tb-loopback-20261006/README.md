# Thunderbolt 2 controller-to-controller loopback on incite (2026-10-06)

Thunderbolt 2 cables between ports of the same Mac Pro 6,1 on different controllers,
tested with `thunderbolt_net` and USB4STREAM (software connection manager), first on
`6.12.111-p2p` and then on `7.2.9-p2p`. For `thunderbolt_net`, the B end moves into
network namespace `tbB` so traffic crosses the cable instead of loopback. Unless noted,
the cable is top-left (`1-3`, domain1) to bottom-left (`0-3`, domain0).

Files:

| File | Purpose |
|---|---|
| `run.sh`, `out-cross/`, `out-veth/` | First MTU survey (6.12), software-only veth baseline |
| `bench.sh`, `summarize.py`, `runs/a-e` | `thunderbolt_net` latency/throughput for one configuration; `XA`/`XB` pick the cable ends |
| `sweep-throttle.sh`, `nhi-int-throttle-param.patch`, `runs/d-*` | 6.12 NHI interrupt-throttle sweep (out-of-tree `thunderbolt.ko`, never installed) |
| `tbstream-bench.c`, `stream-run.sh`, `stream-pair.sh`, `runs/f-*` | USB4STREAM ping-pong and throughput (7.2) |
| `dma-repro.sh`, `bisect-test.sh`, `stream-once.sh`, `tb-reset.sh`, `repro-*.txt`, `runs/g-*`, `runs/h-*` | Finding why USB4STREAM wedged the controllers |
| `stream-tri.sh`, `runs/rev2/` | Final kernel (`7.2.9-2p2p`): one stream, one stream per cable on three cables, link bounces |
| `locked.sh` | Runs a command under `~/trashcan-exp/.lock` with the performance governor and turbo off |

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
| Bus 2 | domain2 | a4:00.0 | 11:0a.0 | in use as `2-1`/`2-3` with three cables; positions from the diagram |

Top-left to middle-left loops a single controller: both XDomain peers appear in domain1
with domain1's own UUID. With three cables forming a triangle (`0-3`/`1-3`, `0-1`/`2-3`,
`1-1`/`2-1`) every link bonded two lanes.

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

## Linux 7.2.9: `thunderbolt_net` (`runs/e-*`)

Same configuration c (performance, no turbo, pinned). 7.2's `thunderbolt_net` sets the ring
throttle per ring; `kernel/patches-7.2/0004` makes it the `throttling` module parameter.

| config | sockperf p50 / p99 us | ping avg / min us | fwd Gb/s | rev Gb/s | bidir Gb/s | 4 streams Gb/s |
|---|---|---|---|---|---|---|
| 6.12, 128 us (c) | 127.8 / 131.1 | 70 / 34 | 8.78 | 8.76 | 6.86 + 7.61 | 8.77 |
| 7.2, 128 us (default) | 127.8 / 129.8 | 65 / 41 | **9.23** | 9.18 | 7.90 + 7.87 | 9.23 |
| 7.2, `throttling=32000` | **40.2** / 44.8 | 67 / 40 | 9.21 | 9.17 | 7.87 + 7.88 | 9.22 |

## Linux 7.2.9: USB4STREAM (`runs/f-*`)

`/dev/tbstream0` (`0-3.1/data`) to `/dev/tbstream1` (`1-3.1/data`), HopIDs 8/8, ring size
256, `kernel/patches-7.2/0005` applied (no E2E on the Tx ring, see below).
`tbstream-bench pingpong` times 20,000 round trips after 2,000 warm-up; `stream` writes
for 10 s.

| `throttling` | 64 B p50 / p99 us | 1 KiB p50 us | 4 KiB p50 us | one way Gb/s (4 KiB-1 MiB writes) |
|---|---|---|---|---|
| 8192 (default) | **25.1** / 26.2 | 28.0 | 33.3 | 9.65-9.68 |
| 32000 | 32.0 / 32.6 | 32.0 | 36.3 | 9.69-9.72 |
| 0 | 26.6 / 27.8 | 27.8 | 33.2 | 9.54-9.56 |

Two streams on the same cable (`runs/f-7.2-stream-pair`): one alone 9.55 Gb/s; two in the
same direction 4.79 + 4.79; two in opposite directions 8.23 + 8.23. The ceiling is about
9.6 Gb/s per direction per cable, one 10 Gb/s lane's worth, although sysfs reports two
bonded lanes. The `thunderbolt_net` ceilings (8.8 Gb/s on 6.12, 9.2 on 7.2) sit under it.

## Why stock USB4STREAM wedged the controllers (`runs/g-*`, `runs/h-*`)

Stock 7.2.9 `thunderbolt_stream` failed every stream (`runs/g-bisect-stock`: 0/10), and
the controller then stopped answering config reads ("timeout reading config space 0 from
0x10/0x12", "DMA RX path activation failed: -110") until reset or reboot. Reverting the
suspect path-activation change b69af182b556 didn't help (`runs/g-bisect-revert-b69af182`).

The debug trace (`runs/h-stream-once-1/dmesg.txt`) shows the sequence: USB4STREAM starts
its Tx ring with E2E flow control, the ring never gets credits, stopping it takes 0.5 s,
and the next config read times out. Upstream reverted Tx-side E2E in `thunderbolt_net` for
the same reason (1881f2efbf7f, ASMedia ASM4242; the thunderbolt-ibverbs project reports the
same on AMD Strix Halo), but `thunderbolt_stream` still sets it. With Tx-side E2E removed
(`runs/h-stream-once-rxe2e-2`, the `runs/f-*` numbers) streams work and no config read
times out.

Other findings from the same work:

- Deleting a stream's configfs directory after a failed start warns
  `ida_free ... tbstream_dev_release` (frees an ID it didn't allocate). Upstream bug.
- A far end only copies the peer's HopIDs if the peer's property update has already arrived
  when its directory is created; the scripts set both ends by hand.
- `tb-reset.sh` resets each NHI through sysfs `reset` (a bus reset of the NHI link). That
  recovered a wedged controller once; a controller that has failed repeatedly needs a
  reboot. Removing the devices or resetting the PLX downstream ports by hand crashed the
  PCI core (`pcie_update_aspm_capable` NULL dereference) and needed a power cycle.
- `thunderbolt_net` link bring-up on 7.2: 3 DMA activation failures in 240 bounces with the
  thunderbolt-ibverbs-patched modules and three cables (`repro-7.2-tbv.txt`), 0 in 240 with
  stock modules (`repro-7.2-stock.txt`), 0 in 20 per run with one cable, and 0 in 240 with three cables on the
  final kernel (below). The ibverbs patch stack is the likely cause; the final kernel doesn't carry it.

## Final kernel, three cables (`runs/rev2/`)

`7.2.9-2p2p` (patches 0001-0005, no thunderbolt-ibverbs patches), installed modules, all three cables
(`0-3`/`1-3`, `0-1`/`2-3`, `1-1`/`2-1`), clocks locked.

- One stream on `0-3`/`1-3` at the default 8 us throttle: same as `runs/f-7.2-stream-8192` (`rev2-stream-8192/`).
- Each cable alone: 9.63, 9.62, 9.63 Gb/s. All three at once, one direction: 4.60 + 6.68 + 4.60 = 15.9 Gb/s, with
  domain0 sending two streams and domain2 receiving two (about 11.3 Gb/s each). Six streams, both directions on
  every cable: about 24.7 Gb/s in total (`rev2-tri/tri.txt`).
- 40 rounds of bouncing all six `thunderbolt_net` interfaces with `thunderbolt_stream` loaded: no DMA activation
  failures in 240 (`rev2-repro-3cables.txt`).
- Rebooting with all three cables attached hangs in firmware before the startup chime; Linux had shut down cleanly.
  Unplugging the cables lets it chime and boot. With one cable, reboots are normal.

## Open questions

- Why each direction stops at about one lane (~9.6 Gb/s) on a two-lane bonded link, and what limits three
  concurrent streams to 15.9 Gb/s.
- Whether a cable chain without a loop also hangs the firmware on reboot.
- The 40 us (`thunderbolt_net`) and 25 us (USB4STREAM) latency floors versus 16 us over veth.
