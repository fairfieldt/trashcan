# A reliable USB4STREAM transport for every model (2026-10-13)

**Result.** Every model's links now run over the cables' USB4STREAMs where they exist, with TCP as fallback, and the
streams set themselves up (tblink). Correct: 181 GB went over the streams in link-test (suites, soaks of all six
links at once, faults) with every byte checked and none wrong or missing; a stream given up mid-session fell back to
TCP in 10 ms and was back about 1 s later, a silent peer was detected in 125 ms, and lost and damaged frames, ring
overruns, a killed peer, two processes on one cable and a delayed start behaved as designed. These controllers read a
received frame only once up to three more have arrived behind it (0, 1 or 3, by receiving end); with three extra
frames after every write, the stream is as fast as TCP or faster on every link in every test (64 B ping-pong 32-47
µs against 40-95, chain token 121-160 µs against 283-351, 40 KiB exchange 79-88 µs against 169-219). End to end on
six Macs, Flash-Next decodes 2.2% faster and MTP 2.0% faster than main over TCP, with byte-identical validation dumps
and identical tokens. Qwen3.8-27B TP4 on two Macs decodes 10.7% faster than main over TCP; its MTP was 4.0% slower in
the one pair run. Main's own stream lanes did not start once on these devices: a CLOSE left by an earlier user kills
their handshake, which the link survives. In window 1, a tblink reload on inflight left its links down and the reset of
all six that followed ended in a kernel Oops on inflight; the scripts no longer reload modules or reset controllers.

## Why the old lanes failed

USB4STREAM (`drivers/thunderbolt/stream.c`, 7.2.9 with `kernel/patches-7.2/0005`) as the engine's TP lanes used it
had no flow control and no checks, and the setup was manual. Reading the driver and measuring it (Driver, below):

- **No Tx flow control.** 0005 removes E2E credits from the Tx ring, because these Falcon Ridge NHIs never return them
  and the ring then wedges the controller (results/kernel/tb-loopback-20261006). A frame that finds the other side's
  receive ring (256 frames of 4 KiB) full is lost.
- **Damaged frames reach the reader.** Frames the NHI flags with a CRC error or a buffer overrun are only logged
  (`tbstream: RX CRC error`); their bytes are returned by `read()` like any others. A lost frame therefore looks like
  wrong bytes, not like a gap.
- **No frame boundaries.** `read()` returns the bytes of consecutive frames back to back.
- **Stale frames.** Frames written while the other end is closed wait and arrive in a burst when it opens, as does a
  CLOSE sent to a closed end: a new session can read an old session's data or an end of file. Every close sends a
  CLOSE, so whichever end of a session closes last leaves one at the other (window 2: main's lanes then fail their
  handshake with "unexpected end of file").
- **Slow close.** Every close flushes the receive ring with a 500 ms timeout, and the ring's in-flight list (every
  posted buffer) never empties: each close takes at least half a second.
- **Received frames wait** (found in windows 1 and 2). A receiving controller reads a frame only once up to three
  more have arrived behind it on that stream (0, 1 or 3, by receiving end). The old lanes only ever exchanged (both
  sides write at once), so they rarely waited.
- **Silent death after a reconnect.** An fd held open while the XDomain goes away and comes back stays valid, but its
  paths were set up at open time and are not again.
- **Names that move.** A stream's configfs group is named after the stream service (`0-3.1`), whose index follows
  XDomain discovery order and can swap with the network service's after a reconnect; the group then never attaches.
  Module reloads and controller resets remove the groups. Every one of these needed `scripts/tbstream-up.sh` by hand.

The engine's lanes (`link.rs` before this work) cut exchanges into rounds smaller than the ring and started each
session with a hello on the stream itself; a lost or stale frame broke the session or, in principle, its data.

## Design

### The link (`engine/src/link/`)

A `Link` is one ordered, reliable byte stream each way between two processes over one cable.

- **TCP anchors the session.** `Link::connect`/`accept` open a TCP connection over the cable's `thunderboltN`
  interface (`<peer>.tb<bus>` names), exchange a random session id and each side's stream for that cable: the
  interface holding the socket's address leads to its XDomain, its stream service and the configfs stream `trashcan`
  (`/dev/tbstreamN`). The two ends' controller UUIDs must match crosswise (the same cable). TCP is also the fallback
  path and carries control records. A refused connect is retried for 60 s (a peer still starting is not an error).
- **Records** (`record.rs`): a 48-byte header (kind, session, stream epoch, byte sequence number, cumulative ack, frame
  counters, payload CRC32C, header CRC32C) and up to 4048 bytes on a stream (exactly one 4 KiB frame: records are
  written whole with `writev`, every one but the last of a write filling its frame, so a lost frame costs whole
  records) or 60 KiB over TCP. The reader finds records by their headers and, after damage, by searching for the
  next header whose CRC checks out.
- **Sequence numbers and acks.** Every byte has its offset in the stream as its number; the receiver takes bytes in
  order from whichever path brings them and drops what it has. So the sender may move anything unacknowledged to
  another path at any time.
- **Credits.** Frames are counted per stream epoch: each record carries its frame number (`tx`) and how far the
  sender has read the other side's frames (`rx`, one past the last frame read; a lost frame is passed over by the next
  one read). Data stays within the other side's ring minus 32 frames, control records within the ring minus 4, so
  the ring cannot overflow. The reading thread returns credits itself every 32 frames and with any record it sends.
- **Loss.** A gap on the stream (a lost or damaged frame) is answered with a NAK; the sender goes back to the first
  byte the receiver lacks (go-back-N). Without an acknowledgement for 5 ms it goes back by itself, and asks for credits
  with a ping when its window is full. After 100 ms without an acknowledgement, on a device error or on end of file,
  the stream is given up: the other side is told over TCP, and everything not acknowledged goes again over TCP.
  Only an acknowledgement resets that clock, so a stream that loses the same record again and again falls back.
- **Three extra frames.** A receiving controller here reads a frame only once up to three more have arrived behind it
  (window 2), so every write on the stream ends with three header-only frames (acknowledgements with the latest ack
  and credits), each in its own write. They cost one syscall and a 4 KiB ring slot each.
- **Epochs.** The connecting side opens each attempt at a stream as an epoch: it opens its device, sends START over
  TCP, and both sides send PROBEs on the stream until each has heard the other and been heard (`rx > 0`). Records of
  other sessions or epochs are dropped, so frames and CLOSEs left from earlier never mix in; a CLOSE while an epoch
  opens is ignored as stale. After a fallback the connecting side tries again after 1, 2, 4 ... 30 s. New data moves
  to the stream once everything sent over TCP is acknowledged.
- **Liveness.** A keepalive every second on an idle stream; no frame for 3 s gives it up. TCP keepalive (2 s, 3 probes
  1 s apart) covers the anchor. A receiver holding 32 MiB the application has not read stops reading and says FULL
  over TCP; the sender's stream timers wait meanwhile, so back-pressure is never mistaken for a dead stream.
- **Several cables.** `Lanes` (tensor parallelism) opens one link per cable; with two or three cables each lane also
  joins a TCP path over the next cable, so a lane whose cable goes away carries on over another one. A lane's
  data path fails over when its TCP path errors or leaves data unacknowledged for 2 s while another path remains.
- **Exclusive devices.** A device is opened with an exclusive `flock`; a second process on the same cable keeps to
  TCP. Opening drains what the device holds.
- **Threads.** The application thread drives a link while it sends, receives or exchanges: exchanges spin, idle
  reads spin 100 µs (chains) or 50 ms (TP control) and then sleep in `poll()` on the device and the socket. A
  maintenance thread (`service.rs`) pumps every link the application has not touched for 1 ms (acks, credits,
  keepalives, retransmissions, reading so the other side's sends keep going). Device opens (which wait for an
  earlier close of the device) run on a thread of their own and closes are handed to one, so no application or
  maintenance thread ever waits for the driver's half-second close.
- **CRC32C** on the Mac Pros: the crc32c crate reaches the SSE4.2 `crc32` instruction through a call per 8 bytes
  unless the whole build enables SSE4.2 (2.5 GB/s on the Mac Pros); `record.rs` runs it inline and three records at
  a time (14-21 GB/s on the six Macs; 14.9-17.5 GB/s in window 1's short preflight runs, `w1/preflight-*`), checked
  against the crate in the tests.
- **Above the link**, the wire formats are unchanged: `Lanes` keeps `exchange`/`send_message`/`recv_message` for
  tensor parallelism, and `Conn` the chains' `[u32 header length][u64 blob length][JSON header][blob]` frames
  (`qwen::net`, `qwen4::net`). A `Link` is also `Read` and `Write`.

### The setup (`nodes/macpro/root`)

- `tblink` (the node service, every 5 s) loads `thunderbolt_stream` and gives every working link to another node the
  stream `trashcan` under its current stream service, with input and output HopID 15 at both ends: thunderbolt_net
  takes the lowest free HopIDs from 8 and the controllers allow up to 15, so neither end needs the other's values.
  A group under an old service name (an index swap) or of a link that went away is removed. `tblink links` has a
  `stream` column (`/dev/tbstreamN` or `-`); `tblink reload` removes the streams first, under the same lock as the
  service's passes.
- `/etc/udev/rules.d/70-tbstream.rules`: the devices belong to `render` (mode 0660), which the engine's user is in.
- `scripts/tb-reset.sh` stops tblink on the Macs it resets and starts it again afterwards; the streams come back
  with it. `scripts/tbstream-up.sh` is gone.

## Tests without hardware

`cargo test` (macOS and Linux): the link over localhost with and without a simulated stream (`link/sim.rs`: 4 KiB
frames, a ring that loses what overflows it, frames back to back, frames and CLOSEs waiting for a closed end,
faults) under lost, damaged and cut frames, an overrun, a dead stream, a closed stream, a device that will not open,
stale frames of an earlier session, the other side going away, a receiver that stops reading, a reader and a writer
thread on one link, three lanes of which one loses its TCP path, the old lanes' message and exchange tests, and the
qwen4 chain's frame tests (`qwen4::net`, its `write_frame`/`read_frame` over a link): 86 tests on macOS and in the
aarch64 Linux build VM after the rebase onto 7b417070. `scripts/test-tblink.py`: tblink's streams against a fake
sysfs/configfs.

## Hardware

Six Mac Pros (incite, indite, inflight, alight, invite, enlight) on 7.2.9-p2p, the six-cable ring of
results/qwen4/chain-20261012 (one cable per neighbouring pair, each on its own controller at both ends, two lanes).
Every window ran under six-Mac gpuq leases (`chain6.sh`); link tests and faults used no GPU.

### Window 1: driver checks, latency, soak, faults (`tools/window1.sh`, `w1/`)

2026-10-13 17:29-17:59, binary `trashcan-stream` c9ff3f324f14 (this branch on main 2293550, before the rebase; the
link module has since only gained `Link::spin_readable` for `qwen4::net`), this branch's tblink run by hand
(`tools/stream-ops.sh up`). One tblink pass gave all 12 link ends their stream
(`/dev/tbstream0`/`1`, HopIDs 15/15), and 6 of 6 links moved their sessions to it. `w1/steps.log` has the timeline;
`tools/summary.py w1` prints the tables below.

**Driver** (incite-indite, `tools/tbcheck.py`, `w1/driver.jsonl`):

- Opening takes 0.5-3.7 ms, closing 500-530 ms (5 of 5).
- 300 frames written while the other end was closed: the writer met 3007 EAGAINs over 1.9 s. The reader that opened
  1.5 s later got 256 frames (numbers 26-280: 255 intact, 1 damaged and returned as data); the rest were lost.
- 1000 frames written in 5 ms to an end that does not read for 3 s: it got frames 0-254, the other 745 vanished
  without an error or a log line, and the writer's close read as one end of file.
- A lone frame is not delivered until something follows it (below).

**Lone frames** (`tools/probe-pp.sh`, raw devices, single frames there and back, both ends spinning on non-blocking
reads; a side that has written nothing for 20 ms writes a filler frame):

| frames | follow-up frames written at once | round trip p50 / p90 | fillers per round |
|---|---|---|---|
| 112 B, incite -> indite | 0 | 20.1 / 20.1 ms | 1 each side |
| 112 B, indite -> incite | 0 | 20.1 / 20.1 ms | 1 each side |
| 4096 B, incite -> indite | 0 | 20.2 / 20.2 ms | 1 each side |
| 112 B, incite -> indite | 1 | 57 / 61 µs | 0 |
| 112 B, incite -> indite | 3 | 57 / 61 µs | 0 |
| 112 B, indite -> incite | 1 | 68 / 73 µs | 0 |

A frame that arrives alone stays unread (the driver's read sees no completed frame) until another frame arrives on
that stream or the receiving side writes; one more frame releases it, more do not help. The link's retransmission
timer (5 ms) is what releases lone frames today, which is why request/reply traffic takes whole multiples of 5 ms
below. The cause is below the driver's read (the Rx completion callback does not run for the frame; the stream's
interrupt throttling of 8192 ns is the first suspect, thunderbolt_net polls its rings instead).

**link-test, all six links at once** (every byte checked; 0.24 GB per link per suite):

| link | path | 64 B ping-pong p50 / p99 µs | 20 KiB ping-pong p50 µs | 40 KiB exchange p50 / p99 µs | 5 MiB one way MB/s (each way) | chain token p50 / p99 µs |
|---|---|---|---|---|---|---|
| incite -> indite | tcp | 60 / 101 | 245 | 163 / 402 | 424 / 374 | 324 / 5056 |
| indite -> inflight | tcp | 75 / 128 | 247 | 165 / 304 | 402 / 403 | 340 / 5952 |
| inflight -> alight | tcp | 68 / 108 | 257 | 202 / 297 | 406 / 441 | 327 / 5222 |
| alight -> invite | tcp | 61 / 73 | 272 | 179 / 288 | 472 / 428 | 332 / 6045 |
| invite -> enlight | tcp | 60 / 94 | 254 | 201 / 462 | 493 / 339 | 348 / 7345 |
| enlight -> incite | tcp | 57 / 93 | 243 | 177 / 318 | 520 / 368 | 295 / 5185 |
| incite -> indite | stream | 5089 / 5182 | 5396 | 116 / 260 | 363 / 324 | 6611 / 9654 |
| indite -> inflight | stream | 15081 / 15267 | 5445 | 180 / 276 | 366 / 212 | 346 / 6331 |
| inflight -> alight | stream | 20064 / 20087 | 5425 | 170 / 214 | 335 / 174 | 6604 / 9604 |
| alight -> invite | stream | 20061 / 20079 | 5431 | 128 / 193 | 345 / 204 | 6581 / 9565 |
| invite -> enlight | stream | 20060 / 20097 | 5407 | 154 / 198 | 329 / 220 | 363 / 8587 |
| enlight -> incite | stream | 20098 / 20126 | 5421 | 137 / 246 | 326 / 214 | 6607 / 16003 |

Exchanges, where both sides write at once, are faster on the streams than over TCP. Everything that waits for a lone
frame waits for a retransmission: every 64 B ping-pong record went twice (rewinds = duplicates = rounds), and the
5 MiB transfers stall on lone acknowledgements and credits.

**Soak** (10 minutes, all six links at once, rounds of chain x100, 40 KiB exchange x100, 64 B ping-pong x100 and
5 MiB each way; `w1/soak-*`):

| link | rounds | checked GB | chain p50 / p99 / max µs | 40 KiB exchange p50 / p99 µs | 5 MiB MB/s | 64 B ping-pong p50 µs | crc errors | fallbacks |
|---|---|---|---|---|---|---|---|---|
| incite -> indite | 473 | 4.9 | 6606 / 9582 / 12593 | 179 / 258 | 338 / 475 | 5073 | 0 | 0 |
| indite -> inflight | 266 | 2.8 | 6564 / 9913 / 12690 | 184 / 249 | 348 / 398 | 15077 | 0 | 0 |
| inflight -> alight | 216 | 2.2 | 6603 / 9642 / 12744 | 186 / 220 | 334 / 294 | 20060 | 0 | 0 |
| alight -> invite | 217 | 2.3 | 6614 / 9615 / 10845 | 169 / 195 | 332 / 310 | 20058 | 0 | 0 |
| invite -> enlight | 216 | 2.2 | 6614 / 9670 / 12279 | 187 / 216 | 332 / 259 | 20059 | 0 | 0 |
| enlight -> incite | 217 | 2.3 | 6603 / 9617 / 10999 | 163 / 218 | 336 / 320 | 20097 | 0 | 0 |

16.7 GB checked, none wrong or missing; every link stayed on its stream for the whole 10 minutes.

### Fault injection (window 1)

| Fault | How | Expected | Measured |
|---|---|---|---|
| peer process killed mid-session | `pkill -9` of the server during a chain session (incite -> indite) | clean error at once; a new session back on the stream | the stream ended 13 ms after the kill (the dying process's close), the session failed 0.52 s after it ("connection reset"; the process exit waits for the driver's 0.5 s close); the next session ran on the stream, 21 MB checked |
| stream given up by the peer | server `give_up_stream` 2 s in (indite -> inflight) | fallback to TCP in-session; back on the stream | on TCP 10 ms after the give-up, back on the stream 1.02 s after it; 210 MB checked, no errors |
| peer stopped 2 s | `pkill -STOP`, 2 s, `-CONT` of the server (inflight -> alight) | stream given up after 100 ms of silence; the session goes on; stream back | given up 125 ms after the stop, the session went on over TCP. The stream did not come back in that session: one new epoch had no answer on the stream, later opens of the device failed with EIO. The client hit its 180 s cap (6.6 ms chain tokens), so only `w1/f-stop-inflight.err` records it |
| lost and damaged frames | every 97th frame dropped, every 101st damaged at the client (alight -> invite) | retransmission; every byte checked | chain: 52 MB checked, 12074 damaged records found by CRC, 6528 NAKs; the stream was given up twice and was back 1.0 and 2.0 s later. 5 MiB transfers: 105 MB checked, no fallback |
| ring overrun | 300 frames past the credits (invite -> enlight) | real overruns at the receiver; every byte checked | 105 MB checked; 281 gaps at the receiver, 3 NAKs, 102 rewinds, no fallback |
| two processes on one cable | two sessions at once (enlight -> incite) | one on the stream, the other on TCP at once | one session on the stream, the other on TCP; 31 MB each, checked |
| delayed start | client 8 s before the server (incite -> indite) | the client waits, then the session is on the stream | on the stream, 10 MB checked |
| stream module reload | `modprobe -r thunderbolt_stream`, `modprobe` on indite, one tblink pass | streams back; sessions on the stream | both sessions through indite on the stream, 21 MB each |
| XDomain reconnect | `tblink reload` on inflight | links and streams back | inflight's links to indite and alight did not come back within 90 s (no route, no name); no session ran |
| controller reset | `tb-reset.sh enlight incite`, two rounds | links and streams back | incite-enlight and incite-indite (one lane) back, sessions on the stream; enlight-invite stayed down |

Every session that ended checked every byte: 18.8 GB in the stream suites, the soak and the faults (most of it on the
streams), none wrong or missing.

**inflight's kernel Oops.** The window ends by checking the links; five Macs still had links down, so it ran
`scripts/tb-reset.sh` on all six (17:58:53). inflight had logged "thunderbolt 0-3: failed request link state change,
aborting" every second since its tblink reload; during the reset's reload its host router failed to come up
("failed to allocate host router", -ETIMEDOUT), nhi_shutdown warned "TX ring 0 is still active", and a NULL pointer
dereference in tb_xdp_handle_request (workqueue) hung it at 17:59:10; it needed a power cycle. No GPU work ran. Module
reloads, tblink reloads and controller resets are out of the window scripts: not on the shared Macs.

### Window 2 part A: the hold per link, throttling, one extra frame (`tools/window2a.sh`, `w2a/`)

2026-10-14 11:36-11:53, after the ring was recabled (incite -> inflight -> alight -> indite -> invite -> enlight, each
link derived from the live `tblink links` by `tools/ring6.py`), on main e155a4f7 plus this branch. CPU only, no
reloads or resets. The streams came from one pass of this branch's tblink on every Mac.

**The hold differs by link, and throttling does not change it** (`tools/probe_table.py w2a/probe-*.jsonl`): with the
stream's interrupt throttling at the driver's 8192 ns and at 0, the numbers agree within 50 µs.

| client link | follow-up frames | round trip p50 | fillers per round |
|---|---|---|---|
| incite -> inflight, inflight -> alight, alight -> indite, invite -> enlight, enlight -> incite | 0 | 60.1 ms | 3 |
| the same five | 1 | 40.1 ms | 2 |
| indite -> invite | 0 | 51 µs | 0 |
| indite -> invite | 1 | 51 µs | 0 |

On five links a frame is read once three more have arrived behind it; on indite-invite every frame is read at once;
in window 1, incite-indite needed one.

**One extra frame per write** (`trashcan-stream-push`, 307b6dc24912) was not enough on the five links:

| link | 64 B ping-pong p50 µs (TCP) | 20 KiB ping-pong p50 µs (TCP) | 40 KiB exchange p50 µs (TCP) | 5 MiB MB/s each way (TCP) | chain token p50 µs (TCP) |
|---|---|---|---|---|---|
| indite -> invite | 34 (45) | 107 (235) | 82 (173) | 552 / 457 (339 / 388) | 133 (351) |
| the other five | 5079-5086 (40-95) | 5383-5494 (241-322) | 149-183 (169-219) | 305-356 / 320-376 (292-470 / 307-532) | 6611-6619 (283-337) |

Where nothing is held, the stream beats TCP in every test. The 10-minute soak (`w2a/soak-*`) gave indite -> invite 5783
rounds and 60.1 GB against about 470 rounds and 4.9 GB on each of the others, with 0 CRC errors and 0 fallbacks;
87.0 GB went over the streams in part A, every byte checked. On the five links a stream that fell back (given up by
the peer, a peer stopped for 2 s, the delayed start) came up again and fell back within ~100 ms, over and over: the
acknowledgements that would reset the 100 ms timer wait behind the hold. **Three extra frames per write**
(`trashcan-stream-push3`, c7549a6b1d32): in the simulator holding three frames, 200 ping-pongs take 20 s with one
extra frame and under 2 s with three.

### Window 2 part B: the automatic setup, the link check, Flash-Next and TP4 (`tools/window2.sh`, `w2b/`)

2026-10-14 12:10-12:35 on the six Macs, this branch's build with three extra frames per write (c7549a6b1d32) against
main's engine trashcan-15e5457. No reloads or resets; every link stayed up on two lanes.

**The automatic setup** (`tools/deploy-setup.sh`, before part B's first try at 11:56): tblink (93f08bfda7e5) and
`/etc/udev/rules.d/70-tbstream.rules` (32d656bb08bd) installed on all six and tblink.service restarted; 4 s later every
link was up with its stream at both ends (`/dev/tbstream0`/`1`, root:render 0660). Stopping tblink and removing the
streams, then starting it again (step 5), brought every stream back within one 5 s pass.

**Link check** (CPU only; the window would have stopped before any GPU run unless every link stayed on its stream,
every 64 B ping-pong p50 was under 500 µs and no session failed):

| link | 64 B ping-pong p50 µs (TCP) | 20 KiB ping-pong p50 µs (TCP) | 40 KiB exchange p50 µs (TCP) | 5 MiB MB/s each way | chain token p50 µs (TCP) |
|---|---|---|---|---|---|
| incite -> inflight | 40 (69) | 116 (300) | 84 (177) | 566 / 484 | 130 (316) |
| inflight -> alight | 32 (61) | 102 (322) | 88 (219) | 604 / 453 | 121 (283) |
| alight -> indite | 40 (40) | 111 (259) | 81 (186) | 591 / 520 | 159 (337) |
| indite -> invite | 46 (45) | 109 (235) | 82 (173) | 565 / 462 | 159 (351) |
| invite -> enlight | 47 (95) | 111 (241) | 86 (174) | 556 / 447 | 160 (333) |
| enlight -> incite | 38 (43) | 130 (299) | 79 (169) | 588 / 507 | 141 (325) |

TCP from part A's suite on the same ring. The raw probe with three follow-up frames: 46-142 µs on every link, no
filler needed. The 2-minute soak: 1154-1198 rounds and 12.0-12.5 GB per link, chain tokens 127-131 µs p50, 0 CRC
errors, 0 fallbacks. A stream given up by the peer 2 s into a session was back 1.01 s later and stayed; with the peer
stopped for 2 s the session went on over TCP and ended on the stream. 75.2 GB checked in part B, none wrong or
missing.

**Flash-Next, six Macs** (alternated, main's binary first; `tools/w2_summary.py w2b`):

| metric | main, TCP | this branch, streams | change | this branch, TCP |
|---|---|---|---|---|
| validate, 14-token reference | | dumps byte-identical to main's | | |
| decode, 128 tokens after 512 (two pairs) | 20.88, 21.15 tok/s | 21.35, 21.59 | +2.2% (+2.3, +2.1) | 20.99 |
| MTP, three chat prompts (two pairs) | 29.41 tok/s | 30.01 | +2.0% | 28.98 |
| plain decode inside the MTP runs | 20.69 | 21.04 | +1.7% | 20.46 |
| prefill 512 / 2048 / 8192 | 519.1 / 884.8 / 1126.3 | 519.9 / 887.7 / 1126.2 | +0.15 / +0.33 / -0.01% | 516.6 / - / - |

Every MTP run of a prompt generated the same tokens on every side, each identical to its plain decode. Over TCP the
link costs nothing against main (20.99 against 21.01). Every card held 850 MHz through every run; invite's GPU 0
peaked at 63-66 °C in the pairs (72 °C in the TCP run).

### Qwen3.8-27B TP4 on incite + enlight (`tools/tp4-session.sh`, `tp4b/`; first try in `w2b/tp4/`)

The pair is incite and its ring neighbour with the 27B model (enlight; inflight has none), over their one cable.
Main's stream lanes (`--tp-stream` / `--stream`) did not start once in seven tries, in part B and in this session:
the lane handshake read an end of file. Read dry just before (`tbcheck.py drain`), incite's device held one pending
CLOSE; the drain's own close left one at enlight (a close sends a CLOSE, and one that reaches a closed end waits there
for the next open), and main's node failed on that. Main's lanes have no way past a stale CLOSE; the link ignores it
(epochs, `link/proto.rs`) and started on the same devices every time. So the baseline is main over TCP (`--tp-connect`
/ `--listen`), alternated (12:40-12:51):

| metric | main, TCP | this branch, stream | change |
|---|---|---|---|
| decode, 128 tokens after 512 (two pairs) | 24.37, 25.33 tok/s | 26.64, 28.40 | +10.7% (+9.3, +12.1) |
| prefill 512 | 143.0, 144.8 tok/s | 140.5, 140.1 | -2.5% |
| MTP (one pair, cards cooled to 60 °C first) | 33.88 tok/s | 32.54 | -4.0% |

The MTP runs accepted the same drafts (75 in 51 rounds); the draft steps took the same time and the verification
rounds took 71.7 ms against 68.8 ms. `qwen-check --mtp` and `--verify 3` passed in part B. `qwen-validate` failed there
("lanes stalled for 500ms": in a validation run the captures held the other machine past the 0.5 s a gated wait
allows); forwards that capture tensors are no longer event-gated (`qwen/gpu.rs`), and it passed in this session
(greedy 9419 0 2500 628 353 1438 488 3242 30 248046).

### Where the hold comes from

The ring-position probe (`tbcheck.py pos`, both directions of incite-enlight): a lone frame into enlight needed three
more frames at every position of the receiving ring (mod 4: 3, 3, 3, 3), into incite none (0, 0, 0, 0). So it is not
a write-back per 64-byte line of descriptors but a fixed lag at the receiving controller, 0, 1 (window 1, incite-
indite) or 3 frames depending on the receiving end. All 18 controllers are the same part (Falcon Ridge 8086:156c, rev
0, generation 2), and the same domain number holds on one Mac and not on another, so nothing static explains which
end holds. The interrupt throttling does not change it. Three extra frames after every write cover every end measured.

### Next

7.3 (with the USB4STREAM `busy_poll` attribute: rings without interrupts, completions read by `read()` and `write()`)
is going onto the Macs. If a polled ring sees a lone frame that the interrupt does not report, the first test on one
cable, the raw probe with `busy_poll` on and no follow-up frames, answers within microseconds, and the link drops the
extra frames: its hot receives already spin, and its idle receive, which sleeps in `poll()` on the device (no longer
possible with `busy_poll`), sleeps briefly between non-blocking reads instead. If the probe still needs three frames,
the completion is late in the controller itself and the extra frames stay.
