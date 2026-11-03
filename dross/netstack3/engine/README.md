# Networking engine

`oscar-netstack3-engine` is a userspace dual-stack TCP/UDP facade over the pinned Netstack3
core and Oscar bindings. Native and ONET compatibility adapters share it. It
contains no channel RPC, Linux errno conversion, kernel ABI or NIC capability
acquisition.

The caller constructs `Engine` with MAC, IPv4 address/prefix, optional gateway,
MTU and current monotonic nanoseconds. Ethernet frames enter through
`receive_ethernet`; `take_ethernet` returns bounded outbound frames for OETH.
Neighbors are learned through ARP/NDP rather than configured static peer entries.
Loopback has routes for 127/8 and this machine's configured address. Addresses
elsewhere in 127/8 are materialized as bounded aliases as needed; their unused
entries are pruned. New connections to 127/8 select 127.0.0.1 as their source; local connections
to the configured address keep that address.

IPv6 is enabled explicitly with `configure_ipv6(Ipv6Config)`. The optional static
unicast address/prefix and default gateway install Ethernet routes; `::1` and
the configured address have loopback routes. Enabling without a global address
provides `::1` and a MAC-derived link-local address. Ethernet addresses undergo
one duplicate-address-detection probe. `ipv6_ready()` becomes true only when
all configured Ethernet addresses are assigned. The service must pump and
receive packets during a bounded startup wait while the link is up; a duplicate
leaves it false. A service may start for local sockets while the link is down.
Link-up repeats Ethernet DAD; local loopback remains enabled.
`ipv6_dad_failed()` is a sticky diagnostic for core-reported duplicate-address
or ongoing conflicts. The engine does not learn configuration from router advertisements, SLAAC or
DHCP. IPv6 requires a configured MTU of at least 1,280 bytes. Interface 1 is
Ethernet and interface 2 is loopback. Link-local endpoints require scope 1;
non-link-local endpoints require scope 0. IPv4-mapped IPv6 and multicast socket
endpoints are unsupported in this interface. ICMPv6 and NDP still pass through
the ingress validator and core.

The adapter calls `pump(now_ns)` after messages, NIC input and timer deadlines.
Each pump processes at most 64 due timers and eight loopback receive batches.
`Pump.more_work` asks for another bounded dispatch; `next_deadline` provides an
absolute deadline. `Pump.failed` reports an exhausted bindings budget and calls
for worker fail-stop. No periodic 1 ms poll is part of this engine.

## Socket contract

`SocketId` is an opaque internal key, never a public client-selected ID. An
adapter maps owned channels to keys. `create`, `bind`, `listen`, `connect`,
`accept`, `read` (with optional peek), `write`, `shutdown`, address queries,
readiness and error accessors return domain errors rather than errno.

Connect reports WouldBlock until established. Read with nonempty storage reports
WouldBlock for an empty live stream, and zero only for EOF or receive shutdown.
Buffered data is returned before a terminal error. Write accepts a bounded
prefix, reports WouldBlock when full, and rejects local send shutdown. Receive,
send and both-direction shutdown are distinct; a rejected shutdown does not
alter either direction. `peek_error` is nondestructive; `take_error` explicitly
consumes the recorded error.

`create_tcp(Family)` and the `*_ip` bind/connect/accept/address methods add
IPv6 while preserving the original IPv4 methods. `IpEndpoint` explicitly
contains either `Endpoint` or `Endpoint6`; the latter includes `scope_id`.

`create_udp(Family)`, `udp_bind`, optional `udp_connect`/disconnect, `send_to`
and `recv_from` operate on the same `SocketId` budget. A send either submits
all bytes or returns an error; success acknowledges local IP acceptance, not
delivery. Unresolved-neighbor and remote queues can drop UDP packets. Payloads
larger than MTU minus 28 IPv4 or 48 IPv6 return `MessageTooLarge`; this interface
does not request IP fragmentation. A queued local delivery must be pumped
before another local UDP submission, bounding this front door to core's larger
loopback queue.

A receive returns sender, actual local destination, ingress interface, copied
length, original length and explicit truncation. Interface metadata preserves
core's local-delivery destination device: `::1` reports loopback 2, while a
local address also assigned on Ethernet can report Ethernet 1 even though the
packet travels internally over loopback. Peek retains the whole packet;
a consuming truncated receive discards its tail. A zero-byte datagram is data,
not EOF. Receive shutdown clears and disables the queue; subsequent receives
return `BrokenPipe`. Send shutdown also returns `BrokenPipe`. UDP readiness,
address queries, errors, close and abort share the regular engine methods.
`udp_queue_usage` reports queued packet count, payload bytes and overflow drops.

The service compares readiness snapshots after each dispatch and supplies its
own monotonically increasing observer generation. Listener accept notifications
update a shared ready flag; connected children receive their core buffers
directly, so child readiness does not depend on a replacement listener notifier.

Abort consumes the key immediately. Closing a pending connection also aborts
immediately. Closing an established control object permits up to five seconds of
flush before forced cleanup; it is not an unlimited orphan/TIME-WAIT retention
policy. Keeping an object after send half-close permits the usual receive/EOF
lifecycle. Link loss aborts external connected/pending streams and records
NoRoute, while local streams and listeners remain available.

## Bounds and remaining account limits

- At most 32 admitted socket states, counting listener backlog reservations and
  closing objects, across both families and TCP/UDP. Backlogs are clamped to
  1–4. Both TCP directions use fixed 32 KiB rings. UDP receive queues retain
  at most 16 packets and 32 KiB payload each, including a packet slot for empty
  datagrams; overflow drops the new packet. The combined admitted payload
  allowance is at most 2 MiB (all TCP). Small queue metadata allocations and
  core internals remain additional.
- Up to 32 additional 127/8 aliases, with fixed interface and route configuration.
- At most 64 outbound Ethernet frames, each at most 2,048 bytes. The service adds
  its own bounded channel queues and retains backpressured frames/replies.
- At most 256 scheduled bindings timers and 64 tracked deferred removals.
  Exhaustion marks the worker failed; it does not silently ignore a failed quota.
- The recorded conntrack patch strictly limits each IP table to 1,024 tuple
  entries: up to 2,048 across IPv4+IPv6. Normal flows use two entries;
  self-connected flows can use one. Resource diagnostics report each family.
- Upstream NUD has a hard 1,024-neighbor limit per IP family/device, garbage-collection
  threshold 512, and ten pending frames per unresolved neighbor. Upstream
  loopback queues have a 10,000-frame limit. These are not a single global
  allocator quota. Many unresolved neighbors or retained internal packets can
  still exhaust the process account/heap; no graceful OOM claim is made.
- IPv4 fragments and all IPv6 fragment headers, including atomic fragments,
  are rejected before core delivery. The IPv6 validator walks at most 16
  well-sized extension headers within the declared payload, permitting TCP,
  UDP, ICMPv6 and NoNextHeader; unsupported or malformed chains are dropped.
  DHCP, DNS policy and resolver parsing live above this engine.

Native diagnostics measured a 200-byte IPv4 connection object and 248-byte IPv6
object, alignment 8, plus an estimated 16-byte Arc header: 216 or 264 requested
bytes per retained flow, before allocator overhead.
At each strict cap, ordinary 512-flow storage is about 108 KiB IPv4 or 132 KiB
IPv6; one-entry self flows can require about 216 KiB or 264 KiB respectively.
The previous IPv4 hash-map estimate adds roughly 50 KiB at this size; IPv6
buckets/control bytes are additional and have not been measured. These are
layout estimates excluding allocator overhead,
not a measured total engine heap bound. The native pressure run reported 1,024
entries, 988 limit hits, zero drops (eligible flows were evicted), four queued
frames and six scheduled timers. Existing unit tests cover the all-established
case where new packets must be dropped instead.

## Regression executable

```sh
CARGO_TARGET_DIR=target/engine-build cargo build -p oscar-netstack3-engine \
  --bins --target x86_64-unknown-none --offline
```

The binary uses real native entropy and deterministic packet scheduling/logical
time between two engines. It exercises dynamic ARP, SYN/data loss, peek, partial
32 KiB writes and wrap, half-close/EOF, 127.0.0.1/127.0.0.2/own-address routes,
admission, gateway ARP, link loss, 1,100 ARP inputs, fragment rejection, conntrack
saturation and pending-connect cancellation. Staging/execution belongs to the
parent integration workflow; this command acquires no hardware NIC or lease.

`dual-stack-test` additionally exercises learned IPv6 neighbors after real DAD,
global and scoped link-local TCP bytes/half-close, UDP4/6 sender/local metadata,
peek/truncation, empty packets, connect/disconnect, RX overflow, shutdown,
`::1`/self-address routes, shared socket admission, scope validation, minimum MTU
and malformed/fragmented IPv6 rejection. These are deterministic in-process
links; physical NIC acceptance is an integration test, not implied by building
the binaries. No new performance claim is made.
