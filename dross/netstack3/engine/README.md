# Networking engine

`oscar-netstack3-engine` is a userspace IPv4/TCP facade over the pinned Netstack3
core and Oscar bindings. Native and ONET compatibility adapters share it. It
contains no channel RPC, Linux errno conversion, kernel ABI or NIC capability
acquisition.

The caller constructs `Engine` with MAC, IPv4 address/prefix, optional gateway,
MTU and current monotonic nanoseconds. Ethernet frames enter through
`receive_ethernet`; `take_ethernet` returns bounded outbound frames for OETH.
Neighbors are learned through ARP rather than configured static peer entries.
Loopback has routes for 127/8 and this machine's configured address. Addresses
elsewhere in 127/8 are materialized as bounded aliases as needed; their unused
entries are pruned. New connections to 127/8 select 127.0.0.1 as their source; local connections
to the configured address keep that address.

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
  closing objects. Backlogs are clamped to 1–4. Both TCP directions use fixed
  32 KiB rings, so admitted payload storage is at most 2 MiB.
- Up to 32 additional 127/8 aliases, with fixed interface and route configuration.
- At most 64 outbound Ethernet frames, each at most 2,048 bytes. The service adds
  its own bounded channel queues and retains backpressured frames/replies.
- At most 256 scheduled bindings timers and 64 tracked deferred removals.
  Exhaustion marks the worker failed; it does not silently ignore a failed quota.
- The recorded conntrack patch strictly limits the table to 1,024 tuple entries.
  Normal flows use two entries; self-connected flows can use one.
- Upstream NUD has a hard 1,024-neighbor limit per device, garbage-collection
  threshold 512, and ten pending frames per unresolved neighbor. Upstream
  loopback queues have a 10,000-frame limit. These are not a single global
  allocator quota. Many unresolved neighbors or retained internal packets can
  still exhaust the process account/heap; no graceful OOM claim is made.
- IPv4 fragments are rejected before core delivery, preventing fragment-cache
  retention. V1 exposes IPv4/TCP only and rejects IPv6 ingress. Full core IPv6
  code remains imported, but NDP/IPv6 endpoints, UDP, DHCP and DNS are not exposed
  by this facade.

Native saturation diagnostics measured a 200-byte connection object, alignment
8, plus an estimated 16-byte Arc header: 216 requested bytes per retained flow.
At the strict cap, ordinary 512-flow storage is about 108 KiB; one-entry self
flows can require about 216 KiB. Hash-map buckets/control bytes add roughly
50 KiB at this size. These are layout estimates excluding allocator overhead,
not a measured total engine heap bound. The native pressure run reported 1,024
entries, 988 limit hits, zero drops (eligible flows were evicted), four queued
frames and six scheduled timers. Existing unit tests cover the all-established
case where new packets must be dropped instead.

## Regression executable

```sh
CARGO_TARGET_DIR=target/engine-build cargo build -p oscar-netstack3-engine \
  --bin engine-test --target x86_64-unknown-none --offline
```

The binary uses real native entropy and deterministic packet scheduling/logical
time between two engines. It exercises dynamic ARP, SYN/data loss, peek, partial
32 KiB writes and wrap, half-close/EOF, 127.0.0.1/127.0.0.2/own-address routes,
admission, gateway ARP, link loss, 1,100 ARP inputs, fragment rejection, conntrack
saturation and pending-connect cancellation. Staging/execution belongs to the
parent integration workflow; this command acquires no hardware NIC or lease.
