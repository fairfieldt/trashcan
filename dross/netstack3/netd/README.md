# Native Netstack3 service

`oscar-netd` builds the native `unix-netd` executable. One Netstack3 engine
serves both the private ONET v1 adapter used by Unix sockets and the native TCP
capability protocols for TCP, datagrams and resolver jobs. TCP, UDP, ARP, NDP,
timers, routing and socket buffers run in this
process. The service receives Ethernet frames from a separate NIC driver.
It receives no PCI or system resource.

The required bootstrap handles are `ethernet` and `ready`. Optional `service`
is the private ONET connection from the kernel. Optional `native/connect`,
`native/listen` and `native/network` grant native CONNECT, LISTEN or both.
`native/network` continues to grant CONNECT|LISTEN only. `native/datagram`
grants datagram creation; `native/resolver` grants lookup through the explicitly
configured DNS server. At least one service or factory grant is required. There is no global lookup
path. `ip`, `prefix` and `gateway` select IPv4 configuration. The default is
10.0.2.15/24 with gateway 10.0.2.2; an explicit IP has no default gateway unless
one is supplied. A native-only instance does not need Unix registration.

IPv6 uses `::1`, a MAC-derived link-local address, and optional static `ip6`,
`prefix6` (default 64), and `gateway6`. Ethernet MTU must be at least 1280. DAD
runs before READY with a five-second bound when the link is up. A link-down
startup still permits local traffic; link-up runs DAD through the normal packet
and timer loop. Scope 1 names Ethernet and scope 2 names loopback. IPv4 and IPv6
sockets are separate; IPv4-mapped IPv6 addresses are unsupported. This is manual
configuration, without router advertisements or SLAAC.

A resolver grant requires `dns-server`, an IPv4 or IPv6 literal. `dns-port`
defaults to 53; `dns-scope` supplies an IPv6 link-local zone. There is no ambient
resolver or default DNS server. Lookup returns a fresh query channel immediately.
COLLECT is nonblocking and preserves the original answer expiry. CANCEL stops
query I/O and retains a Canceled result until the control channel closes.

Native factories return fresh stream or listener channels. Each channel keeps
its logical rights independently of its transport rights. Accepted stream
rights cannot exceed the listener's grant. Native requests use the common
32-byte framing and the shared `rt::protocol::network` decoder. Inline I/O is
limited to 16,384 bytes; READ distinguishes data, EOF and WouldBlock. Channel loss
is a service failure. A write reply confirms entry into the local TCP send
queue. Clients do not replay a write after an uncertain reply.

The loop waits on one-shot kernel event subscriptions. It requests WRITABLE
only while a reply, Ethernet frame or observer event is pending. Otherwise it
waits for input, peer closure or the engine's actual TCP/connect deadlines.
Readiness is refreshed after frames, requests and timers, including before a
request consumes readable data. One observer registration per object has at
most one service-owned pending event; older events may remain in the kernel's
bounded channel queue. Observer channel type and WAIT authority are validated before ACK; invalid
reverse messages or later read/write failures retire only that watch.

There are 32 engine socket slots, including listener backlog reservations and
closing sockets; backlog is 1–4, and each TCP ring holds 32 KiB. The daemon keeps
at most 32 socket channels, four network factories, one resolver factory and
32 queued Ethernet frames. Up to eight query channels retain pending or terminal
DNS jobs; their UDP/TCP transport sockets share the same 32 engine slots. Reply
storage and fresh endpoint registrations are prepared before destructive
reads or accepts. Capacity exhaustion reports native NoResources and Unix
ENOBUFS. Closing an established stream permits up to five seconds of FIN cleanup,
then forces removal. A pending connection is aborted immediately. Link loss stops affected Ethernet connections
while preserving loopback traffic. Important driver events and unexpected
attachments are handled explicitly.

The upstream core still has infallible allocation paths. Core allocation or
binding bookkeeping exhaustion can stop this worker, closing its channels;
this is not a total graceful-OOM guarantee. The private ONET v1 adapter remains
IPv4/TCP. Native UDP sends one complete packet, including empty packets; receive
returns original/copied lengths, truncation, source, destination and interface.
Receive truncation consumes the whole packet. Each socket holds at most 16
packets/32 KiB. Outbound fragmentation and incoming fragment reassembly are
unsupported. Oversize sends report MessageTooLarge.

DNS supports A and AAAA, validated source/question/ID, up to eight CNAME links,
minimum absolute TTL expiry, bounded UDP retries and TCP fallback after a
truncated reply. The original deadline is capped at ten seconds. There is no
cache. Shared-memory I/O, forwarding and administration remain separate work.

`native-net-test` exercises factory isolation, connecting objects, observer
rearm/cancel and receiver replacement, 96 KiB + 17 exact binary bytes, empty-live
WouldBlock, half-close/EOF, reply-after-FIN, rights attenuation and rejection of
an observer sink without WAIT authority. It additionally exercises IPv4/IPv6
UDP boundaries, zero packets, truncation, metadata, rights and MTU limits;
IPv6 TCP; DNS A/AAAA, unordered and followup CNAME TTLs, foreign source/name/ID,
TCP fallback with split framing, negative answers, cancellation and deadlines, plus the eight retained-job
admission limit and reclamation after capability closure. The Unix suite also starts a second
instance with native factories and a fake OETH link only, without a legacy
service, PCI or system handle, then checks link-close cleanup.

The same fixture has an explicit `mode=external` for a finite physical-peer
proof. It takes only `connect`, `datagram` and `resolver` grants, plus trusted
`peer4`/`peer6` and optional `peer-udp-port`/`peer-tcp-port` values. It checks TCP
binary echo and half-close, UDP packet/empty-packet echo and source metadata,
and DNS A/AAAA with UDP-to-TCP fallback. This mode is an acceptance tool;
applications use the same native APIs without test configuration.

QEMU SMP 1 and SMP 4 passed the full native-only and Unix/OpenSSH suites.
On enlight (MacPro6,1, 24 CPUs), 96 native checks, the native fixture, 49 Unix
probes and the complete SSH suite passed. The 8 MiB functional sample measured
2.28 MiB/s upload and 10.79 MiB/s download; this is not a controlled performance
comparison. A separate boot used the same binaries to verify physical IPv4/IPv6
TCP and UDP with an independent Linux peer, plus A/AAAA DNS over IPv6 UDP and
UDP-truncation fallback to IPv6 TCP. Exact sources, images and transcripts are
archived at `results/dross/enlight-netstack3-dualstack-20261103/` outside this branch.
No GPU workload ran. See the parent validation record for limits and cleanup.

[`build-netd.py`](../build-netd.py) stages both binaries into the Unix suite's
native output. See [native networking](../../NETWORKING.md) and
[service protocols](../../protocol/README.md) for the public contracts.
