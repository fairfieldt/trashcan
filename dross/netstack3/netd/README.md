# Native Netstack3 service

`oscar-netd` builds the native `unix-netd` executable. One Netstack3 engine
serves both the private ONET v1 adapter used by Unix sockets and the native TCP
capability protocol. TCP, ARP, timers, routing and socket buffers run in this
process. The service receives Ethernet frames from a separate NIC driver.
It receives no PCI or system resource.

The required bootstrap handles are `ethernet` and `ready`. Optional `service`
is the private ONET connection from the kernel. Optional `native/connect`,
`native/listen` and `native/network` grant native CONNECT, LISTEN or both.
At least one service or factory grant is required. There is no global lookup
path. `ip`, `prefix` and `gateway` select IPv4 configuration. The default is
10.0.2.15/24 with gateway 10.0.2.2; an explicit IP has no default gateway unless
one is supplied. A native-only instance does not need Unix registration.

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
bounded channel queue. Bad observer capabilities are rejected before ACK, and
later watch failures remain local to that watch.

There are 32 engine socket slots, including listener backlog reservations and
closing sockets; backlog is 1–4, and each TCP ring holds 32 KiB. The daemon keeps
at most 32 socket channels, three factories and 32 queued Ethernet frames. Reply
storage and fresh endpoint registrations are prepared before destructive
reads or accepts. Capacity exhaustion reports native NoResources and Unix
ENOBUFS. Closing an established stream permits up to five seconds of FIN cleanup,
then forces removal. A pending connection is aborted immediately. Link loss stops affected Ethernet connections
while preserving loopback traffic. Important driver events and unexpected
attachments are handled explicitly.

The upstream core still has infallible allocation paths. Core allocation or
binding bookkeeping exhaustion can stop this worker, closing its channels;
this is not a total graceful-OOM guarantee. IPv4/TCP is the initial public
service. IPv6 requests return Unsupported; UDP, name resolution, shared-memory
I/O, forwarding and administration need separate API work.

`native-net-test` exercises factory isolation, connecting objects, observer
rearm/cancel and receiver replacement, 96 KiB + 17 exact binary bytes, empty-live
WouldBlock, half-close/EOF, reply-after-FIN, rights attenuation and rejection of
an observer sink without WAIT authority. The Unix suite also starts a second
instance with native factories and a fake OETH link only, without a legacy
service, PCI or system handle, then checks link-close cleanup.

Full QEMU SMP 1/SMP 4 Unix and OpenSSH acceptance passed. On enlight (MacPro6,1,
24 CPUs), 89 native checks, 49 Unix probes, the native fixture and the complete SSH
suite passed over the physical Broadcom NIC. The 8 MiB transfer measured
2.26 MiB/s upload and 10.69 MiB/s download in that functional run; there is no
before/after performance claim. Exact sources, images, commands, hashes and
cleanup evidence are archived in
`results/dross/enlight-netstack3-service-20261103`. No GPU workload ran.

[`build-netd.py`](../build-netd.py) stages both binaries into the Unix suite's
native output. See [native networking](../../NETWORKING.md) and
[service protocols](../../protocol/README.md) for the public contracts.
