# Native networking interfaces

Oscar networking uses capability objects provided by a userspace service. Rust
clients get typed objects and explicit ownership. Other languages use the same
versioned messages and transferred handles. Netstack3 supplies the protocol
engine; Oscar defines service access, object lifetimes and resource policy.

The service provides IPv4 and IPv6 TCP and UDP, plus a separate DNS forwarding
resolver for A and AAAA records. Administration remains a separate future
interface. The Linux/POSIX adapter retains its IPv4 TCP contract and translates
descriptor, address and error conventions at the compatibility boundary.

## Objects and authority

| Object | Authority |
|---|---|
| Network | Create connections, listeners or datagram sockets within its delegated realm and grant |
| TcpConnection | Use one TCP connection and observe its state |
| TcpListener | Accept connections on one bound address |
| DatagramSocket | Send and receive complete datagrams within its grant |
| Resolver | Resolve names through an explicitly supplied resolution service |
| NetworkAdmin | Configure interfaces, addresses, routes and related policy |

Applications receive these capabilities through bootstrap or a service directory.
A process name, user ID or known service name does not itself grant access.
Administration and raw-packet access require separate capabilities. Drivers own
PCI, interrupts and DMA; the network service receives an Ethernet channel.

Connect, listen and datagrams have independent factory grants. The existing
combined factory keeps its connect/listen grant; datagrams require explicit
authority. These are service permissions
checked on that factory connection. Kernel READ and WRITE rights on its transport
channel describe message access; they do not authorize every networking operation.
An accepted connection grants access to that connection, not the listener or its
network factory.

A network realm contains its interfaces, local addresses and routes. Loopback is
local to that realm. A wildcard listen address means the addresses in that realm.
Production delegation should also constrain destinations, listen addresses and
ports, connection counts, and buffer budgets. The first bootstrap grants do not
yet implement a general destination-policy language or client-driven narrowing.
The first daemon seeds factories through bootstrap. Advertising factories through
the service directory and creating new narrowed factory connections remain
future work.

## TCP connections

The Rust convenience interface keeps one deadline through the exchange:

```rust,ignore
let mut connection = network.connect_tcp(peer, stream_rights::ALL, deadline)?;
connection.write_all(&request, deadline)?;
connection.shutdown_send(deadline)?;

match connection.read_wait(16 * 1024, deadline)? {
    ReadResult::Data(data) => use_bytes(data.bytes()),
    ReadResult::Eof => finish_response(),
}
```

At the protocol level, creating a connection returns a fresh capability as soon
as it is admitted. That object may still be connecting. `begin_connect` exposes
this state; the convenience `connect_tcp` waits for connected or terminal error.
The service enforces the supplied absolute connect deadline and aborts a handshake
that remains unfinished at that deadline. Successful connection means the TCP
handshake completed; it does not mean the remote application accepted a request.

Reads and writes are nonblocking protocol operations. A read returns bytes, EOF,
or WouldBlock. EOF appears after buffered input has been drained. A zero-length
native read request is invalid, so an empty buffer cannot be mistaken for EOF.
The Rust waiting helper combines these operations with readiness observation.

A write returns the number of bytes accepted into the local send buffer. It may
accept a prefix. That result does not promise peer acknowledgment, application
receipt or persistence. `write_all` retries only the unaccepted suffix, waits for
capacity when needed, and preserves the original deadline. On failure it reports
confirmed progress and whether the latest request's outcome is unknown.

The first protocol sends bounded inline bytes. The service copies accepted bytes
into its own buffers. Large transfers use repeated bounded operations. Shared
memory can be added later with explicit release results; handle transfer alone
does not establish exclusive access or stable input.

## Listening and shutdown

`listen_tcp(local, listener_rights, accepted_rights, backlog, deadline)` returns
a listener capability. Port zero requests an available port; INFO returns the
bound address and selected port.
Backlog and connection admission are bounded. A listener does not grant arbitrary
outbound connection access.

`accept(rights, deadline)` returns a fresh connection capability, or WouldBlock.
The requested rights must fit the listener's accepted-stream grant. INFO on the
connection returns its local and peer addresses. The waiting helper uses the
listener's one-shot readiness observer.
Accepting one connection does not consume the listener.

Shutdown operations have distinct meanings:

| Operation | Result |
|---|---|
| ShutdownSend | Reject subsequent writes and send FIN after previously accepted bytes |
| ShutdownReceive | Stop delivering input to the application, with explicit receive-side closure |
| Abort | Stop the connection and discard queued local work; already transmitted bytes cannot be recalled |
| Close the last control endpoint | Release application ownership and begin bounded connection retirement |

Read shutdown leaves sending available. Write shutdown leaves reading available.
Closing an object is not an application delivery acknowledgment. The service must
bound retained closing connections, buffers, retransmission work and TIME-WAIT
state; closing a client channel cannot create an unlimited retirement list.

## Readiness and deadlines

Each connection or listener can have a one-shot observer channel. Registration
checks current state before arming; readiness already present is reported
immediately. Events identify the subscription, caller key, state revision,
readiness bits and latched error. Reading an event does not consume stream data.

The client dequeues an event before rearming. Rearm checks current state again,
preventing a gap between draining data and waiting for the next change. Readiness
is a snapshot: another reader or writer can change it before an operation runs.
The state revision orders observations; it is not authority to operate an object.

The observer receiver is an ordinary channel. It can be registered with Oscar's
existing object-wait queue alongside timers, processes and other service events.
The service retains at most one pending event per observer. The underlying kernel
channel has its own bounded queue; a raw client repeatedly rearming without
dequeue can retain historical events there. The SDK follows dequeue-before-rearm.

RPCs carry an absolute monotonic deadline. The same deadline passes through
admission, backpressure, observation and retries. A timeout before sending means
no request was submitted. After sending, the result may be unknown. The client
retires an uncertain serial RPC connection so a late response cannot satisfy a
later call. It never silently repeats a write that may already have succeeded.

Peer EOF, connection reset and networking-service loss are distinct results.
Service loss closes its capabilities and fails pending operations. Reconnecting
to a restarted service creates new connections; it cannot reconstruct existing
TCP sessions or safely replay their writes.

## Addresses and errors

Addresses are values, not POSIX structures. The wire encoding has an explicit
family, numeric port, scope and address bytes, independent of Rust layout or C
padding. IPv6 scope identifies an interface inside the delegated realm; it does
not grant interface authority. Decoding an address does not mean it is assigned,
routable or within a capability's policy.

Networking errors describe refused, reset, unreachable, timed out, closed,
address in use, unsupported and resource-limit outcomes. Transport/protocol
failure is reported separately from a networking result. Native clients do not
interpret Linux errno numbers. INFO observes addresses, connection state,
readiness and latched errors without consuming buffered data.

## IPv6 configuration

IPv4 and IPv6 use the same connection and datagram object contracts. Each socket
has one address family; IPv4-mapped IPv6 addresses and application multicast
sockets are unsupported. Loopback includes `::1` and the configured local global
address. Ethernet learns neighbors through NDP and checks new addresses with
Duplicate Address Detection. Ethernet IPv6 requires an MTU of at least 1,280.

A supervisor supplies an optional static global address, prefix and gateway.
The service derives a link-local address and configures `::1`. It does not learn
global addresses or default routes from router advertisements. Link loss keeps
loopback available; returning links restart DAD. Scoped link-local endpoints
name interface 1 (Ethernet); ordinary global and loopback endpoints use scope 0.
Received packets report their actual owning interface, including local delivery.

## Datagram sockets

`Network::open_datagram(local, rights, deadline)` creates a UDP socket bound to
one local endpoint. Port zero selects a port. A DatagramSocket preserves complete
packet boundaries. Each receive returns one packet with sender, local destination,
interface, original length, copied bytes and a truncation flag. An empty packet
is data. A maximum copy length of zero consumes a packet and returns its metadata;
it does not mean EOF.

A send accepts the whole packet or returns an error. Success means local
acceptance and makes no delivery promise. A zero-byte successful send transmits
an empty packet. Payloads larger than the supported path MTU return
MessageTooLarge. IP fragmentation and jumbo frames are not enabled. The receive
queue holds at most 16 packets and 32 KiB; excess incoming packets are dropped.
UDP, TCP, listener reservations and closing connections share the 32-socket
admission budget.

Datagrams use the same one-shot observers and absolute deadlines as TCP.
`receive_wait` waits for readable/error state; `receive` and `send_to` are
nonblocking protocol operations. Closing the last control endpoint retires the
socket immediately. The engine also implements connected-peer filtering; the
first public datagram protocol uses explicit destinations on each send.

## DNS resolver jobs

A Resolver capability grants name lookup through a configured recursive server.
It does not grant raw sockets, arbitrary upstream selection or network
administration. The supervisor must supply the server address explicitly.

`lookup(name, kind, rights, deadline)` immediately returns a query capability.
The kind is A or AAAA. `collect` returns WouldBlock while pending, then an address
set or a defined terminal error. `lookup_wait` is the synchronous helper using
the same overall deadline. Query objects support INFO and the common one-shot
observer protocol. At most eight queries, including retained terminal results,
may exist. A query's active lifetime is at most ten seconds and never exceeds its
original deadline.

Cancellation retires network I/O before acknowledgment and leaves an inspectable
Canceled query until its capability closes. Packets accepted before cancellation
cannot be recalled. Closing a query cancels it and releases its retained result.
Repeated collection does not restart work or refresh a record's lifetime.

Names are ASCII labels with optional trailing dots; callers supply IDNA names in
their ASCII form. There are no search suffixes or ambient resolver files. Results
contain up to 16 addresses, a fixed monotonic expiry and remaining TTL. CNAME
following is bounded to eight links; expiry is the minimum absolute expiry across
the chain and answer set. There is no shared answer cache in this version.

The resolver validates source address/port, transaction ID, question, class and
type. It understands compressed names, ignores unrelated UDP responses, retries
lost queries and uses length-prefixed TCP after a truncated UDP reply. Responses
are bounded to 16 KiB and 64 resource records. NXDOMAIN, no data, temporary server
failure, resource exhaustion, timeout and malformed TCP responses are distinct
outcomes. It forwards to the trusted configured server and does not validate
DNSSEC itself. Protocol details follow [RFC 1035](https://www.rfc-editor.org/rfc/rfc1035),
[IPv6 DNS records](https://www.rfc-editor.org/rfc/rfc3596), and
[DNS over TCP](https://www.rfc-editor.org/rfc/rfc7766).

TLS belongs above a TCP connection and can be supplied by a separate library or
service. Certificate validation, peer identity and authenticated shutdown have
their own contracts; TCP connectivity alone does not establish them.

## Service integration

Netd owns one Netstack3 engine and its configured realm. Native connections and
the Linux/POSIX adapter refer to the same engine objects. The initial private
ONET adapter retains the supported Linux socket behavior while native clients
use the common framed protocol. Future movement of the Linux environment out of
the kernel does not require changing the native connection model.

Acceptance includes the existing local/remote socket and SSH workloads, native
capability restriction, observed readiness without lost wakeups, partial I/O,
half-close, cancellation and channel closure. Resource tests must include pending
accepts, closing sockets, packet queues, timers and fragments, as well as explicit
client objects. A top-level socket count alone does not bound memory use.
