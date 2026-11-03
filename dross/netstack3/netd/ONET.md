# Kernel socket service, version 1

This is the private interface between the opt-in kernel Unix personality and netd.
The supervisor registers one end of a native channel with `UNIX_NET_REGISTER`
using its System resource. Netd receives the other end as `@service`; it receives
no System or PCI capability. A new registration replaces only a closed service.

To create a socket, the kernel sends eight bytes `ONET\x01\0\0\0` and transfers
one new channel endpoint. That endpoint identifies the socket. Requests have no
client-selected socket ID and cannot name another endpoint's socket.

One request at a time is outstanding on a socket channel, including across fork
and dup. A request starts with a little-endian u32 opcode. A reply starts with a
little-endian i64: nonnegative result or negative Linux errno. Payloads follow
immediately. Addresses use the 16-byte Linux `sockaddr_in` layout: family 2 in
little endian, port in network order, four IPv4 bytes, eight reserved bytes.

| Opcode | Request payload | Successful reply |
| --- | --- | --- |
| 0, initialize | Empty | 0 |
| 1, bind | Address | 0 |
| 2, listen | u32 backlog | 0 |
| 3, connect or check progress | Remote address | 0, or EAGAIN while pending |
| 4, accept | Empty | Peer address and one transferred socket endpoint |
| 5, receive | u32 maximum bytes, u32 peek boolean | Count, followed by bytes |
| 6, send | Bytes | Count accepted into the TCP send buffer |
| 7, readiness | Empty | Linux poll bits in the result field |
| 8, shutdown | u32 direction | 0; receive, send, or both according to direction |
| 9, local address | Empty | Address |
| 10, peer address | Empty | Address |
| 11, take socket error | Empty | Linux errno in the result field, then cleared |

All request integers are little endian unless stated otherwise. Replies are at
least eight bytes. Only accept transfers a handle. Receive and send chunks are at
most 60 KiB; TCP buffers are 32 KiB in each direction. The service limits live
socket channels to 32 and listening backlogs to four. It reserves bound ports
exclusively; it does not implement SO_REUSEADDR sharing. The kernel checks Unix
credentials before binding privileged ports. Bind accepts loopback, the configured
local address or INADDR_ANY; port zero allocates a free ephemeral port.

The service returns EAGAIN instead of blocking on network progress. The kernel
implements blocking calls, nonblocking flags and poll/select deadlines around
those results. A kernel RPC has a two-second bound. An incomplete transaction
makes that socket unusable; it never consumes a later response as a new reply.
Ordinary signals are deferred until the transaction is consumed. Process death
still cancels the wait.

Closing the last fd drops the kernel endpoint. Netd notices peer closure, releases
the channel and port reservation, and drains an established stream for up to five
seconds before removing it. Closing a factory prevents further requests through
that factory. The service exits when all factories and socket endpoints close,
or when its Ethernet endpoint fails. Restart requires fresh capabilities and
does not restore connections.

The Ethernet driver contract is independent:
[OETH v1](../../unix/ethernet/PROTOCOL.md).
