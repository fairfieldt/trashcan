# Native Rust SDK

`dross-rt` provides the native Oscar entry point and low-level syscalls, plus an
owned SDK for new applications. Existing programs can continue using `sys` and
the legacy text bootstrap. The public service protocols are still drafts.

Use `default-features = false` when using the SDK inside another runtime, including
a Linux-compatible program. The default `runtime` feature supplies the native
entry point, a heap and panic handler.

## Waiting

`events::Events` is what one thread waits for: make one per thread. `watch` reports
a key from `wait` for as long as a handle shows a signal, re-arming by itself, and
`wait` returns everything ready in one call. `poster` lets other threads add events.
Threads that should take turns over the same sources use `try_clone`.

`call_async` sends a request and reports its reply as an event. `TcpStream::read_async`
and `finish_read` do this for reads, so that a thread with many ready sockets sends
all its reads before waiting for any answer.

```rust,ignore
use dross_abi::event;
use dross_rt::events::Events;

let mut events = Events::new(1024)?;
events.watch(stream.as_handle().unwrap(), event::SOCKET_READABLE, 7)?;
loop {
    for ready in events.wait(deadline)? {
        // ready.key is 7: read from stream.
    }
}
```

With 64 sockets on one thread, under KVM, an echo takes about 270 µs with an observer
per socket, 170 µs with `watch` on what netd publishes, and 115 µs with `read_async` too.

webb3's network thread, `desktop/webb3/src/net.rs`, works this way, with one queue for
its connections, its timeouts and the requests other threads post to it.

## Handles and messages

`owned::Channel`, `Memory`, `Process`, `Timer` and `EventQueue` own their handles
and close them on drop. `as_handle()` borrows a handle. `try_duplicate(rights)`
creates an independent owner and can fail; it never adds rights. No owned handle
implements `Clone`.

`owned::Message` owns its bytes and attachments. All buffer growth is fallible.
Build messages with `try_from_bytes`, `try_extend_bytes` and `try_push_handle`.
Pushing a handle consumes it on success; `PushHandleError` returns it on failure.

```rust,ignore
use dross_rt::deadline::Deadline;
use dross_rt::owned::{Channel, Memory, Message};

let (sender, receiver) = Channel::pair()?;
let memory = Memory::create(4096, 0)?;
let mut request = Message::try_from_bytes(b"buffer")?;
request.try_push_handle(memory.into_untyped())?;

// Failure returns SendError { error, message }, including every attachment.
sender.try_send(request)?;

let mut received = receiver.receive(Deadline::IMMEDIATE)?;
let memory = received.take_handle(0).unwrap().try_into_memory()?;
assert_eq!(memory.size()?, 4096);
```

The example omits the surrounding function's error type. Handle insertion,
conversion and send errors retain owned resources; a caller can recover them
from the error or drop them to close them. A received attachment stays untyped
until an operation verifies its kernel object type. A service protocol annotation
alone does not verify that type.

Taking an attachment leaves its position empty. Sending a partially consumed
message is rejected rather than silently changing attachment positions. Dropping
a message closes all remaining attachments. Receive allocates its complete output
storage before taking a message from the kernel queue; allocation failure leaves
the queued message available.

## Memory mappings

`Memory::map` returns a `Mapping` that may outlive the memory handle. Dropping the
mapping unmaps its range; `unmap()` reports errors explicitly and returns the
mapping on failure. Closing the handle alone does not unmap memory.

A mapping exposes raw pointers, with a writable pointer available only when its
permissions include write access. Dereferencing them requires unsafe code that
establishes bounds, lifetime and synchronization. Another process, mapping or
device may access the same backing. The SDK does not turn a shared mapping into
an ordinary Rust slice or assume a transferred handle removes other writers.

## Waiting and timers

`deadline::Duration` is a relative interval. `Deadline` is an absolute monotonic
time, `IMMEDIATE`, or `FOREVER`. Compute one deadline and pass it through retries:

```rust,ignore
use dross_rt::deadline::{Deadline, Duration};

let deadline = Deadline::from_now(Duration::from_millis(200));
let message = connection.receive(deadline)?;
```

A timeout ends the wait; it does not cancel work already sent. `Timer::arm` uses
the same deadline type. Event queue subscriptions own their registration, cancel
it on drop and expose explicit rearm after dequeue. Their queue must outlive them.

## Threads and synchronization

`thread::spawn_with` starts a thread with an `Options { cpu, stack_bytes }`. The
default lets the kernel choose the CPU with the fewest threads, and a thread stays
on its CPU. `thread::sleep` and `sleep_until` wait without a handle.

`sync::Mutex`, `Condvar` and `channel` block by parking in the kernel, so they work
in native and Linux programs. A `Condvar` wait can end with no change. `channel` is
unbounded, with many senders and one receiver; `send` returns the value if the
receiver is gone or there is no memory to queue it. `Lock` still spins with a yield,
for short critical sections.

## Program heap

The runtime's heap adds 16 MiB arenas as it needs them and keeps them. A block of
1 MiB or more is a memory object of its own, returned to the kernel when freed.
The limit is the program's memory account, not a heap size. When the account is
exhausted an allocation fails: `try_reserve` reports it, and an infallible
allocation ends the program.

## Console

A program with a console has a `console` handle: a channel of raw bytes in each direction, up to 60 KiB a message, the
stream `oscar-terminal` has as `@terminal`. Input is what the keys send, with VT sequences for the others. The program
does its own echo and line editing, and writes CR LF to end a line. Without one, `println!` goes to the log.

## Process launch

`launch::spawn` takes a borrowed executable image, process name, owned bootstrap
message and `SpawnOptions`. Capabilities are supplied explicitly in that message.
The returned `Process` observes the child; dropping it does not terminate the
child. `wait_until` returns its exit code. Successful launch does not imply service
readiness.

`LaunchError` identifies the failure stage. Before enqueue it returns
`ReturnedBootstrap::Unsent(message)`. If the native SPAWN syscall fails after
enqueue, it returns `ReturnedBootstrap::Queued(channel)`. Receive from that channel
to recover the bootstrap capabilities. Their original numeric slots have already
been transferred and are not restored. Dropping either error payload closes its
remaining capabilities.

Without an account selection, launch creates a child account under the caller's
default budget. An explicit account selects shared funding and requires account
write authority. No parent handle table is implicitly inherited.

See [Native program interfaces](../../API.md) for kernel contracts and
[Oscar application API design](../../API-DESIGN.md) for service protocols,
delegation and the environment subsystem boundary.

## Native TCP networking

`protocol::network` provides `Network`, `TcpStream`, `TcpListener` and
`Observation`. `Network::new(capability)` consumes a fresh network-service channel
owner. Each client has one request/reply reader; do not reuse an old connection
in a new client. The service enforces CONNECT and LISTEN grants separately from
kernel channel rights.

`begin_connect(peer, stream_rights, deadline)` returns an admitted, possibly
connecting stream. `connect_tcp(peer, stream_rights, deadline)` waits for the
handshake and requires INFO and OBSERVE in the requested stream rights. These
rights are explicit; the helper does not add them. `listen_tcp(local,
listener_rights, accepted_stream_rights, backlog, deadline)` creates a listener;
port zero selects an available port, reported by `listener.info(deadline)`.
`accept(rights, deadline)` returns a fresh stream or WouldBlock. `accept_wait`
uses the listener's readiness observer while waiting.

```rust,ignore
use dross_rt::protocol::network::{Endpoint, ReadResult, stream_rights};

let rights = stream_rights::READ | stream_rights::WRITE
    | stream_rights::INFO | stream_rights::OBSERVE | stream_rights::SHUTDOWN_SEND;
let mut stream = network.connect_tcp(Endpoint::v4([127, 0, 0, 1], 8080), rights, deadline)?;
stream.write_all(b"request", deadline)?;
stream.shutdown_send(deadline)?;
match stream.read_wait(4096, deadline)? {
    ReadResult::Data(data) => use_bytes(data.bytes()),
    ReadResult::Eof => finish_response(),
}
```

`read(maximum, deadline)` is nonblocking and can return WouldBlock. Positive data
owns its received message; `ReadData::bytes()` borrows private copied bytes without
another allocation. EOF has a separate result and follows buffered input.
Networking-service loss is a transport error, not EOF. `read_wait` waits for
readiness and retries using the same deadline.

`write(bytes, deadline)` returns a possibly partial count accepted into the local
TCP queue. It does not promise delivery to the peer. `write_all` waits for local
capacity and retries only the unaccepted suffix. It returns `WriteAllError` with
`accepted` counting confirmed bytes. If
`latest_request_may_have_succeeded()` is true, the latest batch may have accepted
additional bytes: do not replay that batch automatically. Waiting on backpressure
requires OBSERVE authority. ShutdownSend, ShutdownReceive and Abort have separate
methods; successful Abort retires the local control client.

All helpers preserve one absolute deadline through admission, readiness and
retries. An uncertain or invalid RPC reply retires that serial control channel.
After CONNECT admission, a later transport timeout is OutcomeUnknown even if the
follow-up INFO or OBSERVE request itself could not be sent. Dropping the stream
releases its control endpoint; it does not prove no handshake happened remotely.

There is one normal observer per stream or listener. `observe(mask, key,
deadline)` returns its independent event receiver; `Observation::as_handle()` can
be registered for READABLE in a kernel event queue. `receive(deadline)` timing out
leaves the watch registered. Dequeue an event before `stream.rearm(&mut watch,
deadline)` or the corresponding listener call. Rearm uses the observed revision
and can return a new snapshot with no requested readiness bits; retry the actual
operation before waiting again. Explicit `cancel` uses a control RPC; dropping a
watch closes its receiver without waiting for a service reply.

The waiting helpers temporarily use that same observer slot when they need to
wait. They cannot simultaneously claim it while an application holds its own
normal watch. The service keeps at most one unsent event; historical events can
still occupy the bounded underlying kernel queue if a raw client violates
normal dequeue-before-rearm use.

Inline operations carry at most 16 KiB. IPv4 and IPv6 use explicit address
families; IPv4-mapped IPv6 addresses are not translated. IPv6 link-local endpoints
use interface scope 1 for Ethernet or 2 for loopback. See
[networking contracts](../../NETWORKING.md) and the
[C TCP codec](../../protocol/include/oscar_network.h).

## Native UDP

`Network::open_datagram(local, rights, deadline)` needs a factory with DATAGRAM
authority and returns a fresh `datagram::DatagramSocket`. Request SEND, RECEIVE,
OBSERVE and INFO independently through `datagram::rights`. A local port of zero
selects an available port, reported by `socket.info(deadline)`.

`send_to(destination, bytes, deadline)` accepts one whole packet or returns an
error. A successful count equals `bytes.len()`, including zero; it promises local
acceptance, not delivery. `MessageTooLarge` reports the current packet size limit.
The inline ceiling is 16 KiB, and the interface MTU can impose a smaller limit.

`receive(maximum, deadline)` consumes one packet or returns WouldBlock.
`Datagram::bytes()` borrows its owned message. `metadata()` reports the sender,
local destination, interface, original and copied lengths, and truncation. A
zero-length packet is data. A zero maximum consumes a packet and reports its
original length without copying bytes. `receive_wait` uses one temporary
READABLE/ERROR watch and preserves the same absolute deadline. The explicit
`observe`, `rearm`, and `cancel_observation` methods use the same one-shot rules as
TCP. See the [C UDP codec](../../protocol/include/oscar_datagram.h).

## Native DNS jobs

Construct `resolver::Resolver` from a delegated resolver capability. It uses the
service's configured recursive upstream; clients do not inherit a search path or
an ambient resolver. `lookup(name, QueryType::A, rights, deadline)` admits a query
immediately and returns an owned `ResolveQuery`. AAAA is a separate query type.
Names contain ASCII LDH/underscore labels of 1..63 bytes. A final dot is optional;
`.` selects the root. The maximum is 253 bytes, or 254 with a final dot.

The lookup deadline bounds the query I/O. `collect(deadline)` is nonblocking and
returns WouldBlock while pending. On success, `Answers` owns up to 16 addresses;
`addresses()` yields endpoints with port and scope zero. Repeated collection
keeps the original `expires_at_monotonic_ns()`. `ttl_seconds_remaining()` ages the
minimum TTL across the CNAME chain and answer records. An expired result remains
a historical answer with zero cache lifetime.

`query.wait(deadline)` waits using a temporary COMPLETE/ERROR observer; a shorter
event-wait timeout does not cancel the owned query. `Resolver::lookup_wait`
combines admission and waiting and requires COLLECT and OBSERVE authority.
`cancel(deadline)` acknowledges that I/O has stopped and retains terminal Canceled
status for INFO and repeated COLLECT. `unobserve` cancels only the readiness watch.
Dropping the query closes its control endpoint and cancels outstanding work.

NameNotFound, NoData, TemporaryFailure and ResponseInvalid are distinct DNS results.
Transport loss remains `ResolveError::Client(NetworkError::...)`. An uncertain or
malformed control reply retires the client, and every received attachment is
owned even when validation fails. See the
[C DNS codec](../../protocol/include/oscar_resolver.h).

## Validation

From `dross/`:

```sh
cargo test -p dross-rt --no-default-features --lib
python3 protocol/tests/check.py
python3 protocol/tests/check_network.py
python3 protocol/tests/check_network_extensions.py
SMP=1 PROFILE=release KERNEL_FEATURES=unix,audit-test ./run-qemu.sh
SMP=4 PROFILE=release KERNEL_FEATURES=unix,audit-test ./run-qemu.sh
SMP=4 PROFILE=release KERNEL_FEATURES=audit-test ./run-qemu.sh
```

The SDK extension suite passes 91 native checks on one CPU and 96 on four CPUs, plus
28 POSIX checks in each configuration. Tests include heap exhaustion before
receive, failed-send ownership, bootstrap recovery after failed launch, native
Rust and C directory clients, rights attenuation and provider failure cleanup.
Host codecs agree on 267 framing/bootstrap cases, 1,534 TCP cases and 2,010 UDP/DNS
cases. The service hardware cohort passes 96 native checks and the native
TCP/UDP/DNS fixture on a physical Mac Pro 6,1 with 24 logical CPUs, followed by the full
Unix and OpenSSH acceptance suite. See [network validation](../../netstack3/VALIDATION.md)
for the exact cohort and remaining limits.
