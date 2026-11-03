# Oscar application API design

Oscar applications use capabilities to access objects and messages to ask services
to do work. Rust is the primary application language. The kernel ABI and service
protocols remain usable from other languages without linking Rust code.

This is the intended public API design. [Native program interfaces](API.md)
describes the implemented kernel contracts. The [owned Rust SDK](user/rt/README.md)
implements the first ownership and deadline types, with a direct native launch
helper. Common service framing, bootstrap v2 and the service directory are draft
protocols being exercised by native clients. Full asynchronous reply dispatch,
supervision and the environment subsystem bridge remain future work. Existing
services still use their own protocols.

## Interface layers

The kernel provides handles, memory, channels, waits, timers and the mechanisms
needed to isolate processes and devices. Userspace services provide files,
network connections, windows, audio and service discovery. Service policy and
application convenience functions belong in userspace.

| Layer | Contract |
|---|---|
| Kernel ABI | Explicit syscall numbers, integer widths, rights, errors and calling convention for each supported architecture |
| Service protocols | Versioned bytes and transferred handles, independent of language |
| Rust SDK | Owned objects, borrowed access, typed clients, fallible operations and synchronous or asynchronous waits |
| C bindings | Opaque handles, fixed-width integers, checked pointer and length pairs, explicit ownership and status results |

Rust layouts, references, trait objects, `Vec`, `String`, `Future` and compiler
calling conventions never cross a process or language boundary. A C caller can
use the same protocols directly; languages with a C interface can use the C
bindings. Rust crate versions and wire protocol versions are separate contracts.
The current kernel calling convention targets x86-64.

## Relationship to Fuchsia channels

Fuchsia's Zircon channels and dross channels share the basic transport model:
two endpoints exchange ordered messages containing bytes and transferred handles.
A service protocol gives those messages meaning. Fuchsia's
[FIDL](https://fuchsia.dev/fuchsia-src/concepts/fidl/overview) separates that
protocol definition and language bindings from the channel mechanism. Oscar's
shared schema follows the same separation with a smaller initial scope.

There are deliberate differences in the current contracts:

| Contract | Fuchsia | dross |
|---|---|---|
| Ordinary failed send | Consumes attached handles, including on failure | Returns all attachments to the sender |
| Channel endpoint duplication | Not permitted | Permitted when the handle has DUPLICATE authority |
| Synchronous reply matching | Optional kernel channel_call operation matches transaction IDs | Userspace protocol clients match request IDs |
| Queue pressure | Kernel IPC policy exceptions at implementation-defined limits | Bounded queues return ShouldWait and expose WRITABLE readiness |

See Fuchsia's [channel object](https://fuchsia.dev/fuchsia-src/reference/kernel_objects/channel),
[write](https://fuchsia.dev/reference/syscalls/channel_write),
[call](https://fuchsia.dev/reference/syscalls/channel_call) and
[IPC limits](https://fuchsia.dev/fuchsia-src/concepts/kernel/ipc_limits) contracts.

The SDK does not implicitly clone endpoints. Applications should share a typed
client with one reply dispatcher, or obtain independent connections through the
service directory. Explicit raw duplication requires coordination between readers
and awareness that peer closure waits for the last endpoint reference.

Fuchsia's [component framework](https://fuchsia.dev/fuchsia-src/concepts/components/v2/introduction)
routes capabilities into component namespaces. Oscar's launcher and service
directories apply the same principle of explicit delegation without requiring
the full component framework. The Linux/POSIX environment uses those native
capabilities underneath its own conventions.

## Environment subsystems

An environment subsystem supplies the conventions expected by a family of
applications while using Oscar's native objects and services underneath.
Windows NT's support for multiple application environments is a useful precedent;
Microsoft describes its original POSIX environment and the later Interix subsystem
in its [history of the Windows command line](https://devblogs.microsoft.com/commandline/windows-command-line-the-evolution-of-the-windows-command-line/).

Oscar should have a native environment and a limited Linux/POSIX environment.
The native environment presents capabilities, messages and explicit object
lifetimes directly. The Linux/POSIX environment translates supported Linux binary
interfaces and POSIX conventions into those same services. Application language
does not select an environment: a C program can be native, and a Rust program
compiled for Linux can use the compatibility environment.

| Responsibility | Intended location |
|---|---|
| Native object isolation, address spaces, scheduling, interrupt delivery and DMA protection | Kernel |
| Files, network connections, windows and audio devices | Shared native services |
| Linux executable loading, descriptor tables, path environment, process IDs, errno translation and POSIX policy | Linux/POSIX subsystem and its runtime libraries |
| Entry from an unmodified Linux syscall instruction, validated register transfer and return | Small kernel bridge with an explicitly authorized subsystem |
| Signal delivery, thread setup, memory changes and process control requiring privileged state changes | Generic kernel mechanisms where possible, with compatibility policy in the subsystem |

A library can implement source compatibility for programs rebuilt against it.
It cannot intercept every syscall issued by an existing Linux executable. Binary
compatibility therefore needs a kernel entry bridge or an equivalent execution
mechanism. The intended bridge forwards supported calls to the selected userspace
subsystem and safely resumes the caller with the result. This forwarding
mechanism does not exist yet; Linux syscall dispatch currently runs in
`kernel/src/linux/`.

The subsystem receives authority over its enrolled processes and their delegated
resources. It does not receive unrestricted access to all processes or to the
machine's root filesystem and network. Its Unix path namespace is built from
directory capabilities supplied at launch. A Linux descriptor is a local entry
whose underlying object may be a native service capability. Descriptor duplication
and passing must preserve that object's authority ceiling.

Process identity, open-file state, shared offsets, signals and fork/exec semantics
need one coherent owner within an environment instance. They cannot be independent
translations performed by unrelated libraries. An instance may contain several
related processes; several instances can receive different namespaces and budgets.
Delegating access to one file does not require exposing its whole filesystem.

Forwarded calls need bounded queues and cancellation rules. A blocked client must
not prevent the subsystem from servicing the operation that can wake it. Server
failure must have a defined result for enrolled processes and outstanding calls;
transparent restart is unsafe if the server has lost descriptor or process state.
Subsystem shutdown also follows the same buffer and device quiescence rules as
other services.

Preserve the currently supported Linux workloads during migration. First specify
the supported compatibility profile and keep its tests. Move descriptor, path and
service policy behind native protocols before replacing syscall dispatch. Add
forwarding and privileged process mechanisms only with tests for concurrent calls,
interruption, invalid client memory and subsystem failure. Keep direct native
capability access for existing trusted drivers that combine Linux libraries with
Oscar device interfaces.

The native API remains independently usable throughout this work. New native
services expose capability protocols; compatibility clients translate into them.
Supporting an additional environment must not require adding that environment's
file descriptors, global paths or error numbers to every native service.

## Capability ownership

A handle combines a reference to an object with permission to use it. Its number
is local to a process and can be reused after close. An object name or request ID
is not a substitute for a capability.

The Rust SDK should provide these ownership rules:

| Type or operation | Meaning |
|---|---|
| `OwnedHandle<T>` | Owns one handle; cannot be copied; closes it on drop |
| `BorrowedHandle<'a, T>` | Borrows an existing handle without taking responsibility for closing it |
| `try_duplicate(rights)` | Explicit, fallible duplication that can only retain or remove authority |
| `into_raw()` | Gives ownership of the raw handle to the caller |
| Unsafe `from_raw()` | Takes responsibility for a valid, uniquely owned raw handle of the declared type |

There is no infallible `Clone` for owned handles. Constructors and received
handles must validate the object type before exposing typed operations; Rust
types do not replace kernel rights checks. Until a type can be verified, a received
handle remains untyped. The SDK must not assume every wrapper is safe to share
between threads merely because its handle is an integer.

Dropping a process handle releases observation or control of that process; it
does not imply termination. Closing a service endpoint does not revoke its
duplicates or undo completed work. Drop performs local cleanup without waiting
for a service reply. Shutdown, flush and cancellation are explicit operations.

Kernel rights describe access to kernel objects. For example, `WRITE` on a
channel permits sending messages; it does not grant every operation supported by
the server. A service must enforce its own authority through the objects and
restricted connections it issues. A read-only directory capability must remain
read-only even if its transport endpoint allows both requests and replies.

Delegation is explicit. A launcher or service passes only the capabilities its
recipient needs, with narrower rights where supported. A privileged service must
use the authority selected for that connection when acting on a caller's behalf.

## Messages and transfer

A message owns its bytes and attached handles. The receiver identifies attachments
by their positions in that message, never by a sender's numeric handle values.
Protocol schemas define each attachment's type, rights and meaning.

The SDK send operation consumes a message on success. On failure it returns an
error containing that same message and its still-owned handles. This follows the
kernel's atomic handle-transfer contract. A full channel returns backpressure;
an optional waiting send helper observes writability and retries with the same
absolute deadline.

An illustrative Rust interface is:

```rust,ignore
fn try_send(&self, message: Message) -> Result<(), SendError>;
// SendError { error: Error, message: Message }

fn receive(&self, deadline: Deadline) -> Result<Message, ReceiveError>;
```

Message construction and receive storage allocation are fallible. The SDK must
reserve valid output storage before receiving and own every received handle
immediately, including while decoding fails. Rejected or unused attachments are
closed. Unknown attachment positions, duplicate references to an attachment,
unexpected handle counts, invalid lengths and unsupported versions are protocol
errors. Resource limits apply before allocating from untrusted lengths.

The kernel currently limits a message to 64 KiB and 64 handles, and each receiving
queue to 256 messages. A protocol header counts toward that byte limit. Larger
data travels through explicitly bounded memory objects or a flow-controlled
stream. Services also bound accepted work, retained buffers and outstanding
requests; a bounded channel alone does not bound service resource use.

## Common service framing

New public protocols should share a small framing format. This draft uses a
32-byte header with explicit little-endian fields; it is encoded field by field,
not by copying a Rust or C struct.

| Byte offset | Field | Meaning |
|---|---|---|
| 0 | `u32 protocol` | Protocol identifier assigned in the shared schema registry |
| 4 | `u16 major` | Incompatible protocol revision |
| 6 | `u16 kind` | Request, reply or unsolicited event |
| 8 | `u32 ordinal` | Operation or event number within that protocol |
| 12 | `u32 body_bytes` | Exact byte length following the header |
| 16 | `u64 request_id` | Nonzero for requests and matching replies; zero for unsolicited events |
| 24 | `u16 handle_count` | Exact number of transferred handles |
| 26 | `u16 flags` | Zero in the first revision |
| 28 | `u32 reserved` | Zero in the first revision |

The initial schema assigns numeric identifiers and kind values before the first
implementation. Each operation then defines its body, result variants, permitted
attachments and limits. Fields use fixed widths; strings use bounded UTF-8 with
explicit byte lengths. Padding and alignment are defined by the schema. No raw
pointers or virtual addresses are meaningful to the peer.

Unknown major versions, flags and request ordinals receive a defined unsupported
result when the header is valid enough to reply safely. A malformed header closes
the connection and releases attachments. Extensible fields must have explicit
lengths and a rule for skipping them; appending bytes silently is not a versioning
strategy. Optional operations are discovered through a protocol description
operation, rather than inferred from the OS version.

A request ID is unique for the lifetime of its connection and is never reused
there. Exhaustion requires a new connection. A client connection has one reply
dispatcher, which routes replies to pending calls and events to subscribers.
Independent threads must not race to read replies from the same endpoint.
Duplicate or unexpected replies are protocol errors. Timed-out calls remain
tracked until a terminal response or connection retirement; that tracking is
bounded, and exhaustion stops new calls.

Existing untagged protocols require stricter handling: after sending a request,
an uncertain or timed-out reply makes that connection unsuitable for another
call. Retire it so a late reply cannot satisfy the next request. Retiring a
connection does not prove the server stopped the original operation.

## Memory and shared buffers

`Memory` owns a handle to backing storage. A `Mapping` owns a CPU mapping with its
own lifetime. Handles, CPU mappings, DMA mappings and operations may independently
retain the backing. Duplicating or transferring a handle does not remove those
other references. Memory accounts limit backing and associated metadata; an
allocation may fail even when the application has spare virtual address space.

Rust ownership of a handle does not establish exclusive access to the bytes.
Another process, an existing mapping or a device may still modify them. Safe
mapping APIs must not expose ordinary Rust references whose aliasing or mutation
rules the OS cannot enforce. Shared mutable data needs an explicit synchronized
access API, a snapshot, or an unsafe interface with documented obligations.
Read-only mapping permissions alone do not make bytes immutable to other writers.

Every buffer operation specifies the memory range, format, maximum length, access
direction and release condition. A service requiring stable input either copies
it into its own memory or uses an ownership mechanism that actually excludes
other writers. Moving a Rust handle is insufficient evidence of exclusivity.
Small control values should usually travel in the message itself.

An operation retains everything needed to access its buffers until release is
confirmed. A device timeout must not free or reuse memory while DMA may continue.
The driver must establish quiescence or retain the affected mappings and backing
until recovery makes reuse safe.

## Deadlines and cancellation

`Deadline` represents an absolute monotonic time or an explicit infinite wait.
`Duration` represents an interval. They are distinct types; wall-clock timestamps
are a third type. A convenience method computes a deadline once at the start of
an operation. That deadline passes through queue admission, retries and reply
waits without restarting the interval.

A wait timeout means the caller stopped waiting. It does not say whether a sent
request was accepted, performed or completed. The client must preserve this
distinction in its result: failures before send can return the unsent request;
timeouts after send report that the outcome may still be pending. Retrying a
write, launch or connection attempt requires a protocol-defined retry rule or
an explicit caller decision. Request correlation alone does not make retries
safe, especially across reconnection.

Cancellation is a service request referring to the original request ID. Its
terminal response says whether cancellation prevented the operation, whether it
had already completed, or whether it is still in progress. A successful
cancellation acknowledgment guarantees the promised buffer release. It cannot
promise to undo an already committed filesystem change or displayed frame.
Protocols needing multi-stage completion define the stages explicitly.

Dropping an asynchronous future stops that future from receiving a result. It
must not silently release memory still in use by a service or device. The
connection's pending-operation state retains owned resources until completion,
acknowledged cancellation or safe recovery. The API must expose a way to finish
or cancel that state without requiring an unbounded wait in `Drop`.

## Events and completion

Object wait subscriptions remain the common readiness mechanism. The SDK should
own a registration, identify it by its subscription ID, and cancel it on drop.
Rearm follows dequeue and checks the current state. An event is a snapshot of
readiness; consuming it does not consume the underlying message or operation.

Service events travel as messages on the service connection or a separately
delegated event channel. Servers define ordering between replies and events.
Ordering on one channel does not establish ordering across different channels.

Completion names state what has happened:

| Result | Guarantee |
|---|---|
| Accepted | The service admitted the request; work may remain |
| BufferReleased | The operation will no longer access the specified client buffer |
| Presented | The window protocol's defined presentation point was reached |
| Durable | The filesystem's stated persistence boundary was reached |
| Played | The audio protocol's defined playback point was reached |

Each protocol specifies whether its completion also releases buffers, how much
work may be queued, and what errors can still occur afterward. A GPU fence, a
successful channel send and an audio playback completion have different meanings.

## Process launch and supervision

A launch request identifies the executable, arguments, resource account and
named bootstrap capabilities. There is no implicit inheritance of the parent's
handle table. Filesystem roots, service directories, networking authority and
diagnostic sinks are passed explicitly. Linux compatibility may build its usual
environment on top of this launch contract.

Bootstrap decoding takes ownership of attachments. Taking a named capability
removes it from the bootstrap object; unused capabilities close when that object
is dropped. Duplicate names, missing required capabilities and malformed values
are errors. Protocol identifiers distinguish capabilities with different service
interfaces even when each is transported by a channel.

The launcher separates starting a process from the child reporting readiness.
`Started` returns a process observation handle and launch identity. `Ready` is an
explicit child protocol message after its services are usable. Exit observation
includes a defined exit reason. Starting a child does not guarantee readiness.

Launch is a sequence of ownership transfers, not an assumed atomic transaction.
The API reports which stage failed and cleans up capabilities it still owns. It
must not promise to return original handles after they have already been sent
to another process. Failures after a possible start require an observable launch
identity so the supervisor can resolve whether a child exists.

Supervision policy belongs in userspace: restart decisions, dependency ordering,
budgets and cooperative shutdown. A process observation handle does not imply
authority to terminate it. Forced termination needs a separate control capability
and a defined device cleanup path before DMA memory is reused.

## Service discovery

A service directory is a capability to resolve a bounded set of names. Connecting
through it returns a fresh client endpoint for a specified protocol. A name has
meaning only within that directory; it is not an ambient system-wide permission.
An application normally receives a directory containing its allowed services.

Directory lookup grants access only as allowed by that directory. Registering or
replacing providers requires separate authority. Listing services, where allowed,
reveals only the delegated view. Creating a narrower directory is an explicit
delegation operation; it cannot add names or authority absent from its source.

Connections identify a particular service instance. Provider replacement affects
future connections; it does not redirect existing ones silently. Clients see peer
closure or a defined shutdown event and explicitly reconnect. Reconnection uses
the original deadline and does not automatically replay outstanding operations.
After delivering endpoints, the directory releases its copies. Pending connection
setup is discarded when the requester or provider closes its connection to the
directory.

## Applying the design

| Interface | Capability and operation model |
|---|---|
| Filesystem | Directory and file objects; offset-based I/O; explicit buffer release and durability |
| Networking | Delegated network access, resolver, listener and connection objects; bounded streams and explicit half-close |
| Windows | Session and window objects; surface buffers, input events, resize generations and presentation results |
| Audio | Output or stream objects; bounded buffers, format negotiation, playback position and release results |
| Synchronization | Language atomics plus explicit parking identity tied to a memory object and offset; bounded waits |
| Time | Monotonic deadlines and timer objects; wall-clock access through a separate interface |
| Diagnostics | Delegated log and inspection access; process exit and crash reports with bounded retained data |

Implement the foundation in this order:

1. Owned and borrowed handles, fallible messages, memory mappings and deadline
   types over the existing syscalls, preserving the raw interface for bringup.
2. One shared protocol schema and codec with Rust and C bindings. Use bootstrap
   and the service directory to prove ownership and interoperability before
   applying it to every existing service.
3. Launcher and directory protocols with explicit delegation, process readiness,
   failure stages, environment selection and service replacement behavior.
4. Migrate filesystem, networking, windows and audio clients to those contracts
   as their public interfaces develop. Keep their device-specific details inside
   their services.
5. Separate Linux/POSIX policy into an environment subsystem while preserving the
   supported binary profile. Introduce the syscall bridge after the native
   services and process mechanisms it needs are ready.

Before declaring a protocol stable, verify failed sends preserve ownership,
decode errors close attachments, rights cannot grow through delegation, deadlines
survive retries, late replies cannot reach another call, and cancellation releases
buffers only after access stops. Include Rust-to-C and C-to-Rust exchanges and
provider restart while a request is outstanding. Wire compatibility requires
these behaviors as well as matching bytes.
