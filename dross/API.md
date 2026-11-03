# Native program interfaces

dross programs use handles to reach kernel objects and channels to reach system
services. The shared ABI is in `abi/src/lib.rs`; `user/rt/src/sys.rs` provides
Rust wrappers. Linux programs can use the same wrappers through the native syscall
namespace. Linux compatibility supplies its own file descriptors and conventions
over these mechanisms.

New Rust clients can use the [owned SDK](user/rt/README.md) for handle ownership,
fallible messages, mappings, typed deadlines and direct native process launch.
The low-level `sys` interface remains available for existing programs.

See [Oscar application API design](API-DESIGN.md) for the proposed Rust SDK,
language-neutral service framing, capability ownership, environment subsystems,
process launch and service discovery. This page describes the implemented kernel
interface.

## Handles and authority

A handle names one object in the calling process and carries rights. Its integer
value has no meaning in another process. Handle zero is invalid. Treat handle
values as opaque: a closed slot can be reused for a different object.

Operations check both the object's type and the required rights. Duplicating a
handle requires `DUPLICATE` and can only retain or remove rights. The duplicate is
an independent handle to the same object. Transferring a handle requires
`TRANSFER`. Observation requires `WAIT`; it does not authorize reading messages,
changing memory or controlling a process.

Closing a handle releases that reference. Other handles, mappings and active
operations can keep the object alive. Closing one handle does not revoke every
other reference to its object. The native API has no general revocation operation.

## Memory objects and mappings

A RAM memory object owns physical backing, allocated and zeroed at creation. A
mapping grants access to a range of that object at a virtual address. Handles,
CPU mappings and DMA mappings retain their backing independently. Shared backing
is charged once to its memory account. The account survives while a charge remains.

Native CPU mappings require `MAP` and the requested access rights. On x86,
accessible native mappings include read access: write-only and execute-only
requests are rejected. A mapping records its maximum permissions so Linux
`mprotect` cannot add authority later. Splitting or inheriting a mapping preserves
that ceiling. A memory handle recovered through `VM_MEMORY` receives the mapping's
current access rights. DMA permissions are checked independently for device reads
and writes.

Converting a memory handle to a Linux fd preserves its access rights. Shared
mapping and protection changes cannot add permissions. These fds refer to fixed
backing and cannot be resized. A readable source may be copied into separately
owned private writable memory; changes to that copy do not affect the source.

Closing a handle does not undo existing mappings. Unmapping native memory releases
the mapping's reference; an explicit memory object still retains its whole
allocation. Linux anonymous mappings can release individual unused pages. CPU and
device translations must be invalidated before released pages are reused.

A service protocol must define when an operation stops using a shared buffer.
Sending its handle does not remove existing writable mappings. A service requiring
a stable snapshot must enforce its ownership or take a copy. A completed wait or
a timeout alone is not evidence that a device has stopped accessing a buffer.

See [Memory ownership and allocation](MEMORY.md) for accounting and reclamation
details, including ancestor budgets, overhead admission charges and the absence
of demand paging.

## Messages

A channel has two endpoints. A write sends one complete message to the other end.
Message boundaries are preserved. Each message contains at most 64 KiB of bytes
and 64 transferred handles; each receiving queue holds at most 256 messages.

A successful write moves every supplied handle into the message. A failed write
keeps the sender's handles and numbers intact. Duplicating first lets the sender
retain a separate reference. A full queue returns `ShouldWait`; writes do not
block. Reads accept an absolute deadline. A destination buffer that is too small
returns `BufferTooSmall` without consuming the message. Callers must provide valid
output buffers; an output-address fault can consume a received message.

Already queued messages can be read after the peer closes. Once those are drained,
reads return `PeerClosed`. Readiness describes current channel state, not a promise
that another thread will leave that state unchanged until the caller acts.

Unreachable queued-capability cycles are traced and reclaimed. Reachable endpoint
transfers remain supported; the supervisor can request `channel_collect(system)`
explicitly. See [memory ownership](MEMORY.md#queued-capability-cycles).

Requests, replies, cancellation of work and shared-buffer release are service
protocols. An event subscription does not supply those protocols automatically.

## Object wait subscriptions

Create an event queue with capacity from 1 through 256, then register conditions
on handles. The runtime provides:

| Call | Result |
|---|---|
| `event_queue_create(capacity)` | Queue handle |
| `object_subscribe(queue, source, signals, key)` | Subscription ID |
| `subscription_rearm(queue, id)` | Registration armed or an error |
| `subscription_cancel(queue, id)` | Registration removed or an error |
| `event_queue_wait(queue, deadline_ns)` | One `Event { subscription, key, observed }` or an error |

The current waitable objects are:

| Object | Conditions |
|---|---|
| Channel | `READABLE`, `WRITABLE`, `PEER_CLOSED` |
| Notification | `PENDING` |
| Process or thread | `EXITED` |
| Timer | `EXPIRED` |

Notification observation does not clear its pending bits. Channel observation does
not consume messages. Reading one event removes that queue entry, without changing
the underlying object's condition. Services can report socket and GPU completion
events over channels or notifications.

```rust
let queue = sys::event_queue_create(16)?;
let signals = event::READABLE | event::PEER_CLOSED;
let subscription = sys::object_subscribe(queue, connection, signals, 7)?;
let ready = sys::event_queue_wait(queue, sys::after_ms(200))?;
// ready.subscription identifies this registration; ready.key is 7.
// Drain available messages, including any queued before peer closure.
if ready.observed & event::PEER_CLOSED != 0 {
    sys::subscription_cancel(queue, subscription)?;
} else {
    sys::subscription_rearm(queue, subscription)?;
}
```

Registration checks current state under the same lock used to update it. If a
requested condition is already true, an event is queued immediately. Otherwise
the registration observes subsequent updates without a registration gap.

Each registration has a stable 64-bit subscription ID and a caller-supplied key.
An event contains that ID, key and a snapshot of signal bits at enqueue time.
Another thread may change the source state before the event is read. Readiness is one-shot:
delivery disarms the registration. After dequeuing its event, rearm checks current
state again. Rearming an active registration or one with a queued event returns
`Busy`. The registration occupies its slot until cancelled, including while
disarmed. Each slot reserves room for one event, so delivery cannot overflow the
queue. Capacity exhaustion is reported when registering.
Each object also allows at most 256 active observers, and each logical source
handle allows at most 256 registrations across all queues.

Cancellation removes the registration and its queued event, and prevents future
delivery from that registration. It cannot retract an event another thread has
already dequeued. An invalid output address leaves a queued event available.

Subscriptions follow the logical source handle when it is transferred. A
duplicate has a different identity. Closing the subscribed handle cancels its
active observation but preserves an event already queued; rearm returns
`PeerClosed`. Reusing its integer slot does not attach the old subscription to the
new handle. Cancel closed or unused registrations to release their queue slots.

Queue access can be divided by rights: registration, rearm and cancellation
require `WRITE`; dequeue requires `READ | WAIT`. Observing a source requires its
`WAIT` right. Queue destruction releases registrations without retaining sources.
An operation already waiting on the queue retains it until that operation ends;
closing another reference does not cancel that wait.

## Timers

`timer_create()` returns a timer handle. `timer_arm(timer, deadline_ns)` sets a
one-shot absolute deadline on the monotonic clock. Arming clears `EXPIRED` and
replaces any previous deadline. A past deadline expires immediately. `FOREVER`
leaves the timer disarmed. `timer_cancel(timer)` disarms it and clears `EXPIRED`.
Arming and cancellation require `WRITE`; subscribing to `EXPIRED` requires `WAIT`.

Expiration sets a persistent signal. Read the queued event, then arm the timer and
rearm the subscription to use it again. Timer cancellation does not remove an
event already queued by an earlier expiration. That event is a historical signal
snapshot; use subscription cancellation when the queued event must also go away.
Closing the last timer reference cancels its pending deadline.

Timers currently have a limit of 256 objects per creating CPU. Creation reports
exhaustion; it does not allocate an unbounded timer list. Expiration uses the
tickless scheduler and bounded interrupt work, so a deadline is an earliest wake
time, not a guarantee of immediate execution.

## Deadlines

Native waits use absolute nanoseconds on the monotonic clock returned by `CLOCK`.
`FOREVER` means no deadline. A deadline of zero performs an immediate check: ready
data wins, otherwise the wait returns `TimedOut`. Other past deadlines behave the
same way. Conversion rounds up and saturates so large deadlines cannot wrap into
earlier ones. `after_ms` also saturates instead of wrapping.

Compute a deadline once and pass it through the operation's successive waits.
Recreating a relative timeout at every retry extends the total allowed wait.
Event-queue wait timeout leaves registrations intact. Cancelling a subscription
stops observation; it does not cancel the service operation being observed.
Deadlines bound waiting, not scheduling latency or the completion time of device
work. They are not a hard real-time guarantee.

## Design references

The event model draws on Fuchsia's
[object wait subscriptions](https://fuchsia.dev/reference/syscalls/object_wait_async),
[port cancellation](https://fuchsia.dev/reference/syscalls/port_cancel) and
[absolute-deadline port waits](https://fuchsia.dev/reference/syscalls/port_wait).
dross uses explicit subscription IDs and rearm operations, with event capacity
reserved at registration. These are dross contracts; it does not implement the
Zircon ABI.
