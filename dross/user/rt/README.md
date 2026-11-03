# Native Rust SDK

`dross-rt` provides the native Oscar entry point and low-level syscalls, plus an
owned SDK for new applications. Existing programs can continue using `sys` and
the legacy text bootstrap. The public service protocols are still drafts.

Use `default-features = false` when using the SDK inside another runtime, including
a Linux-compatible program. The default `runtime` feature supplies the native
entry point, a heap and panic handler.

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

## Validation

From `dross/`:

```sh
cargo test -p dross-rt --no-default-features --lib
python3 protocol/tests/check.py
SMP=1 PROFILE=release KERNEL_FEATURES=unix,audit-test ./run-qemu.sh
SMP=4 PROFILE=release KERNEL_FEATURES=unix,audit-test ./run-qemu.sh
SMP=4 PROFILE=release KERNEL_FEATURES=audit-test ./run-qemu.sh
```

The API foundation passes 75 native checks on one CPU and 80 on four CPUs, plus
28 POSIX checks in each configuration. Tests include heap exhaustion before
receive, failed-send ownership, bootstrap recovery after failed launch, native
Rust and C directory clients, rights attenuation and provider failure cleanup.
Host codecs agree on 267 framing and bootstrap cases. The Unix audit build also
passes 80 native and 28 POSIX checks on a physical Mac Pro 6,1 with 24 logical
CPUs, including both native service clients and failure cleanup tests.
