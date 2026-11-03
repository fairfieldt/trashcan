# Memory ownership and allocation

This describes the current memory implementation and the remaining work. These
interfaces account for physical backing and selected kernel overhead, not a
process's complete memory cost.

## Ownership

A memory object owns RAM, borrows boot-module pages, or refers to device memory.
RAM is allocated and zeroed before creation succeeds. There is no demand paging
or swap. Closing a handle releases that handle's reference. CPU mappings, DMA
mappings and transferred handles independently keep backing alive.

Native memory objects retain their whole allocation while a handle or mapping
refers to the object. Unmapping part of such an object does not decommit it.
Linux anonymous mappings instead use page-owned backing: splitting a mapping
keeps references to its surviving pages, and unmapping a range can release the
unused pages. An explicit handle to the original object still retains all of its
pages. Fork copies private pages eagerly and preserves aliases; it does not use
copy-on-write. Shared mappings keep the original backing and its original charge.

CPU translations are invalidated before unmapped frames are reused. DMA
translations are invalidated before a domain releases its reference to backing.
A device domain's destruction first detaches its devices, then releases tables
and memory. A closed application handle alone cannot free a buffer a device can
still reach.

## Process and thread exit

A dying process rejects new threads. Each exiting thread switches to another
stack before its kernel stack and saved floating-point state are released.
Separate counts track logical thread exit and completion of the switch away.
Once every thread has stopped, the process releases its user mappings and lower
page tables, then publishes its exit status. Process and thread handles can
therefore preserve identity and status without preserving the program's RAM.

Small status objects and the address-space root page remain until the last
reference to the process goes away. The kernel's shared stack page-table branches
also remain allocated; stack virtual addresses are not reused.

User stacks have a separate lifetime. The low-level `THREAD_CREATE` API receives
a caller-supplied stack address and does not own its mapping. The native runtime's
`thread::spawn` uses `THREAD_START`, which creates a kernel-owned user stack.
Thread exit releases that stack after switching away, before publishing `EXITED`.
Cleanup verifies the original backing so it cannot unmap a replacement mapping.
`thread::spawn_joinable` returns a join handle that waits for completion and can
be retried after a timeout. Dropping it detaches the thread; stack cleanup still
happens when the thread exits. `thread::spawn` returns a raw status handle that
the caller must close.

The native runtime still creates a 16 MiB heap object at startup. Its mapping
retains that backing until process exit; the runtime does not yet return unused
heap pages while the process stays alive. This is separate from kernel-heap
segment reclamation.

## Allocation accounts

An account records charged bytes, peak charged bytes and a byte limit. An
allocation reserves its charge atomically in the selected account and every
ancestor before acquiring resources. Failure rolls back all reservations and
acquired frames. Peak usage includes temporary reservations, including requests
that later fail. A limit permits allocation; it does not reserve RAM or guarantee
that a request will succeed. Parent counters include their descendants, so adding
parent and child counters would count the same allocation more than once.

Backing is charged once. Duplicating handles, mapping the same object again,
transferring it, or retaining it after its creator exits does not charge its
backing again. New handle storage has its own metadata charge. Accounts live as
long as their children or charged allocations need them. Several processes can
share one account.

The native interfaces are:

| Interface | Behavior |
|---|---|
| `MEM_CREATE` | Uses the caller's default account. |
| `ACCOUNT_CREATE(system, limit)` | Creates a child of the caller's account. Requires the system resource. |
| `ACCOUNT_CHILD(parent, limit)` | Creates a child of a supplied account. Requires write rights on that account. |
| `ACCOUNT_INFO(account, output)` | Writes charged bytes, peak bytes and limit as three `u64` values. Requires read rights; account 0 selects the caller's default. |
| `MEM_CREATE_IN(account, size, flags)` | Creates memory using a supplied account. Requires write rights on it. |
| `SPAWN(..., account)` | An optional final account selects the new process's default account; write rights are required. Zero inherits a child of the caller's account. |
| `SPAWN_LINUX_IN(account, args)` | Starts a Linux program with the supplied default account; write rights are required. |

`dross-rt` provides these calls, including native bootstrap `spawn_in`. Ordinary
native and Linux spawn create child accounts. Fork also creates a child, and exec
keeps its process's account. Creating more processes or accounts cannot escape an
ancestor limit. Only boot creates independent roots; account trees have at most
32 levels. Child-account metadata costs 128 charged bytes in its parent.

The counters include RAM backing, address-space root and lower page-table frames,
thread kernel stacks and saved CPU state. Empty lower user page-table branches
are reclaimed after translation invalidation. Exec transfers table ownership
with the replacement mappings. Small status/root resources can remain charged
while a status handle remains open.

Native channel pairs, queued messages, notifications, timers, event queues and
handle-table storage also have admission charges. Message payload and handle
storage remain charged to the sender until the queued message is released.
Event queues reserve charges for their bounded subscription and packet capacity.
Handle-table growth is fallible, has at most 65,535 usable slots, and reserves slot
identities before object creation or blocking receives. Failed calls release
unused reservations. Sparse trailing capacity is compacted when possible, and
process exit releases the whole table. Linux socket endpoints, listeners and
queued payload/descriptor storage have corresponding charges.

These metadata charges are admission costs, not an exact measurement of every
heap allocation. Shared kernel-heap arenas, allocator metadata, some standard
`Arc` and collection allocations, Linux file descriptions, DMA page-table frames,
boot reservations and device memory are not completely attributed to applications. The counters
must not be presented as total process memory use.

## Reclaimable state

Read-only mounted-file snapshots use weak references in filesystem nodes, strong
references in active open descriptions and mappings, and a separate bounded cache.
The cache retains at most 32 MiB and 64 snapshots; larger files remain usable but
are not retained solely for caching. Memory-object allocation drops cache-owned
references and retries once on physical allocation failure, outside the frame
allocator lock. Active users remain valid. Dropping a cache reference does not
necessarily free RAM if an active user still holds the backing.

Writable mounts read bounded ranges through their service rather than retaining
whole-file snapshots for ordinary reads. Mappings and executable loading still
use explicit snapshots.

Futex registry entries exist only while wait/wake callers retain them. Their
identity follows the backing physical page and offset, including across mapping
splits; an active call keeps its backing resident so that the page cannot be
reused under that identity.

## Allocation failures and limits

Runtime address-space roots, thread stacks, saved CPU state and DMA page tables
now use fallible physical allocation. Partially created stack mappings and DMA
table paths are rolled back. Boot-only allocation may still stop boot when the
machine cannot provide its minimum resources.

The kernel heap starts with a 4 MiB reserve and grows by adding mapped segments.
An expansion is at least 4 MiB, or large enough for the request. Empty expansion
segments are retired and reclaimed at syscall or scheduler safe points. Their
virtual extents are reused only after local and remote translations have been
invalidated. This replaces the former fixed 64 MiB heap. The segment directory
has 256 entries; physical memory and available directory entries still bound
allocation. Failed expansion rolls back data pages and later prunes empty tables.

This is not complete exhaustion handling. Some standard `Arc`, collection and
metadata constructors remain infallible on the current stable Rust toolchain.
Page-owned Linux backing also needs per-page reference metadata in the heap. A
global heap failure in one of those constructors can still stop the kernel.
Empty DMA table branches remain until their domain is released. Boot modules
are permanent.

## Queued capability cycles

Reference counting alone cannot release an endpoint queued into its own receive
queue, or several endpoints queued into one another. Native channel collection
traces queue ownership under a lock shared with queue mutations. References
outside those queues, including in-flight operations, are roots. A reachable
receiver preserves its queued messages. Holding only the sending peer does not
make an otherwise unreachable receive queue recoverable.

Only unreachable queues are drained. Capabilities and backing are dropped after
the graph lock is released, so their destructors can safely wake or close other
objects. Ordinary endpoint transfer and reachable cycles remain supported.
Linux `SCM_RIGHTS` collection applies the same rule to file descriptions, socket
receive queues and pending accepts; weak peer and pathname links are not roots.

Handle close and process teardown request collection. Busy requests are retried
at syscall and scheduler safe points, with at most two tracing passes per call. Physical-memory
pressure also requests collection before retrying an allocation. The supervisor
can call `CHANNEL_COLLECT(system)` explicitly; it returns the number of native
queued messages reclaimed. Scratch allocation failure leaves queues intact and
waits for another operation or pressure request. This traces the known IPC
ownership graphs; it is not a general collector for arbitrary kernel objects.

## Validation

The `audit-test` build checks physical stack allocation rollback, page-table map
and protection rollback, DMA table rollback, budget rollback, partial anonymous
unmap, preservation of native whole-object ownership, fork alias identity, futex
registry cleanup, cache limits and rejection of thread publication after kill.
It also checks ancestor reservation rollback, charged handle storage, exec table
ownership, and growth beyond 64 MiB followed by complete empty-segment reclaim.

The native boot self-test checks backing-budget exhaustion and recovery,
permission checks, shared backing after creator exit, and memory reclamation
while process/thread status handles remain open. `posixtest` checks actual Linux
`mmap`/partial `munmap` accounting, surviving-page contents and queued-fd cycles.
Native tests cover reachable and unreachable channel cycles, concurrent close,
selected shared accounts and native/Linux spawn enforcement. QEMU covers one
and multiple CPUs. The final audit kernel also passes all 60 native and 28 POSIX
checks on a 12-CPU Mac Pro 6,1, including timers, stack joins, budget enforcement
and queued-capability collection. An 8 MiB SSH upload/download matches byte for
byte while the GPU desktop is active. Hardware logs are recorded separately.
These are correctness checks, not throughput measurements.
