# Netstack3 network service

This standalone workspace packages a pinned Fuchsia Netstack3 core for Oscar.
`engine/` supplies IPv4/IPv6 TCP and UDP, learned ARP/NDP, routes, loopback,
bounded socket admission and monotonic timers. `dns-wire/` validates DNS
messages; `resolver/` forwards bounded A/AAAA jobs with retry and TCP fallback. `netd/` runs it as a native userspace service.
The service replaces smoltcp and exposes both native network capabilities and
Oscar's existing private Linux socket adapter. Drivers supply OETH frame channels;
the service receives no PCI or system capability.

[Native networking interfaces](../NETWORKING.md) describes the public contract.
[ONET v1](netd/ONET.md) documents the Linux adapter. The proof programs remain
separate: simulated links and logical time exercise the engine without a NIC.
Read [VALIDATION.md](VALIDATION.md) for the tested artifacts and their limits.

Build the service and its native acceptance program:

```sh
python3 build-netd.py
```

The script prepares pinned sources, uses this directory's toolchain and release
profile, and stages `unix-netd` and `native-net-test` under the Unix workspace's
native output directory. `../unix/build.sh` calls it automatically. The normal
Unix/QEMU test image runs native capability tests before the Linux and SSH tests.
Native UDP, IPv6 and DNS are implemented. DHCP, router-advertisement address
configuration, shared-memory I/O and administration remain future work.

Run from this directory with Python 3.12+ and rustup:

```sh
python3 prepare.py
cargo check -p netstack3_core --locked
cargo run -p oscar-netstack3-proof --locked
cargo test -p oscar-netstack3-proof --locked
cargo test -p netstack3_core -p netstack3_base -p netstack3_tcp -p netstack3_ip -p netstack3_device -p netstack3_filter -p netstack3_datagram -p netstack3_udp -p netstack3_icmp_echo --lib --features testutils --locked
```

`prepare.py --offline` only reapplies/verifies the patch and regenerates manifests
from the already fetched subset. Imported sources, generated manifests and build
artifacts stay under ignored `target/`; the pin, packaging, patch and harness are
reviewable here. The parser handles the pinned declarative Bazel target subset,
not arbitrary Starlark. Changing the pin requires reviewing the dependency graph,
patch, third-party versions and features, followed by the complete checks.

The source is Fuchsia revision
[`650b15d5ea9cfe67faf22ea68cb40e8bc3293303`](https://fuchsia.googlesource.com/fuchsia/+/650b15d5ea9cfe67faf22ea68cb40e8bc3293303/src/connectivity/network/netstack3/).
Downloaded directories are extracted into fresh staging directories before
replacement, with revision, URL and archive hash recorded beside the source.
`Cargo.lock` pins resolved third-party dependencies. Runtime features omit the
Fuchsia monorepo's global `std`/`serde` unification; host proc macros can use `std`.
The generated core build script uses upstream's `no_lock_order` setting to disable
compile-time lock-order checking in this extraction. Actual synchronization remains
enabled; concurrent lock stress is outside these tests.

## Compiler

`rust-toolchain.toml` pins `nightly-2026-11-01`, Rust 1.101.0. There is no
`RUSTC_BOOTSTRAP` requirement or unstable-feature source patch. The local default
Rust 1.99.1 is too old: this source uses the never type, accepted in Rust 1.100.
Fuchsia's [toolchain manifest](https://fuchsia.googlesource.com/fuchsia/+/650b15d5ea9cfe67faf22ea68cb40e8bc3293303/manifests/toolchain)
selects Rust commit `42cfc04deaa723bc21611f3f9d7c8651be6300df`, whose
[version](https://fuchsia.googlesource.com/third_party/rust/+/42cfc04deaa723bc21611f3f9d7c8651be6300df/src/version)
is 1.101.0 and whose [accepted features](https://fuchsia.googlesource.com/third_party/rust/+/42cfc04deaa723bc21611f3f9d7c8651be6300df/compiler/rustc_feature/src/accepted.rs)
record never-type acceptance in 1.100. The dated toolchain matches that language
requirement; it is not a claim to reproduce Fuchsia's exact compiler build.
The local `.cargo/config.toml` explicitly selects `-Znext-solver=no`. The dated
compiler's default solver remained in recursive auto-trait checking for over two
minutes; the production host check completed with the legacy solver. Compiler
sample and check logs are retained under `target/`. This is a compiler
compatibility setting, not a networking performance result. Standalone dev/test
profiles disable incremental compilation after a compiler fingerprint verification
panic during regeneration; the failure evidence is retained under
`target/compiler-evidence/`.

## Extraction patch

`patches/0001-net-declare-without-fidl.patch` removes the unused generated-FIDL
common-type reexport from `net-declare`; the Cargo translation omits precisely
that FIDL dependency label. Netstack3's IGMP/MLD code needs the native address
macros from this helper. Host token-generating macro implementations remain;
FIDL macros/types are not supported by this extraction.

`patches/0002-bound-conntrack.patch` caps connection tracking at 1,024 entries
and exposes read-only layout diagnostics for resource tests. It checks space
for all new tuple entries before insertion and repeats upstream victim selection
until enough space is available. Established flows retain upstream protection;
if no eligible victim exists, the new entry is dropped. This is an Oscar
resource policy change; protocol algorithms otherwise remain upstream.
Both patches are checked before application and on repeated runs.

## Test boundary

The service admits at most 32 sockets, including reserved passive backlog slots
and closing connections. A listener reserves one to four backlog slots. TCP
buffers hold 32 KiB in each direction. Closing a pending connection aborts it;
an established connection may drain for at most five seconds after its last
control endpoint closes. Loopback aliases, frame queues, timer dispatch batches,
deferred work and connection tracking are bounded. The connection-tracking
budget is 1,024 tuple entries per address family, up to 2,048 in dual-stack mode.
IPv4 and IPv6 fragments are rejected
before core processing; reassembly is outside the first service.

These limits do not turn every upstream allocation into a fallible operation.
Core allocation failure or timer/deferred-table exhaustion can stop the worker.
Clients then see service loss, never TCP EOF. Restart needs fresh service
capabilities and cannot recover established connections. A hard per-client
memory quota and automatic restart remain future work.

Upstream neighbor state can retain up to 1,024 neighbors per device and ten
pending frames per unresolved neighbor. Its loopback queues also have larger
upstream limits. These allocations share the worker's finite heap; they are not
covered by the TCP buffer budget. Broad-prefix configurations and packet pressure
can still exhaust that heap. Socket admission is not a total memory guarantee.

The host harness runs two full cores with upstream fake bindings and simulated
Ethernet links. Its stream wrapper supplies native-shaped `WouldBlock`, closed,
and EOF behavior; it is not an Oscar syscall implementation. It uses deterministic
fake time and seeded test-only randomness. Its packet frontdoor and application
writes have bounds. Fake bindings' internal frames, diagnostic events, timers and
core allocations are still infallible: guard checks after callbacks do not enforce
a hard allocation quota. Allocation counters record requested live payload bytes,
not allocator metadata, RSS or a guaranteed 16 MiB operating envelope.

The production core is `no_std`; the host harness and upstream test helpers use
`std` and host synchronization. `native/` implements independent production
binding traits against Oscar's runtime: logical and atomic time, cancellation,
locks, fixed TCP rings, fallible TX allocation, bounded queues and secure syscall
entropy, including the IPv6 state builder's required SLAAC secret. The proof
creates a fresh secret for each endpoint; persistent production IPv6 identity
policy is outside this test. It instantiates two full cores with simulated links in one native
process. It does not use the host fake bindings or acquire a live NIC.

Build a verified staging copy without launching it:

```sh
rustup target add x86_64-unknown-none --toolchain nightly-2026-11-01
python3 build-native.py
```

The script builds from this directory, strips a copy with `llvm-strip`, checks
ELF64 `ET_EXEC` and x86-64 machine type, and verifies entry/load bytes against the
unstripped original (excluding changed ELF section-table metadata). It writes
`target/native-stage/oscar-netstack3-native` and `artifact.json`. Hardware staging
and launch are separate operations. Native codegen uses sha2's supported scalar
backend because its SIMD backend fails LLVM lowering for the soft-float
freestanding target; this is recorded only in the isolated target config.

The native runner exercises all four TCP rings across a wrap boundary, in
addition to baseline/loss, byte checks, half-close/EOF, TIME-WAIT and teardown.
Listener wakeups are counted/polled by the finite runner; no native client RPC or
readiness protocol is exposed. Timer/deferred-table overflow and core OOM can
terminate this worker; total per-client admission and supervision are not supplied
by this proof.

The simulated proof alone makes no hardware NIC or throughput claim.
The service acceptance results are recorded separately in VALIDATION.md.
A supervisor that enforces a total worker memory budget and recreates service
capabilities remains future work. Restarting loses connections and must report
failure coherently to clients.

## Notices

[FUCHSIA-LICENSE](FUCHSIA-LICENSE) preserves the pinned Fuchsia root BSD license.
Imported files retain their original copyright headers. Third-party dependencies
have their own licenses and notices; the Fuchsia license does not cover them.
Generated dependency/license inventory and copied notices belong under `target/`
when preparing a binary distribution. Run `python3 notices.py` to record the
resolved dependency metadata and copy available source notices under
`target/notices/`; its output identifies packages needing further notice review.
This inventory includes host, test and platform-specific workspace dependencies;
it is not a native-binary SBOM or a completed redistribution audit.
No production distribution is made here.

See [VALIDATION.md](VALIDATION.md) for the tested commands, results and limits.
