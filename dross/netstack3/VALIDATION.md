# Validation record

Source: Fuchsia `650b15d5ea9cfe67faf22ea68cb40e8bc3293303`.
Compiler: `rustc 1.101.0-nightly (f9ceb4a4d 2026-10-31)`, dated toolchain
`nightly-2026-11-01`, with `-Znext-solver=no`. Incremental compilation is disabled
in the standalone dev/test profiles after a reproducible compiler fingerprint
verification panic following regeneration. The extraction now carries an Oscar connection-tracking resource policy patch;
see README.md and patches/0002-bound-conntrack.patch.

Run from `dross/netstack3`; this selects its toolchain/config and inherits Oscar's
freestanding static relocation/code-model flags. Python preparation is pinned to
the same source revision for all imported directories. A fresh preparation in
`/tmp/oscar-netstack3-clean-proof` also succeeded; the fetched source is replaced
from fresh staging directories and cannot retain removed files from an older pin.

## Service integration, 2026-11-03

The service replaces the smoltcp implementation. The binary remains named
`unix-netd` for existing boot configurations; it is built from package
`oscar-netd` in this workspace. The old crate and its dependency closure have
been removed from the Unix workspace. No kernel ABI changed.

| Check | Result |
| --- | --- |
| Complete `python3 dross/unix/build.py` | Pass, including the separate pinned-toolchain service build |
| Native engine under Oscar, four CPUs | Pass: learned ARP, lost SYN/data, bounded partial I/O, peek, half-close, EOF, 127/8 and self-address routes, admission, default gateway, link loss, fragments, pending cancellation and rejected shutdown |
| Connection tracking pressure | 1,500 SYN flows; 1,024 retained tuple entries, 988 limit hits, no binding-budget fault |
| Strict mixed self/normal tuple budget | Both IPv4/IPv6 regressions fail the old check at 1,025 entries and pass the new pre-insert check |
| Upstream filter unit suite after the policy patch | 676 passed |
| Preserved host proof tests | 5 passed |
| Native SDK and ownership tests | 84 native checks on one CPU, 89 on four; 28 POSIX checks in each cohort |
| Independent C/Rust network codecs | 1,534 comparisons passed |
| Existing framing/bootstrap codecs | 267 comparisons passed |
| Full Unix and OpenSSH suite | Pass on one and four CPUs |

The native daemon fixture runs against the same engine as Linux sockets, then
against a second daemon with no Linux factory or network registration. It checks
connect/listen grants, IPv6 Unsupported, readiness registration/rearm/cancel,
observer capability rejection without killing other clients, 96 KiB plus 17 bytes
of exact binary data, partial I/O, FIN/EOF, sending after peer FIN, accepted-stream
rights attenuation and abort. The second daemon exits after its fake Ethernet
endpoint closes.

The SSH tests authenticate with fixed test keys, upload and download 8 MiB with
hash checks, test ENOSPC, PTY input/resize, Ctrl-Z/foreground/Ctrl-C, rejected keys,
exit status and reconnect cleanup. QEMU uses TCG on macOS with an emulated e1000;
its host forwarding binds Pinkerton's tailnet address. The final one-CPU run
measured 0.81 MiB/s upload and 5.13 MiB/s download; four CPUs measured 0.79 and
7.13 MiB/s. These are emulation observations, not hardware results or a controlled
comparison with the previous stack. No GPU work ran; GPU clocks do not apply.

The final native engine test is SHA-256
`2d9d91658d4f1c5fc750c5705b3aee827c124d2d1f9070a9ee23a4dad82a34a9`,
7,904,008 bytes. The service is SHA-256
`e3fa634d67b7d1a63efd9e18cbfb98128f2fb2cac17eb9ee180453b7f176408b`,
3,817,608 bytes before staging strip. Logs, source hashes and artifact manifests
are archived at repository-root `results/dross/netstack3-service-20261103/`,
outside this branch. The engine cohort also exercises normal init's
`native-check` command and passes 89 native checks with zero failures.

Reproduce the service acceptance from the repository root:

```sh
python3 dross/unix/build.py
SMP=1 python3 dross/unix/test.py
SMP=4 python3 dross/unix/test.py
python3 dross/protocol/tests/check.py
python3 dross/protocol/tests/check_network.py
```

Physical acceptance also passed on enlight, a Mac Pro 6,1 with 24 logical CPUs.
The BCM57762 driver supplied real Ethernet frames over OETH. The same service
passed 89 native OS checks, the native TCP fixture including observer isolation,
and 49 Unix probes plus the full remote OpenSSH suite. The 8 MiB hash-checked transfer
measured 2.26 MiB/s upload and 10.69 MiB/s download. This is the new hardware
baseline; it is not a controlled before/after comparison. The hardware cohort did
not separately launch the complete 28-test POSIX program; that result applies
to the QEMU cohorts above.

The hardware image is SHA-256
`2b1cb96c47737d9d22cb0f2c97f7ecaf9ff6eb5a7e759e8d217a2ee1eb67b1bd`.
Its stripped `unix-netd` is 2,246,456 bytes, SHA-256
`36ecaee0f0c6f7636c8e5adedd8dc44ae105a05100dc47d33107dd38ed3b3023`.
Logs, exact input/source hashes, ELF checks and SSH transcripts are archived at
`results/dross/enlight-netstack3-service-20261103/`. The kernel is unchanged from
the earlier native proof. The finite boot reported all checks passed and requested
a clean reset. Linux returned with its original BootOrder, no BootNext and no
temporary EFI entry or directory. The capture helper was stopped and removed,
and the hardware lease was released.

These checks do not establish graceful recovery from every upstream allocation
failure, a complete per-client memory quota, automatic restart, IPv6/UDP service
support or DNS resolution. See README.md for the worker's resource boundaries.

## Earlier extraction proof

The following records the frozen proof before service integration. Its hashes
identify those archived artifacts; current builds also include the service and
resource-policy changes described above.

| Check | Result | Evidence |
| --- | --- | --- |
| Production core host compile | Pass | `target/core-check-final.log` |
| Production core `x86_64-unknown-none` compile | Pass | `target/freestanding-check.log` |
| Portable upstream unit suite | 3,278 passed; no failures or ignored tests | `target/upstream-tests.log` |
| Host proof tests | 5 passed | `target/proof-tests-final.log` |
| Host proof executable | Baseline, lost SYN and lost data pass | `target/proof-run.log` |
| Own native bindings and instantiated two-core runner, freestanding check | Pass | `target/native-check.log` |
| Native codegen/link and stripped ELF validation | Pass; ET_EXEC, x86-64 | `target/native-artifact.log` |
| Native worker under Oscar in QEMU | All three scenarios pass; child exit 0 | `target/qemu-pass/d47f1d6606ef9b3e381d61c04f55c04132598596d7b39b0c79596b54eb987381/qemu.log` |
| Native worker on enlight Mac Pro 6,1 | All three scenarios pass; child exit 0 | `results/dross/enlight-netstack3-20261103/acceptance.json` and `netlog.txt` (repository-root paths) |

The upstream suite selects base, core, datagram, device, filter, ICMP echo, IP,
TCP and UDP unit targets. It does not include Fuchsia services, driver/netemul
integration, Loom, fuzz campaigns or every upstream test target.

## Reproduction

```sh
python3 prepare.py
cargo check -p netstack3_core --locked
cargo check -p netstack3_core --target x86_64-unknown-none --locked
cargo test -p oscar-netstack3-proof --locked
cargo run -p oscar-netstack3-proof --locked
cargo test -p netstack3_core -p netstack3_base -p netstack3_tcp -p netstack3_ip -p netstack3_device -p netstack3_filter -p netstack3_datagram -p netstack3_udp -p netstack3_icmp_echo --lib --features testutils --locked
python3 notices.py
```

Install the target once if necessary:
`rustup target add x86_64-unknown-none --toolchain nightly-2026-11-01`.
Build the isolated native worker separately from the host harness:

```sh
python3 build-native.py
```

It produces `target/native-stage/oscar-netstack3-native` (7,546,296 file bytes,
ELF64 little-endian, ET_EXEC 2, machine 62, entry `0x927d50`), SHA-256
`d47f1d6606ef9b3e381d61c04f55c04132598596d7b39b0c79596b54eb987381`.
Its PT_LOAD memory-size fields sum to 7,549,130 bytes; that is a declared ELF
mapping size, not measured process memory. Native codegen requires the supported
`sha2_backend="soft"` configuration on this soft-float target. Compilation,
linking and stripping are verified. The corrected artifact passed native
execution under Oscar in QEMU and on the physical enlight Mac Pro 6,1.

The first native QEMU execution (artifact `61cc95a2…`) entered `main` and failed
because the production IPv6 state builder requires an explicit stable SLAAC
secret, even when this runner enables only IPv4. The runner now supplies a new
secret for each isolated endpoint through its native entropy-backed `RngContext`.
No assertion or core feature was disabled. The failed binary, checksum record,
runner source and QEMU log are preserved under
`target/failed-native/61cc95a20b5c92e4a50817ec29ccd73db4e018f159d78f9e3559b5bb0a57fdd6/`.
The root agent confirmed the replacement QEMU run: baseline, lost SYN and lost
data all passed, each reporting `ring_wraps: [1, 1, 1, 1]`. Lost data reported one
data retransmission. The final worker PASS line was followed by `native-run`
exit `Ok(0)`; the same boot passed 80 native OS tests and 28 POSIX tests.

The QEMU root filesystem contains exactly the corrected worker checksum above.
The cohort kernel SHA-256 was
`9d698dcda8595208c7c4b3f5fd8c06dd9a140975553ed145ef9080b2eaca69f5`,
and rootfs SHA-256 was
`261d49313e41245bc5b7a2dd93ad133fb7b35ac2ecae5a6a2f6c59216644f1b9`.
The saved command uses an IvyBridge CPU under TCG, 2 GiB RAM and `-net none`;
no hardware NIC or host TCP listener is used. Logs, command and cohort manifest
are preserved under the QEMU evidence directory in the table.

The root agent also confirmed the same worker, kernel, launcher and rootfs on
`enlight.tdf.sh`, a Mac Pro 6,1 with 24 logical CPUs. Physical reports matched the
QEMU reports below, including all four ring wraps, final PASS and child exit
`Ok(0)`. The physical boot also passed 80 native OS tests; its configuration did
not run the 28-test POSIX suite, so the POSIX result above applies to QEMU only.
Hardware source inputs, manifest, log and acceptance JSON are archived at the
repository-root path `results/dross/enlight-netstack3-20261103/` outside this
branch. The launcher was `c36783fd8808cec4a903cadba20faa7a5c974ac3f684a797953c76b667593f04`.

| Native scenario, both QEMU and physical | Frames | Timers | Client SYNs | Injected drops | Data retransmissions | Ring wraps |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | 63 | 4 | 1 | 0 | 0 | `[1, 1, 1, 1]` |
| Lost SYN | 64 | 5 | 2 | 1 SYN | 0 | `[1, 1, 1, 1]` |
| Lost data | 57 | 5 | 1 | 1 data | 1 | `[1, 1, 1, 1]` |

Both executions used an empty `BootstrapV2` child with no NIC or system
capabilities. Ethernet links and time are simulated within the native process;
entropy comes from real Oscar syscalls. These results validate native core and
binding execution, not a hardware NIC, native network service or POSIX socket
bridge. The 24-CPU machine result is not multithreaded lock-stress coverage.

## Host workload resource observations

The serialized executable counted allocator-requested live bytes for both cores,
the fake bindings, buffers and packet queue. It excludes allocator metadata, RSS,
kernel resources and any proof of graceful OOM behavior.

| Scenario | Live before / after | Peak requested live bytes | Packet high-water | Timer high-water per endpoint |
| --- | --- | --- | --- | --- |
| Baseline | 612 / 612 | 414,886 | 10 | 1 |
| Lost SYN | 1,700 / 1,700 | 415,974 | 10 | 1 |
| Lost data | 1,700 / 1,700 | 415,974 | 3 | 1 |

The executable uses two endpoints, one listener/backlog 1 and two connected
streams. TCP receive/send admission is 16 KiB per direction; the host frontdoor
has 64 slots of at most 2,048 bytes, with 256 observed-timer and 16,384 event-step
guards. These are small workload observations, not a general 16 MiB operating
guarantee. TIME-WAIT advances through logical time without wall-clock sleeps.

Dropping the first SYN caused two client SYN emissions and an extra timer.
Dropping a single client data segment caused a timer and one data retransmission.
The core's TCP-info retransmission counter does not retain handshake retransmits,
so SYN recovery is asserted from emitted packets and fired timers instead.

The native worker uses independent production binding implementations and no
`FakeBindingsCtx` or `testutils`. Its active timer table and frame queues are
bounded, with fixed-capacity TCP rings; infallible passive-open/core allocations
and timer-budget exhaustion can still terminate the worker. It is a finite local
simulation, not a network service, native wait/readiness implementation, NIC
adapter or supervised recovery implementation. No networking performance claim
is made.

## Final extraction and notice review

Repeated `prepare.py --offline` generated byte-identical manifests, and the
subsequent native rebuild/strip reproduced the frozen `d47f1d66…` artifact
exactly (`target/reproducibility-review.json`). A clean online extraction had
also succeeded independently. Imported source, generated manifests, binaries and
caches are ignored under `target/`; `.cargo/config.toml` is reviewable source and
must be included with the other workspace files.

The notice inventory covers 124 external workspace packages, including host,
test and platform-specific dependencies. A native dependency-tree cross-check
found all 46 external versions in its normal/build closure represented in that
inventory, including host build macros (`target/license-inventory-scope.json`).
Available source notices are copied; packages with missing notice files are
reported for manual review. This is not a native-only binary SBOM or a completed
redistribution audit. The pinned Fuchsia root license is preserved separately.
