# Dross audit notes

Audit worktree based on bd844274, including the original worktree's uncommitted dross changes.

Fixes:
- Serialize channel close with enqueue to prevent messages and handles being stranded after close.
- Preserve handle numbers on failed channel writes, including concurrent access; add a boot regression test.
- Roll back partially installed page mappings on allocation failure and shoot down stale translations.
- Validate write permissions across an entire mprotect range before changing any mapping.
- Prepare mprotect allocations before changing page tables and restore old entries if installation fails.

Validation: host ACPI, PMM and Ethernet tests; four-CPU QEMU native/Linux boot; QEMU Unix and OpenSSH suite. The Unix suite was rebuilt after the channel changes, before the final mprotect validation change. No physical hardware was tested.

Additional fixes: enforce memory write rights for device DMA and reject oversized DMA device and interrupt CPU indices; reject oversized CPU indices and clean up failed thread creation; remove undisclosed handles after a channel read output fault; validate native and Linux ELF ranges and header arithmetic before allocation, and copy segments without a large temporary buffer; handle truncated HTTP bodies and multiline server events in chat; bound aggregate vectored-write allocations; handle the SSH test completion race; skip zero-length user copies before touching pointers; reject overflowing ancillary-message offsets and overflowing mapping lengths and oversized memfd truncation; reject unaligned mprotect and fixed mappings at address zero.

Run `KERNEL_FEATURES=audit-test PROFILE=release SMP=4 MEM=2G ./run-qemu.sh` for page-table allocation failure injection. It checks partial mapping rollback, mprotect rollback and a subsequent successful retry.

The boot self-test races channel close against handle transfer 512 times on two CPUs. This checks return values and handle ownership; it does not prove queued objects are reclaimed in every interleaving. Allocation failure tests do not cover all allocation sites or cross-CPU TLB behavior. Failed native and Linux spawns reserve and restore the bootstrap handle, with a generation check if process exit cancels a reservation. This is not a complete kernel or userland audit.

Final follow-up for task #1759 (2026-11-03, based on c3f843ea): local and TCP `recvmsg` reject overflowing aggregate iovec lengths and totals above `ssize_t::MAX` with `EINVAL`, before receiving data. Regression tests cover both limits on each path. The TCP test also receives a valid message across two buffers after a zero-length iovec; the local socket's existing fd-passing test verifies a subsequent valid receive.

Validation for this follow-up, on the host with QEMU TCG (no physical hardware):

```sh
cd dross
NO_DRIVERS=1 ./build-linux.sh
SMP=4 MEM=2G TIMEOUT=120 ./run-qemu.sh
PROFILE=release SMP=4 MEM=2G TIMEOUT=120 ./run-qemu.sh
KERNEL_FEATURES=audit-test PROFILE=release SMP=4 MEM=2G TIMEOUT=120 ./run-qemu.sh
KERNEL_FEATURES=unix PROFILE=release SMP=4 MEM=2G TIMEOUT=120 ./run-qemu.sh
cd unix
./build.sh
SMP=4 SSH_PORT=2239 ./run-qemu.sh
SMP=1 SSH_PORT=2239 ./run-qemu.sh
```

All four regular boot suites passed (29 native self-tests, 24 POSIX checks, driver unit tests, edu DMA and USB checks); the audit-test build also passed its page-table allocation-failure rollback tests. Both Unix runs passed 49 guest checks and the full host OpenSSH suite. Follow-up logs are `/tmp/dross-1759-*.log` and the Unix worktree's `target/1759-smp4.log` and `target/1759-smp1.log`.
