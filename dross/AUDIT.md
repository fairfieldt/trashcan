# Dross audit notes

Audit worktree based on bd844274, including the original worktree's uncommitted dross changes.

Fixes:
- Serialize channel close with enqueue to prevent messages and handles being stranded after close.
- Preserve handle numbers on failed channel writes, including concurrent access; add a boot regression test.
- Roll back partially installed page mappings on allocation failure and shoot down stale translations.
- Validate write permissions across an entire mprotect range before changing any mapping.
- Prepare mprotect allocations before changing page tables and restore old entries if installation fails.

Validation: host ACPI, PMM and Ethernet tests; four-CPU QEMU native/Linux boot; QEMU Unix and OpenSSH suite. The Unix suite was rebuilt after the channel changes, before the final mprotect validation change. No physical hardware was tested.

Additional fixes: bound aggregate vectored-write allocations; handle the SSH test completion race; skip zero-length user copies before touching pointers; reject overflowing ancillary-message offsets and overflowing mapping lengths and oversized memfd truncation; reject unaligned mprotect and fixed mappings at address zero.

Run `KERNEL_FEATURES=audit-test PROFILE=release SMP=4 MEM=2G ./run-qemu.sh` for page-table allocation failure injection. It checks partial mapping rollback, mprotect rollback and a subsequent successful retry.

Open audit item: the channel close race still needs targeted concurrent stress; allocation failure tests do not cover all allocation sites or cross-CPU TLB behavior. This is not a complete kernel or userland audit.
