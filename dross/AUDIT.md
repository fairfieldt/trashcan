# Dross audit notes

Audit worktree based on bd844274, including the original worktree's uncommitted dross changes.

Fixes:
- Serialize channel close with enqueue to prevent messages and handles being stranded after close.
- Preserve handle numbers on failed channel writes, including concurrent access; add a boot regression test.
- Roll back partially installed page mappings on allocation failure and shoot down stale translations.
- Validate write permissions across an entire mprotect range before changing any mapping.
- Prepare mprotect allocations before changing page tables and restore old entries if installation fails.

Validation: host ACPI, PMM and Ethernet tests; four-CPU QEMU native/Linux boot; QEMU Unix and OpenSSH suite. The Unix suite was rebuilt after the channel changes, before the final mprotect validation change. No physical hardware was tested.

Open audit item: mprotect rollback, the channel close race and page-table allocation failure need targeted concurrent and fault-injection tests. This is not a complete kernel or userland audit.
