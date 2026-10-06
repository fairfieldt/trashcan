# VFCT exact-match fix for upstream

Upstream form of `kernel/patches/0001-amdgpu-vfct-exact-match-first.patch`, for amdgpu and radeon.

Commits 11c141672045 (amdgpu) and 4059e2f02c8a (radeon), "Fix VFCT bus number matching with soft filter",
went into mainline in July 2026 and were backported to Debian's 6.12.111. They take the first VFCT image
whose vendor, device, slot and function match. incite's Apple VFCT has two entries that differ only in the
bus number:

| entry | bus | id | VBIOS |
|---|---|---|---|
| 0 | 02:00.0 | 1002:6798 | 113-C3861XA-028 |
| 1 | 06:00.0 | 1002:6798 | 113-C3861XB-028 |

So the GPU at 06:00.0 gets the XA image. The series prefers an exact bus match and only falls back to the
first identity match when no bus matches. `kernel/patches/0001` is the same change backported to Debian 6.12.111
(it replaces the earlier two-pass version, which `kernel/orig/` still records).

## Status

- `series/`: rebased on `amd-staging-drm-next` at 53750a937 ("drm/amdgpu: reset VI ASIC on MacBookPro14,3").
  It is not sent yet. The patches carry `Assisted-by` and no `Signed-off-by`
  (Documentation/process/coding-assistants.rst); the submitter adds their own sign-off.
- Recipients (get_maintainer.pl plus the authors of the fixed commits): Alex Deucher, Christian König,
  amd-gfx, dri-devel; Cc Mario Limonciello (author) and Oz Tiram (reported the pci=realloc case).

## Evidence (2026-10-05)

- **Replay harness** (`harness/run.sh`): compiles the old and new `*_acpi_vfct_bios()` unchanged in
  userspace and feeds them incite's VFCT. With the old code, 06:00.0 gets XA under both drivers. With the new
  code it gets XB. Single-GPU tables (exact bus and renumbered bus) and a non-matching device ID behave as before.
  To dump the table on incite, run `sudo cat /sys/firmware/acpi/tables/VFCT`. The dump isn't tracked because it
  contains the VBIOS images.
- **Build:** `amdgpu_bios.o` and `radeon_bios.o` build with W=1 with and without CONFIG_ACPI. checkpatch
  --strict reports only the missing sign-off, the CamelCase atombios field names, and unknown commit IDs
  (shallow clone).
- **Hardware:** the 6.12.111 backports were built as modules with `M=` and booted from a separate initramfs
  through one-shot GRUB entries. Logs are in `results/kernel/vfct-20261005/`.
  - amdgpu-fix: `02:00.0` gets `113-C3861XA-028` and `06:00.0` gets `113-C3861XB-028` (`vbios_version`). There's
    no mismatch notice, and both initialize.
  - radeon-new: both initialize, with no VFCT warning and no mismatch notice. radeon doesn't log its VBIOS
    source, but both drivers try ATRM then VFCT, and amdgpu took VFCT on this machine.
  - Hot-loading radeon with `insmod` into a running multi-user boot (no GPU driver at boot) hung incite hard
    on the first load. Test drivers at boot instead. That boot's log was lost because journald is volatile
    (FreedomBox plinth switched it on 2026-10-02).
