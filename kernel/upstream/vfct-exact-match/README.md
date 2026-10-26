# drm/amdgpu, drm/radeon: prefer an exact bus match for VFCT images

Upstream form of `kernel/patches-7.3/0001-amdgpu-vfct-exact-match-first.patch` (amdgpu), plus the same change for
radeon. Not sent.

Mainline's soft filter, db7e8108809a (amdgpu, v7.2-rc4; cherry-picked again as 11c141672045 for v7.3-rc1) and
4059e2f02c8a (radeon, v7.3-rc1), "Fix VFCT bus number matching with soft filter", takes the first VFCT image whose
vendor, device, slot and function match. Still so in mainline and amd-staging-drm-next on 2026-10-26. A Mac Pro 6,1's
VFCT has two entries that differ only in the bus number:

| entry | bus | id | VBIOS |
|---|---|---|---|
| 0 | 02:00.0 | 1002:6798 | 113-C3861XA-028 |
| 1 | 06:00.0 | 1002:6798 | 113-C3861XB-028 |

so the GPU at 06:00.0 gets the XA image. The series prefers an exact bus match and falls back to the first identity
match only when no bus matches.

**Status: ready for maintainer review.** A fix with `Cc: stable`; amdgpu tested on hardware on three kernel
generations, radeon booted once on a 6.12 backport (see limits).

## Base

`https://gitlab.freedesktop.org/agd5f/linux.git` `amd-staging-drm-next` at
77f548b127c6de34ce271fa09ba270cb1ef9f2da ("drm/amdgpu: keep PRT mappings off the VM state lists again"), fetched
2026-10-26. Also applies to mainline d135ab453a74. amd-staging has radeon's VBIOS signature check in the VFCT path
(0795596911b3); 2/2 keeps it.

## Changes since the 2026-10-05 draft

- Rebased on the current amd-staging-drm-next (applies unchanged).
- amdgpu `Fixes:` now names db7e8108809a, the commit that first brought the soft filter into mainline (v7.2-rc4), so
  stable picks the right range; 11c141672045 is its cherry-pick in v7.3-rc1. Cover letter updated to match.
- `Assisted-by` removed (see `../README.md`); cover letter has the tool use and the current testing.

## Checks (2026-10-26)

- `../check-series.sh vfct-exact-match LOGDIR`: `amdgpu_bios.o` and `radeon_bios.o` with W=1 and sparse, with and
  without `CONFIG_ACPI`, on the base and after each patch (defconfig + `DRM`, `DRM_AMDGPU=m`, `DRM_AMDGPU_SI`,
  `DRM_RADEON=m`). No new warnings; the four sparse `dereference of noderef expression` warnings in
  `radeon_bios.c:96-99` are in the base. checkpatch `--strict`: `MISSING_SIGN_OFF`, and 8 `CAMELCASE` checks on 1/2 for
  the atombios structure fields (`VendorID`, `PCIBus`, ...), which can't be renamed here. Logs:
  `results/kernel/upstream-20261026/vfct-exact-match/`.
- `harness/run.sh TREE OLD_REV VFCT.bin` (the old and new `*_acpi_vfct_match()` .. `*_acpi_vfct_bios()` compiled
  unchanged in userspace and fed incite's VFCT): run on the rebased tree against the base; the old code gives
  06:00.0 the XA image under both drivers, the new code XB; single-GPU exact and renumbered tables, both buses
  renumbered, and a non-matching device ID behave as before. `PASSED` (`vfct-replay.txt`). The VFCT dump
  (`sudo cat /sys/firmware/acpi/tables/VFCT` on incite, sha256 `4ff79815...`) contains the VBIOS images and is not
  tracked.
- get_maintainer: Alex Deucher, Christian König (RADEON and AMDGPU), David Airlie, Simona Vetter (DRM), commit
  signers, amd-gfx, dri-devel, linux-kernel.

## Hardware tested

- incite, 6.12.111 backports as `M=` modules from a separate initramfs, 2026-10-05: amdgpu gave 02:00.0 XA and 06:00.0
  XB, no mismatch notice, both initialized; radeon initialized both with no VFCT warning or mismatch notice
  (`results/kernel/vfct-20261005/`).
- amdgpu change on 7.2.9 and 7.3-rc7 (`kernel/patches-7.3/0001`; its `amdgpu_acpi_vfct_match()`/`_bios()` are
  identical to this series' after applying). On 2026-10-26 all six Macs (7.3.0-rc7-p2p-s2) report `vbios_version`
  113-C3861XA-028 for 02:00.0 and 113-C3861XB-028 for 06:00.0 (read-only sysfs check).

## Test limits

- radeon: one boot of the 6.12.111 backport on one Mac; radeon doesn't log its VBIOS source, so that it used VFCT is
  inferred (both drivers try ATRM first, and amdgpu fell through to VFCT on this machine). Not run on a current
  kernel. Don't hot-load radeon to retest: an `insmod` into a running system hung incite.
- The two-identical-GPUs-both-renumbered case still gets the first image for both, as before (stated in the cover).
- Only Apple's VFCT from the Mac Pro 6,1.

## Before sending

1. Read the patches and the cover letter (the "Testing" and "How this was made" paragraphs speak for you).
2. Sign off: `perl -0pi -e 's/\n---\n/\nSigned-off-by: Tom Fairfield <tom\@fairfieldt.com>\n---\n/' series/000[1-9]*.patch`
3. `./scripts/checkpatch.pl --strict PATH/series/000[1-9]*.patch` in a kernel tree (expect only the CAMELCASE
   checks), then send (`--dry-run` first). The `Cc: stable@vger.kernel.org` tags make send-email copy stable too;
   add `--suppress-cc=bodycc` if you'd rather it didn't.

```sh
git send-email --to="Alex Deucher <alexander.deucher@amd.com>" --to="Christian König <christian.koenig@amd.com>" \
  --to=amd-gfx@lists.freedesktop.org --cc=dri-devel@lists.freedesktop.org \
  --cc="Mario Limonciello <mario.limonciello@amd.com>" --cc="Oz Tiram <oz@shift-computing.de>" \
  --cc=linux-kernel@vger.kernel.org series/*.patch
```

Mario Limonciello wrote the soft filter; Oz Tiram reported the pci=realloc case it was added for.
