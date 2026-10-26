# Upstream submission queue (refreshed 2026-10-26)

Patches for upstream, one directory per submission. Nothing here has been sent. The maintainer of this repo reviews
each package, adds his own `Signed-off-by`, and sends it; no agent signs off, sends mail or opens a pull request.

Each package has `series/` (the `git format-patch` output to send), `check.conf` (target tree, base commit, config,
objects), a cover letter source (`cover.txt`) or below-the-`---` notes (`notes/N.txt`), and a README with the base,
commands, hardware tested and test limits. `check-series.sh` (run in the kbuild VM) applies a package to its base,
builds the touched objects with W=1 and sparse on the base and after each patch, runs `checkpatch.pl --strict` and
`get_maintainer.pl`, and regenerates `series/`. Logs: `results/kernel/upstream-20261026/`.

| Package | Target tree, base | Status |
|---|---|---|
| [applesmc-port-lights](applesmc-port-lights/README.md), 2 patches + cover | hwmon `hwmon-next` e1b3c2630d5b | Ready for maintainer review. New feature: send after v7.4-rc1 |
| [p2pdma-ivy-bridge-ep](p2pdma-ivy-bridge-ep/README.md), 1 patch | pci `next` 17bb0e2f6c80 | Ready for maintainer review. New hardware entry: send after v7.4-rc1 |
| [thunderbolt-stream-tx-e2e](thunderbolt-stream-tx-e2e/README.md), 1 patch | thunderbolt `next` a93a8e320000 | Ready for maintainer review (fix; hardware-tested on 7.2.9 and 7.3-rc7, not on this base) |
| [thunderbolt-stream-hopid](thunderbolt-stream-hopid/README.md), 1 patch | thunderbolt `next` a93a8e320000 | Ready for maintainer review (fix; userspace fixture only, no hardware run) |
| [vfct-exact-match](vfct-exact-match/README.md), 2 patches + cover | agd5f `amd-staging-drm-next` 77f548b127c6 | Ready for maintainer review (fix; amdgpu on six Macs, radeon booted once on 6.12) |
| [thunderbolt-review-queue](thunderbolt-review-queue/README.md): local 0004, 0006, 0007 | none | Not for sending: 0007 is superseded in mainline, 0006 conflicts with mainline's CLOSE semantics, 0004 needs an API decision |

All target branches are contained in mainline d135ab453a74 (7.4 merge window, 2026-10-26); each package also applies
there. When a target tree is rebased after v7.4-rc1, fetch it in the VM, update `BASE_COMMIT` in `check.conf` and
rerun `check-series.sh` before sending.

## AI-assistance attribution

The `Assisted-by:` tag was dropped from the kernel's process documents and from checkpatch by 1e035297ffda ("docs:
remove Assisted-by references", in mainline for 7.4). Maintainers asked for tool use to be described in the cover
letter instead; `Documentation/process/generated-content.rst` asks for transparency in cover letters and changelogs.
So these patches carry no `Assisted-by` trailer; each cover letter (or, for single patches, the notes below `---`)
has a "How this was made" paragraph. Those paragraphs and the test claims are written in the first person for the
submitter: read and correct them before signing off. `coding-assistants.rst` still says AI agents must not add
`Signed-off-by`; none is present.

## Deployable patches and the submission queue

`kernel/build-7.3.sh` applies `kernel/patches-7.3/` (the kernel on all six Macs, `7.3.0-rc7-p2p-s2`, plus the applesmc
module override). Every deployable patch maps to one entry here:

| Local patch | Upstream package | Same code? |
|---|---|---|
| 0001 amdgpu VFCT exact match | vfct-exact-match 1/2 (2/2 is the radeon counterpart, not deployed: the Macs use amdgpu) | `amdgpu_acpi_vfct_match()`/`_bios()` identical after applying each |
| 0003 P2PDMA 8086:0e00 | p2pdma-ivy-bridge-ep | Same entry; placed before the 0x6f00 entry added for 7.4 |
| 0004 thunderbolt_net throttling parameter | thunderbolt-review-queue (not for sending as is) | n/a |
| 0005 USB4STREAM no Tx E2E | thunderbolt-stream-tx-e2e | Same change; context differs (7.4's RX polling) |
| 0006 USB4STREAM I/O after CLOSE | thunderbolt-review-queue (conflicts with mainline 8d55710a8c43) | n/a |
| 0007 USB4STREAM no RX flush on close | thunderbolt-review-queue (superseded by mainline 20bea8ceeb6a) | n/a |
| 0009 + 0010 applesmc port lights | applesmc-port-lights 1/2 + 2/2 | `applesmc.c` identical (md5 370d44a2); the ABI file's KernelVersion is 7.5 upstream |
| (none) | thunderbolt-stream-hopid | Not deployed locally |

Excluded from the queue:

- **0008** (`kernel/patches-7.3-diag/`, stream hold experiments; its `poll_mode` switch is known to be faulty):
  diagnostic only, applied only with `DIAG=1` under its own release name, never rolled out or sent.
- **0002** (`kernel/patches-7.2/`, SI DPM GPIO DC on mobile parts only): Timur Kristóf's cd883d0a3937, upstream since
  v7.3-rc6, so `patches-7.3/` skips the number.
