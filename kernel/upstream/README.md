# Upstream submission queue (refreshed 2026-10-27)

Patches for upstream, one directory per submission. Nothing here has been sent. The maintainer of this repo reviews
each package, adds his own `Signed-off-by`, and sends it; no agent signs off, sends mail or opens a pull request.

Each package has `series/` (the `git format-patch` output to send), `check.conf` (target tree, base commit, config,
objects), a cover letter source (`cover.txt`) or below-the-`---` notes (`notes/N.txt`), and a README with the base,
commands, hardware tested and test limits. `check-series.sh` (run in the kbuild VM) applies a package to its base,
builds the touched objects with W=1 and sparse on the base and after each patch, runs `checkpatch.pl --strict` and
`get_maintainer.pl`, and regenerates `series/`. Logs: `results/kernel/upstream-20261027/` (2026-10-26: `upstream-20261026/`).

| Package | Target tree, base | Status |
|---|---|---|
| [vfct-exact-match](vfct-exact-match/README.md), 2 patches + cover | agd5f `amd-staging-drm-next` ba496bb83788 | Ready for maintainer review (fix; amdgpu on six Macs, radeon booted once on 6.12; amdgpu also on the 7.4 base 2026-10-27). Send now |
| [thunderbolt-stream-tx-e2e](thunderbolt-stream-tx-e2e/README.md), 1 patch | thunderbolt `next` a93a8e320000 | Ready for maintainer review (fix; hardware-tested on 7.2.9, 7.3-rc7 and, 2026-10-27, the 7.4 base: 500/500 sessions with it, 0/7 without). Send now |
| [thunderbolt-stream-hopid](thunderbolt-stream-hopid/README.md), 1 patch | thunderbolt `next` a93a8e320000 | Ready for maintainer review (fix; userspace fixture; ran on the 7.4 base 2026-10-27 without a targeted test). Send with tx-e2e |
| [p2pdma-ivy-bridge-ep](p2pdma-ivy-bridge-ep/README.md), 1 patch | pci `next` 17bb0e2f6c80 | Ready for maintainer review. New hardware entry: send after v7.4-rc1 |
| [applesmc-port-lights](applesmc-port-lights/README.md), 2 patches + cover | hwmon `hwmon-next` e1b3c2630d5b | Ready for maintainer review. New feature: send after v7.4-rc1 |
| [thunderbolt-net-coalesce](thunderbolt-net-coalesce/README.md), 1 patch | netdev `net-next` 734bef548a96 | Ready for maintainer review (replaces local 0004; measured on the 7.4 base). net-next is closed: send after v7.4-rc1 |
| [thunderbolt-stream-close](thunderbolt-stream-close/README.md): a mail, no patch | none | Draft report to Mika: 0006 not needed on 7.4; asks whether 7.3.y wants a fix |
| [thunderbolt-review-queue](thunderbolt-review-queue/README.md): local 0004, 0006, 0007 | none | All resolved 2026-10-27: 0004 replaced by thunderbolt-net-coalesce, 0006 not needed on 7.4, 0007 superseded in mainline |

Checked 2026-10-27 against the trees as fetched that day (only amd-staging-drm-next had moved; vfct-exact-match
applies to its new head unchanged): every package applies to its base and to mainline 8a36f53fd5ec, adds no W=1 or
sparse warnings, and checkpatch `--strict` reports only the missing `Signed-off-by`. thunderbolt-stream-tx-e2e doesn't
apply to thunderbolt `fixes` (96a8479804e3, older than 7.4's RX polling); that changes once both branches are on
v7.4-rc1.

To investigate (not a submission yet): incite's 7.3-rc7 panicked in `tb_handle_hotplug()` -> `kobject_get()` when a
peer whose XDomain properties could not be read disconnected (`results/kernel/usb4stream-close-74-20261027/`).

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
| 0004 thunderbolt_net throttling parameter | thunderbolt-net-coalesce (same purpose, ethtool interface instead of a module parameter) | No: different interface |
| 0005 USB4STREAM no Tx E2E | thunderbolt-stream-tx-e2e | Same change; context differs (7.4's RX polling) |
| 0006 USB4STREAM I/O after CLOSE | thunderbolt-stream-close (not needed on 7.4; 7.3.y question in the mail draft) | n/a |
| 0007 USB4STREAM no RX flush on close | thunderbolt-review-queue (superseded by mainline 20bea8ceeb6a) | n/a |
| 0009 + 0010 applesmc port lights | applesmc-port-lights 1/2 + 2/2 | `applesmc.c` identical (md5 370d44a2); the ABI file's KernelVersion is 7.5 upstream |
| (none) | thunderbolt-stream-hopid | Not deployed locally |

Excluded from the queue:

- **0008** (`kernel/patches-7.3-diag/`, stream hold experiments; its `poll_mode` switch is known to be faulty):
  diagnostic only, applied only with `DIAG=1` under its own release name, never rolled out or sent.
- **0002** (`kernel/patches-7.2/`, SI DPM GPIO DC on mobile parts only): Timur Kristóf's cd883d0a3937, upstream since
  v7.3-rc6, so `patches-7.3/` skips the number.

## Sending (for the maintainer of this repo)

Order: now, the two fixes (vfct-exact-match; thunderbolt-stream-tx-e2e with thunderbolt-stream-hopid) and the
thunderbolt-stream-close mail; after v7.4-rc1 (expected around 2026-11-01), p2pdma-ivy-bridge-ep, applesmc-port-lights
and thunderbolt-net-coalesce, each after `check-series.sh` on the rebased target tree.

Each `series/0*.patch` lacks your `Signed-off-by`. Read the patch and its notes below `---` (first person, yours to
correct), then sign it and check it (GNU sed, e.g. in the kbuild VM):

```sh
cd kernel/upstream/PKG/series
for p in 0*.patch; do
  [ "$p" = 0000-cover-letter.patch ] && continue
  sed -i '0,/^---$/s//Signed-off-by: Tom Fairfield <tom@fairfieldt.com>\n---/' "$p"
done
LINUX/scripts/checkpatch.pl --strict 0*.patch      # LINUX: a kernel tree (the VM's ~/a18/linux)
```

Then (`--dry-run` first; recipients from `get_maintainer.pl` in each check log):

```sh
# vfct-exact-match (amd-gfx; 2 patches + cover)
git send-email --to='Alex Deucher <alexander.deucher@amd.com>' --to='Christian König <christian.koenig@amd.com>' \
  --cc='Mario Limonciello <mario.limonciello@amd.com>' --cc='Lijo Lazar <lijo.lazar@amd.com>' \
  --cc=amd-gfx@lists.freedesktop.org --cc=dri-devel@lists.freedesktop.org --cc=linux-kernel@vger.kernel.org \
  kernel/upstream/vfct-exact-match/series/*.patch

# thunderbolt-stream-tx-e2e and thunderbolt-stream-hopid (two separate single patches, same recipients)
for p in tx-e2e hopid; do
  git send-email --to='Mika Westerberg <westeri@kernel.org>' \
    --cc='Andreas Noever <andreas.noever@gmail.com>' --cc='Yehezkel Bernat <YehezkelShB@gmail.com>' \
    --cc='Alan Borzeszkowski <alan.borzeszkowski@linux.intel.com>' \
    --cc=linux-usb@vger.kernel.org --cc=linux-kernel@vger.kernel.org \
    kernel/upstream/thunderbolt-stream-$p/series/0001-*.patch
done

# after v7.4-rc1:
git send-email --to='Bjorn Helgaas <bhelgaas@google.com>' --cc='Logan Gunthorpe <logang@deltatee.com>' \
  --cc=linux-pci@vger.kernel.org --cc=linux-kernel@vger.kernel.org \
  kernel/upstream/p2pdma-ivy-bridge-ep/series/0001-*.patch
git send-email --to='Guenter Roeck <linux@roeck-us.net>' --cc='Henrik Rydberg <rydberg@bitmath.org>' \
  --cc=linux-hwmon@vger.kernel.org --cc=linux-kernel@vger.kernel.org \
  kernel/upstream/applesmc-port-lights/series/*.patch
sed -i 's/^Subject: \[PATCH\]/Subject: [PATCH net-next]/' kernel/upstream/thunderbolt-net-coalesce/series/0001-*.patch
git send-email --to='Mika Westerberg <westeri@kernel.org>' --to='Yehezkel Bernat <YehezkelShB@gmail.com>' \
  --cc='Andrew Lunn <andrew+netdev@lunn.ch>' --cc='David S. Miller <davem@davemloft.net>' \
  --cc='Eric Dumazet <edumazet@kernel.org>' --cc='Jakub Kicinski <kuba@kernel.org>' --cc='Paolo Abeni <pabeni@redhat.com>' \
  --cc=netdev@vger.kernel.org --cc=linux-kernel@vger.kernel.org \
  kernel/upstream/thunderbolt-net-coalesce/series/0001-*.patch
```

The thunderbolt-stream-close mail (`thunderbolt-stream-close/mail-draft.txt`) is plain text for your mail client.
