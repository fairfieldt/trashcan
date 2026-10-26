# PCI/P2PDMA: Ivy Bridge-EP host bridge (8086:0e00) in the allowlist

Upstream form of `kernel/patches-7.3/0003-p2pdma-whitelist-ivy-bridge-ep.patch`: adds the Xeon E7 v2/Xeon E5 v2/Core i7
DMI2 host bridge (`8086:0e00`) to `pci_p2pdma_whitelist[]` with `REQ_SAME_HOST_BRIDGE`. As of the base below it isn't
there (7.4 added Broadwell-EP 0x6f00 and Haswell client 0x0c08, not 0x0e00). Not sent.

On a Mac Pro 6,1 the two D700s sit on root ports `00:02.0` (`8086:0e04`) and `00:03.0` (`8086:0e08`) of that bridge,
so without the entry `pci_p2pdma_distance()` fails, `amdgpu_dma_buf_attach` clears `peer2peer`, and dma-buf imports
migrate to GTT.

**Status: ready for maintainer review.** New hardware entry for the PCI tree: send after v7.4-rc1.

## Base

`git://git.kernel.org/pub/scm/linux/kernel/git/pci/pci.git` `next` (= `main`) at
17bb0e2f6c8043f70a693a5f3797471cbed354c8 ("Merge tag 'pci-v7.4-changes' ..."), fetched 2026-10-26; also applies to
mainline d135ab453a74.

## Changes since the 2026-10-05 draft

- Rebased: 7.4 added the Broadwell-EP 0x6f00 entry right after 0x3c01, which changed the context. The new entry stays
  after Sandy Bridge-EP (0x3c00/0x3c01), before 0x6f00.
- Commit message names the Broadwell-EP entry; `Assisted-by` removed (see `../README.md`); notes below `---`
  (`notes/1.txt`) give the other kernels tested, the missing same-config A/B and tool use.

## Checks (2026-10-26, kbuild VM)

`../check-series.sh p2pdma-ivy-bridge-ep LOGDIR` (defconfig + `MEMORY_HOTPLUG`, `MEMORY_HOTREMOVE`, `ZONE_DEVICE`,
`PCI_P2PDMA`). Logs in `results/kernel/upstream-20261026/p2pdma-ivy-bridge-ep/`: `drivers/pci/p2pdma.o` with W=1
and sparse, no warnings before or after; checkpatch `--strict` only `MISSING_SIGN_OFF`; get_maintainer: Bjorn Helgaas,
Logan Gunthorpe (PCI PEER-TO-PEER DMA), linux-pci, linux-kernel.

## Hardware tested

- incite, 6.12.111-p2p, IOMMU translated, 2026-10-05: `xgpu p2p-verify`, 5 runs, 15 GiB checked, 0 bad words; peer
  writes 7.9 GB/s, reads 3.7 GB/s (`results/kernel/p2pdma-20261005/`). The hang in the first run was a suspend.
- incite, 7.2.9-p2p: `p2p-verify` 3/3 PASS (`results/kernel/linux-7.2-20261006/`).
- inflight, 7.3.0-rc7-p2p: `p2p-verify` 3/3 PASS, export in VRAM (`results/kernel/linux-7.3-20261014/`).

## Test limits

- Not run on the 7.4 base (a one-line table entry; no hardware run was made for this refresh).
- No config-identical A/B with only this entry toggled: the stock comparison (5-8 GB/s, import migrated to GTT) also
  lacked `CONFIG_PCI_P2PDMA`. The effect of the entry alone rests on the code path above. A same-config A/B would need
  a kernel install and reboot on a Mac, which this refresh did not do.
- `0x0e01` (the DMI2 port in PCIe mode, analogue of 0x3c01) is not added: no dual-socket Ivy Bridge-EP to test.

## Before sending

1. Read the patch and the notes below `---` (first person, for you).
2. Sign off: `perl -0pi -e 's/\n---\n/\nSigned-off-by: Tom Fairfield <tom\@fairfieldt.com>\n---\n/' series/0001-*.patch`
3. `./scripts/checkpatch.pl --strict PATH/series/0001-*.patch` in a kernel tree, then send (`--dry-run` first):

```sh
git send-email --to="Bjorn Helgaas <bhelgaas@google.com>" --to=linux-pci@vger.kernel.org \
  --cc="Logan Gunthorpe <logang@deltatee.com>" --cc=linux-kernel@vger.kernel.org series/0001-*.patch
```
