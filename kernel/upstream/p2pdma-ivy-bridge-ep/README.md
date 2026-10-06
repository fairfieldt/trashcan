# Ivy Bridge-EP P2PDMA allowlist entry for upstream

Upstream form of `kernel/patches/0003-p2pdma-whitelist-ivy-bridge-ep.patch`. It adds the Xeon E7 v2/Xeon E5
v2/Core i7 DMI2 host bridge (`8086:0e00`) to `pci_p2pdma_whitelist[]` with `REQ_SAME_HOST_BRIDGE`, next to
Sandy Bridge-EP (`3c00`) and Haswell-EP (`2f00`). As of `pci/next` 1ee0e2a03ac it isn't there.

On incite the two D700s are on root ports `00:02.0` (`8086:0e04`) and `00:03.0` (`8086:0e08`) of that bridge,
so without the entry `pci_p2pdma_distance()` fails, `amdgpu_dma_buf_attach` clears `peer2peer`, and dma-buf
imports migrate to GTT.

## Status

- `series/`: one patch on `pci/next` at 1ee0e2a03ac ("Merge branch 'pci/misc'"). checkpatch --strict is clean
  apart from the missing sign-off. Not sent. It carries `Assisted-by` and no `Signed-off-by`; the submitter adds
  their own.
- Recipients (get_maintainer.pl): Bjorn Helgaas, Logan Gunthorpe, linux-pci, linux-kernel.
- The apparent hang during the first evidence run on 2026-10-05 was the GDM greeter suspending incite after 900 s
  idle (`logind: The system will suspend now!`, 22:09:17); incite doesn't resume. Suspend is now disabled
  (`docs/hardware.md`). The rerun under a sleep inhibitor passed five times.
- Only `0e00` is tested. `0e01` (the DMI2 port in PCIe mode, which a second socket's root bus would present) is the
  analogue of the `3c01` entry, but we have no dual-socket Ivy Bridge-EP to test.

## Evidence (2026-10-05, kernel 6.12.111-p2p, IOMMU translated)

`bench/xgpu`, `./xgpu p2p-verify p2p`. In each direction, one GPU imports the other's exported 32 MiB VRAM buffer.
It writes into the buffer and reads from it over P2P, covering the full buffer and 4 random sub-ranges per round
over 16 rounds. The CPU checks every word through host-visible staging. Five runs, 15 GiB checked, all in
`results/kernel/p2pdma-20261005/` (the partial run that hit the suspend is in `suspended-run/`).

```
GPU0 writes/reads GPU1's VRAM: 1536 MiB checked, 0 bad words; export local copy 100.6 GB/s (in VRAM)
GPU1 writes/reads GPU0's VRAM: 1536 MiB checked, 0 bad words; export local copy 99.7 GB/s (in VRAM)
p2p-verify: PASS
```

Bandwidth (`./xgpu p2p-wall`, submit-to-fence wall time minus an empty submit, 1-64 MiB): peer writes 7.9 GB/s,
peer reads 3.7 GB/s. GPU-timestamped copies (`./xgpu p2p`) overstate writes up to about 1 MiB (80 GB/s at 256 KiB,
14-16 GB/s at 1 MiB) because the timestamp lands before Tahiti's 768 KiB L2 writes back. The
stock kernel (no `CONFIG_PCI_P2PDMA`) migrates the same import to GTT at 5-8 GB/s (`results/xgpu/20261002-*`).
That comparison also changes the config, so the effect of the allowlist entry alone rests on reading the code
path above.
