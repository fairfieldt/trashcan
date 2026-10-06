/* Compiled once per (driver, old/new); VARIANT_INC is the extracted kernel code. */
#include "shim.h"
#include VARIANT_INC
const char *ENTRY(unsigned bus, unsigned short device)
{
	static char pn[32];
	struct pci_bus b = { .number = bus };
	struct pci_dev pdev = { .vendor = 0x1002, .device = device, .devfn = 0, .bus = &b };
	uint8_t *bios;
	char *p;

	snprintf(pn, sizeof(pn), "%02x:00.0", bus);
	pdev.dev.name = pn;
#ifdef IS_AMDGPU
	struct amdgpu_device adev = { .pdev = &pdev, .dev = &pdev.dev };
	bool ok = amdgpu_acpi_vfct_bios(&adev);
	bios = adev.bios;
#else
	struct radeon_device rdev = { .pdev = &pdev };
	bool ok = radeon_acpi_vfct_bios(&rdev);
	bios = rdev.bios;
#endif
	if (!ok)
		return "(none)";
	p = memmem(bios, 65536, "113-", 4);
	snprintf(pn, sizeof(pn), "%.15s", p ? p : "?");
	free(bios);
	return pn;
}
