/* Userspace shims so the kernel's VFCT lookup functions compile unchanged. */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
typedef uint8_t UCHAR; typedef uint16_t USHORT; typedef uint32_t ULONG;
typedef size_t acpi_size; typedef int acpi_status;
typedef struct { char sig[4]; uint32_t length; uint8_t rest[28]; } AMD_ACPI_DESCRIPTION_HEADER;
struct acpi_table_header { char signature[4]; uint32_t length; uint8_t rest[28]; };
#include "vfct_types.h"
#define ACPI_SUCCESS(s) ((s) == 0)
#define GFP_KERNEL 0
#define PCI_SLOT(devfn) (((devfn) >> 3) & 0x1f)
#define PCI_FUNC(devfn) ((devfn) & 0x07)
struct device { const char *name; };
struct pci_bus { unsigned char number; };
struct pci_dev { unsigned short vendor, device; unsigned int devfn; struct pci_bus *bus; struct device dev; };
struct amdgpu_device { struct pci_dev *pdev; struct device *dev; uint8_t *bios; uint32_t bios_size; };
struct radeon_device { struct pci_dev *pdev; uint8_t *bios; };
extern struct acpi_table_header *g_vfct;
static inline acpi_status acpi_get_table(const char *s, int i, struct acpi_table_header **h) { *h = g_vfct; return g_vfct ? 0 : 1; }
static inline void acpi_put_table(struct acpi_table_header *h) {}
static inline void *kmemdup(const void *p, size_t n, int f) { void *d = malloc(n); if (d) memcpy(d, p, n); return d; }
static inline void kfree(const void *p) { free((void *)p); }
#define dev_log(lvl, d, fmt, ...) printf("    [%s] %s: " fmt, lvl, (d)->name, ##__VA_ARGS__)
#define dev_info(d, ...) dev_log("info", d, __VA_ARGS__)
#define dev_notice(d, ...) dev_log("notice", d, __VA_ARGS__)
#define dev_warn(d, ...) dev_log("warn", d, __VA_ARGS__)
#define dev_dbg(d, ...) do {} while (0)
static inline bool check_atom_bios(struct amdgpu_device *adev, size_t size)
{ return adev->bios && size >= 0x49 && adev->bios[0] == 0x55 && adev->bios[1] == 0xaa; }
static inline void amdgpu_bios_release(struct amdgpu_device *adev) { kfree(adev->bios); adev->bios = NULL; adev->bios_size = 0; }
