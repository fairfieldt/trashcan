#include "shim.h"
struct acpi_table_header *g_vfct;
const char *amdgpu_old(unsigned, unsigned short), *amdgpu_new(unsigned, unsigned short);
const char *radeon_old(unsigned, unsigned short), *radeon_new(unsigned, unsigned short);
static const struct { const char *name; const char *(*fn)(unsigned, unsigned short); } v[] = {
	{ "amdgpu old", amdgpu_old }, { "amdgpu new", amdgpu_new },
	{ "radeon old", radeon_old }, { "radeon new", radeon_new },
};
static uint8_t *load(const char *path, size_t *n)
{
	FILE *f = fopen(path, "rb"); static uint8_t buf[1 << 20];
	*n = fread(buf, 1, sizeof(buf), f); fclose(f);
	uint8_t *d = malloc(*n); memcpy(d, buf, *n); return d;
}
static int fails;
static void run(const char *what, unsigned bus, unsigned short dev, const char *expect)
{
	printf("%s: device at %02x:00.0 (1002:%04x), expect %s\n", what, bus, dev, expect);
	for (unsigned i = 0; i < 4; i++) {
		const char *got = v[i].fn(bus, dev);
		bool isnew = strstr(v[i].name, "new");
		bool pass = !strcmp(got, expect);
		if (isnew && !pass) fails++;
		printf("  %-10s -> %-15s %s\n", v[i].name, got, pass ? "ok" : (isnew ? "FAIL" : "wrong (old code)"));
	}
}
int main(int argc, char **argv)
{
	size_t n; uint8_t *real = load(argv[1], &n);
	g_vfct = (void *)real;
	run("Mac Pro table", 0x02, 0x6798, "113-C3861XA-028");
	run("Mac Pro table", 0x06, 0x6798, "113-C3861XB-028");
	run("Mac Pro table, both buses renumbered", 0x03, 0x6798, "113-C3861XA-028");
	run("Mac Pro table, both buses renumbered", 0x07, 0x6798, "113-C3861XA-028");
	run("Mac Pro table, other device ID", 0x02, 0x679a, "(none)");

	/* Single-GPU table (entry 0 + terminator): the pci=realloc case the soft filter was added for. */
	size_t e0 = 76 + 28 + 65536, len = e0 + 28;
	uint8_t *one = calloc(1, len);
	memcpy(one, real, e0);
	((struct acpi_table_header *)one)->length = len;
	g_vfct = (void *)one;
	run("single-GPU table, exact bus", 0x02, 0x6798, "113-C3861XA-028");
	run("single-GPU table, renumbered bus", 0x05, 0x6798, "113-C3861XA-028");
	printf("\n%s (%d failures in new code)\n", fails ? "FAILED" : "PASSED", fails);
	return !!fails;
}
