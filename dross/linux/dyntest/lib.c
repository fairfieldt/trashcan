// libdrosstest.so, which dyntest dlopens: code, a data relocation, libc and thread-local storage.
#include <stdio.h>

static const char *names[] = {"dross"};
static __thread int calls = 7;

int drosstest_add(int a, int b) { return a + b; }

const char *drosstest_name(void) { return names[0]; }

int drosstest_format(char *buf, int n, int v) { return snprintf(buf, n, "v=%d", v); }

int drosstest_calls(void) { return calls++; }
