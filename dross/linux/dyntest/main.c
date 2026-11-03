// A dynamically linked program, started through musl's dynamic linker: checks the auxiliary vector it got, then
// dlopens libdrosstest.so (with a second thread running) and calls into it. Exits 0 if every check passed.
#include <dlfcn.h>
#include <elf.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <sys/auxv.h>
#include <unistd.h>

static int failures;
static void check(const char *what, int ok) {
    printf("%s %s\n", ok ? "ok" : "FAIL", what);
    failures += !ok;
}

static int fds[2];

// Waits for the library's TLS counter, then calls it: its TLS must have been set up for this thread too.
static void *other(void *arg) {
    int (*calls)(void);
    (void)arg;
    if (read(fds[0], &calls, sizeof calls) != sizeof calls) return (void *)-1L;
    return (void *)(long)calls();
}

int main(void) {
    const Elf64_Phdr *ph = (const Elf64_Phdr *)getauxval(AT_PHDR);
    const char *execfn = (const char *)getauxval(AT_EXECFN);
    size_t n = execfn ? strlen(execfn) : 0;
    check("auxiliary vector", getauxval(AT_BASE) && ph && ph->p_type == PT_PHDR && getauxval(AT_PHNUM) &&
          getauxval(AT_ENTRY) && n >= 7 && !strcmp(execfn + n - 7, "dyntest"));

    pthread_t t;
    check("thread", pipe(fds) == 0 && pthread_create(&t, 0, other, 0) == 0);
    void *h = dlopen("libdrosstest.so", RTLD_NOW);
    check("dlopen", h != 0);
    if (!h) {
        printf("%s\n", dlerror());
        return 1;
    }
    int (*add)(int, int) = (int (*)(int, int))dlsym(h, "drosstest_add");
    const char *(*name)(void) = (const char *(*)(void))dlsym(h, "drosstest_name");
    int (*format)(char *, int, int) = (int (*)(char *, int, int))dlsym(h, "drosstest_format");
    int (*calls)(void) = (int (*)(void))dlsym(h, "drosstest_calls");
    check("dlsym", add && name && format && calls);
    if (!add || !name || !format || !calls) return 1;
    char buf[16];
    check("calls into the library", add(40, 2) == 42 && !strcmp(name(), "dross") && format(buf, sizeof buf, 5) == 3 &&
          !strcmp(buf, "v=5"));
    int first = calls(), second = calls();
    void *theirs = 0;
    int sent = write(fds[1], &calls, sizeof calls) == sizeof calls;
    check("thread-local storage", first == 7 && second == 8 && sent && pthread_join(t, &theirs) == 0 && theirs == (void *)7L);
    check("missing library", !dlopen("libdross-missing.so", RTLD_NOW) && dlerror());
    printf("dyntest %s\n", failures ? "FAILED" : "passed");
    return failures != 0;
}
