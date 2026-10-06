// Throughput and ping-pong latency between two /dev/tbstreamX devices on the same host
// (two ends of one Thunderbolt cable).
// build: cc -O2 -pthread -o tbstream-bench tbstream-bench.c
// usage: tbstream-bench pingpong DEV_A DEV_B MSG_BYTES ITERS
//        tbstream-bench stream   DEV_A DEV_B CHUNK_BYTES SECONDS   (A writes, B reads)
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void die(const char *what) { perror(what); exit(1); }

// Read exactly n bytes (a stream read may return less than one write).
static void read_full(int fd, char *buf, size_t n) {
    while (n) {
        ssize_t r = read(fd, buf, n);
        if (r < 0) { if (errno == EINTR) continue; die("read"); }
        if (r == 0) { fprintf(stderr, "EOF\n"); exit(1); }
        buf += r; n -= r;
    }
}

static void write_full(int fd, const char *buf, size_t n) {
    while (n) {
        ssize_t w = write(fd, buf, n);
        if (w < 0) { if (errno == EINTR) continue; die("write"); }
        buf += w; n -= w;
    }
}

struct echo_arg { int fd; size_t len; long iters; };

static void *echo(void *p) {
    struct echo_arg *a = p;
    char *buf = malloc(a->len);
    for (long i = 0; i < a->iters; i++) {
        read_full(a->fd, buf, a->len);
        write_full(a->fd, buf, a->len);
    }
    free(buf);
    return NULL;
}

static int cmp(const void *x, const void *y) {
    double a = *(const double *)x, b = *(const double *)y;
    return a < b ? -1 : a > b;
}

struct drain_arg { int fd; size_t chunk; volatile int stop; uint64_t bytes; };

static void *drain(void *p) {
    struct drain_arg *a = p;
    char *buf = malloc(a->chunk);
    while (!a->stop) {
        ssize_t r = read(a->fd, buf, a->chunk);
        if (r < 0) { if (errno == EINTR) continue; die("read"); }
        if (r == 0) break;
        a->bytes += r;
    }
    free(buf);
    return NULL;
}

int main(int argc, char **argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: %s pingpong|stream DEV_A DEV_B BYTES ITERS|SECONDS\n", argv[0]);
        return 2;
    }
    int fa = open(argv[2], O_RDWR), fb = open(argv[3], O_RDWR);
    if (fa < 0 || fb < 0) die("open");
    size_t len = strtoul(argv[4], NULL, 0);
    if (!strcmp(argv[1], "pingpong")) {
        long iters = strtol(argv[5], NULL, 0), warm = iters / 10;
        struct echo_arg ea = { fb, len, iters + warm };
        pthread_t t;
        pthread_create(&t, NULL, echo, &ea);
        char *buf = calloc(1, len);
        double *rtt = malloc(sizeof(double) * iters);
        for (long i = 0; i < iters + warm; i++) {
            double t0 = now();
            write_full(fa, buf, len);
            read_full(fa, buf, len);
            if (i >= warm) rtt[i - warm] = (now() - t0) * 1e6;
        }
        pthread_join(t, NULL);
        qsort(rtt, iters, sizeof(double), cmp);
        double sum = 0;
        for (long i = 0; i < iters; i++) sum += rtt[i];
        printf("pingpong bytes=%zu iters=%ld rtt_us min=%.1f p50=%.1f p99=%.1f max=%.1f avg=%.1f\n",
               len, iters, rtt[0], rtt[iters / 2], rtt[iters * 99 / 100], rtt[iters - 1], sum / iters);
    } else {
        double secs = strtod(argv[5], NULL);
        struct drain_arg da = { fb, len > 65536 ? len : 65536, 0, 0 };
        pthread_t t;
        pthread_create(&t, NULL, drain, &da);
        char *buf = calloc(1, len);
        uint64_t sent = 0;
        double t0 = now(), t1;
        while ((t1 = now()) - t0 < secs) { write_full(fa, buf, len); sent += len; }
        double tw = now() - t0;
        // Let the reader catch up, then stop it by closing the writer side.
        usleep(200000);
        da.stop = 1;
        close(fa);
        pthread_join(t, NULL);
        printf("stream chunk=%zu seconds=%.2f sent=%.3f Gb/s received=%.3f Gb/s\n",
               len, tw, sent * 8 / tw / 1e9, da.bytes * 8 / tw / 1e9);
    }
    return 0;
}
