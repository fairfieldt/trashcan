/* Minimal test fixture for dross/abi/src/lib.rs. Not a public C SDK. */
#ifndef DROSS_C_TEST_NATIVE_H
#define DROSS_C_TEST_NATIVE_H
#include <stddef.h>
#include <stdint.h>

typedef uint32_t handle_t;
typedef struct { int64_t status; uint64_t value; } native_result;
_Static_assert(sizeof(handle_t) == 4, "native handles are u32");
_Static_assert(sizeof(uintptr_t) == 8, "this fixture targets x86-64");

enum { N_LOG=0, N_PROCESS_EXIT=1, N_CLOCK=5, N_CLOSE=6, N_DUP=7, N_MEM_PHYSICAL=9,
       N_CHANNEL_WRITE=14, N_CHANNEL_READ=15, N_RESOURCE_SUB=20 };
enum { E_ACCESS_DENIED=-4, E_WRONG_TYPE=-3, E_TIMED_OUT=-9 };

/* rax carries status, rdx carries the result; rcx/r11 are syscall clobbers. */
static inline native_result native_call(uint64_t number, uint64_t a0, uint64_t a1,
                                        uint64_t a2, uint64_t a3, uint64_t a4,
                                        uint64_t a5) {
    register uint64_t rax __asm__("rax") = number | UINT64_C(0x40000000);
    register uint64_t rdx __asm__("rdx") = a2;
    register uint64_t r10 __asm__("r10") = a3;
    register uint64_t r8 __asm__("r8") = a4;
    register uint64_t r9 __asm__("r9") = a5;
    __asm__ volatile("syscall" : "+a"(rax), "+d"(rdx)
        : "D"(a0), "S"(a1), "r"(r10), "r"(r8), "r"(r9)
        : "rcx", "r11", "memory", "cc");
    return (native_result){ (int64_t)rax, rdx };
}
static inline uint64_t deadline_ms(uint64_t milliseconds) {
    native_result result=native_call(N_CLOCK,0,0,0,0,0,0);
    return result.value + milliseconds*UINT64_C(1000000);
}
static inline int64_t close_handle(handle_t *handle) {
    if (!*handle) return 0;
    native_result result=native_call(N_CLOSE,*handle,0,0,0,0,0);
    if (!result.status) *handle=0;
    return result.status;
}
static inline native_result receive(handle_t channel, uint8_t *bytes, size_t capacity,
                                    handle_t *handles, size_t handle_capacity,
                                    uint64_t deadline) {
    return native_call(N_CHANNEL_READ,channel,(uintptr_t)bytes,capacity,
                       (uintptr_t)handles,handle_capacity,deadline);
}
/* On success attached handles have moved. On failure they remain caller-owned. */
static inline native_result send_message(handle_t channel, const uint8_t *bytes,
                                         size_t length, const handle_t *handles,
                                         size_t count) {
    return native_call(N_CHANNEL_WRITE,channel,(uintptr_t)bytes,length,
                       (uintptr_t)handles,count,0);
}
static inline void log_text(const char *text, size_t length) {
    (void)native_call(N_LOG,(uintptr_t)text,length,0,0,0,0);
}
#define LOG_LITERAL(text) log_text((text),sizeof(text)-1)
_Noreturn static inline void process_exit(int64_t code) {
    (void)native_call(N_PROCESS_EXIT,(uint64_t)code,0,0,0,0,0);
    for (;;) __asm__ volatile("ud2");
}
#endif
