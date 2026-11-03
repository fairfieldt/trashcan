/* SPDX-License-Identifier: MIT */
/*
 * gpud's Linux kernel API compatibility layer: just enough of <linux/...> for Linux's Southern Islands amdgpu
 * files to compile unmodified in a userspace process. Every linux/, drm/ and asm/ header the imported files
 * include is a one-line header that includes this one. Semantics follow the kernel where the imported code
 * depends on them (locking, atomics, delays, lists, error pointers); the rest are no-ops.
 */
#ifndef GPUD_LKPI_H
#define GPUD_LKPI_H

#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#include "lkpi_config.h"
#include "lkpi_kconfig.h"

/* glibc's <endian.h> defines __BIG_ENDIAN (4321) on every machine; the kernel defines it only on big-endian
 * builds, and amdgpu enables ring, IB and write-back byte swapping under #ifdef __BIG_ENDIAN. */
#undef __BIG_ENDIAN
#ifndef __LITTLE_ENDIAN
#define __LITTLE_ENDIAN 1234
#endif

/* ---- types (the __u32 family and __le32 come from the uapi <linux/types.h>) ---- */
#include <linux/types.h>
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int8_t s8;
typedef int16_t s16;
typedef int32_t s32;
typedef int64_t s64;
typedef uint64_t dma_addr_t;
typedef uint64_t phys_addr_t;
typedef uint64_t resource_size_t;
typedef unsigned int gfp_t;
typedef int64_t ktime_t;
typedef unsigned long kernel_ulong_t;
typedef int suspend_state_t;
typedef unsigned int pci_channel_state_t;
typedef unsigned int __poll_t;
typedef unsigned long __kernel_ulong_t;
typedef long __kernel_long_t;
typedef size_t __kernel_size_t;

#undef __iomem
#define __iomem
#undef __user
#define __user
#undef __force
#define __force
#undef __rcu
#define __rcu
#undef __bitwise
#define __bitwise
#undef __must_check
#define __must_check __attribute__((warn_unused_result))
#undef __maybe_unused
#define __maybe_unused __attribute__((unused))
#undef __always_unused
#define __always_unused __attribute__((unused))
#undef __always_inline
#define __always_inline inline __attribute__((always_inline))
#undef noinline
#define noinline __attribute__((noinline))
#undef __packed
#define __packed __attribute__((packed))
#undef __aligned
#define __aligned(x) __attribute__((aligned(x)))
#undef __aligned_u64
#define __aligned_u64 uint64_t __attribute__((aligned(8)))
#undef __printf
#define __printf(a, b) __attribute__((format(printf, a, b)))
#undef __init
#define __init
#undef __exit
#define __exit
#undef __initdata
#define __initdata
#undef __read_mostly
#define __read_mostly
#undef __cold
#define __cold
#undef __counted_by
#define __counted_by(x)
#undef __nonstring
#define __nonstring
#undef __must_hold
#define __must_hold(x)
#undef __acquires
#define __acquires(x)
#undef __releases
#define __releases(x)
#undef __pure
#define __pure __attribute__((pure))
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#define fallthrough __attribute__((fallthrough))
#define barrier() __asm__ __volatile__("" ::: "memory")
#define mb() __sync_synchronize()
#define rmb() __sync_synchronize()
#define wmb() __sync_synchronize()
#define smp_mb() __sync_synchronize()
#define smp_rmb() __sync_synchronize()
#define smp_wmb() __sync_synchronize()
#define READ_ONCE(x) (*(const volatile __typeof__(x) *)&(x))
#define WRITE_ONCE(x, v) (*(volatile __typeof__(x) *)&(x) = (v))
#define OPTIMIZER_HIDE_VAR(var) __asm__("" : "=r"(var) : "0"(var))
#define unreachable() __builtin_unreachable()
#define BUILD_BUG_ON(c) _Static_assert(!(c), #c)
#define BUILD_BUG_ON_ZERO(e) 0
#define __same_type(a, b) __builtin_types_compatible_p(__typeof__(a), __typeof__(b))
#define typeof_member(T, m) __typeof__(((T *)0)->m)

#define container_of(ptr, type, member) ((type *)((char *)(ptr) - offsetof(type, member)))
#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))
#define BIT(n) (1UL << (n))
#define BIT_ULL(n) (1ULL << (n))
#define BITS_PER_LONG 64
#define BITS_PER_BYTE 8
#define BITS_TO_LONGS(n) (((n) + BITS_PER_LONG - 1) / BITS_PER_LONG)
#define GENMASK(h, l) (((~0UL) << (l)) & (~0UL >> (BITS_PER_LONG - 1 - (h))))
#define GENMASK_ULL(h, l) (((~0ULL) << (l)) & (~0ULL >> (63 - (h))))
#define upper_32_bits(n) ((u32)(((u64)(n)) >> 32))
#define lower_32_bits(n) ((u32)((n) & 0xffffffff))
#define upper_16_bits(n) ((u16)((n) >> 16))
#define lower_16_bits(n) ((u16)((n) & 0xffff))
#define min(a, b) ({ __typeof__(a) _a = (a); __typeof__(b) _b = (b); _a < _b ? _a : _b; })
#define max(a, b) ({ __typeof__(a) _a = (a); __typeof__(b) _b = (b); _a > _b ? _a : _b; })
#define min3(a, b, c) min(min(a, b), c)
#define max3(a, b, c) max(max(a, b), c)
#define min_t(t, a, b) ({ t _a = (a); t _b = (b); _a < _b ? _a : _b; })
#define max_t(t, a, b) ({ t _a = (a); t _b = (b); _a > _b ? _a : _b; })
#define clamp(v, lo, hi) min(max(v, lo), hi)
#define clamp_t(t, v, lo, hi) min_t(t, max_t(t, v, lo), hi)
#define clamp_val(v, lo, hi) clamp_t(__typeof__(v), v, lo, hi)
#define swap(a, b) do { __typeof__(a) _t = (a); (a) = (b); (b) = _t; } while (0)
#define abs(x) ({ __typeof__(x) _x = (x); _x < 0 ? -_x : _x; })
#define DIV_ROUND_UP(n, d) (((n) + (d) - 1) / (d))
#define DIV_ROUND_UP_ULL(n, d) DIV_ROUND_UP((unsigned long long)(n), (d))
#define DIV_ROUND_CLOSEST(x, d) (((x) + ((d) / 2)) / (d))
#define div_u64(n, d) ((u64)(n) / (u32)(d))
#define div64_u64(n, d) ((u64)(n) / (u64)(d))
#define div64_s64(n, d) ((s64)(n) / (s64)(d))
static inline u64 div64_u64_rem(u64 n, u64 d, u64 *rem) { *rem = n % d; return n / d; }
#define do_div(n, base) ({ u32 _rem = (u64)(n) % (base); (n) = (u64)(n) / (base); _rem; })
#define ALIGN(x, a) (((x) + ((__typeof__(x))(a) - 1)) & ~((__typeof__(x))(a) - 1))
#define ALIGN_DOWN(x, a) ((x) & ~((__typeof__(x))(a) - 1))
#define IS_ALIGNED(x, a) (((x) & ((__typeof__(x))(a) - 1)) == 0)
#define PTR_ALIGN(p, a) ((__typeof__(p))ALIGN((unsigned long)(p), (a)))
#define round_up(x, y) ((((x) - 1) | ((__typeof__(x))((y) - 1))) + 1)
#define round_down(x, y) ((x) & ~((__typeof__(x))((y) - 1)))
#define roundup(x, y) ((((x) + ((y) - 1)) / (y)) * (y))
#define rounddown(x, y) ((x) - ((x) % (y)))
static inline int fls(unsigned int x) { return x ? 32 - __builtin_clz(x) : 0; }
static inline int fls64(u64 x) { return x ? 64 - __builtin_clzll(x) : 0; }
static inline unsigned long __ffs(unsigned long x) { return __builtin_ctzl(x); }
static inline unsigned long __fls(unsigned long x) { return 63 - __builtin_clzl(x); }
#define ffs(x) __builtin_ffs(x)
static inline unsigned long ffz(unsigned long x) { return __builtin_ctzl(~x); }
static inline unsigned int hweight32(unsigned int x) { return __builtin_popcount(x); }
static inline unsigned int hweight8(unsigned int x) { return __builtin_popcount(x & 0xff); }
static inline unsigned long hweight64(u64 x) { return __builtin_popcountll(x); }
static inline unsigned long hweight_long(unsigned long x) { return __builtin_popcountl(x); }
static inline bool is_power_of_2(unsigned long n) { return n != 0 && (n & (n - 1)) == 0; }
static inline unsigned long roundup_pow_of_two(unsigned long n) { return n <= 1 ? 1 : 1UL << fls64(n - 1); }
static inline unsigned long rounddown_pow_of_two(unsigned long n) { return 1UL << (fls64(n) - 1); }
#define ilog2(n) ((n) ? fls64(n) - 1 : -1)
#define order_base_2(n) ((n) > 1 ? ilog2((n) - 1) + 1 : 0)
#define lower_32_bits_mask 0xffffffffu
#define U8_MAX 0xff
#define U16_MAX 0xffff
#define U32_MAX 0xffffffffu
#define U64_MAX 0xffffffffffffffffull
#define S32_MAX INT32_MAX
#define S64_MAX INT64_MAX
#define SZ_1K 0x400
#define SZ_4K 0x1000
#define SZ_8K 0x2000
#define SZ_16K 0x4000
#define SZ_32K 0x8000
#define SZ_64K 0x10000
#define SZ_128K 0x20000
#define SZ_256K 0x40000
#define SZ_512K 0x80000
#define SZ_1M 0x100000
#define SZ_2M 0x200000
#define SZ_4M 0x400000
#define SZ_8M 0x800000
#define SZ_16M 0x1000000
#define SZ_32M 0x2000000
#define SZ_64M 0x4000000
#define SZ_128M 0x8000000
#define SZ_256M 0x10000000
#define SZ_512M 0x20000000
#define SZ_1G 0x40000000ULL
#define SZ_2G 0x80000000ULL
#define SZ_4G 0x100000000ULL
#define PAGE_SHIFT 12
#undef PAGE_SIZE /* musl's limits.h has one too */
#define PAGE_SIZE (1UL << PAGE_SHIFT)
#define PAGE_MASK (~(PAGE_SIZE - 1))
#define PAGE_ALIGN(x) ALIGN(x, PAGE_SIZE)
#define PFN_UP(x) (((x) + PAGE_SIZE - 1) >> PAGE_SHIFT)
#define USEC_PER_SEC 1000000L
#define NSEC_PER_SEC 1000000000L
#define NSEC_PER_USEC 1000L
#define NSEC_PER_MSEC 1000000L
#define MSEC_PER_SEC 1000L
#define HZ 1000
#define MAX_SCHEDULE_TIMEOUT LONG_MAX
#define cpu_to_le16(x) ((u16)(x))
#define cpu_to_le32(x) ((u32)(x))
#define cpu_to_le64(x) ((u64)(x))
#define le16_to_cpu(x) ((u16)(x))
#define le32_to_cpu(x) ((u32)(x))
#define le64_to_cpu(x) ((u64)(x))
#define le32_to_cpup(p) (*(const u32 *)(p))
#define le16_to_cpup(p) (*(const u16 *)(p))
#define be16_to_cpu(x) __builtin_bswap16(x)
#define cpu_to_be16(x) __builtin_bswap16(x)
#define be32_to_cpu(x) __builtin_bswap32(x)
#define cpu_to_be32(x) __builtin_bswap32(x)
#define get_unaligned_le16(p) ({ u16 _v; memcpy(&_v, (p), 2); _v; })
#define get_unaligned_le32(p) ({ u32 _v; memcpy(&_v, (p), 4); _v; })
#define put_unaligned_le32(v, p) do { u32 _v = (v); memcpy((p), &_v, 4); } while (0)
#define struct_size(p, member, n) (sizeof(*(p)) + sizeof(*(p)->member) * (n))
#define array_size(a, b) ((a) * (b))
#define flex_array_size(p, member, n) (sizeof(*(p)->member) * (n))
#define sizeof_field(T, m) sizeof(((T *)0)->m)
#define EXPORT_SYMBOL(x)
#define EXPORT_SYMBOL_GPL(x)
#define MODULE_FIRMWARE(x)
#define MODULE_LICENSE(x)
#define MODULE_AUTHOR(x)
#define MODULE_DESCRIPTION(x)
#define MODULE_PARM_DESC(x, d)
#define module_param_named(n, v, t, p)
#define module_param(n, t, p)
#define THIS_MODULE NULL
#define KBUILD_MODNAME "amdgpu"

/* ---- errors ---- */
#define ERESTARTSYS 512
#define ENOTSUPP 524
#define EPROBE_DEFER 517
#define MAX_ERRNO 4095
#define IS_ERR_VALUE(x) ((unsigned long)(void *)(x) >= (unsigned long)-MAX_ERRNO)
static inline void *ERR_PTR(long error) { return (void *)error; }
static inline long PTR_ERR(const void *ptr) { return (long)ptr; }
static inline bool IS_ERR(const void *ptr) { return IS_ERR_VALUE((unsigned long)ptr); }
static inline bool IS_ERR_OR_NULL(const void *ptr) { return !ptr || IS_ERR(ptr); }
#define PTR_ERR_OR_ZERO(p) (IS_ERR(p) ? PTR_ERR(p) : 0)

/* ---- logging ---- */
void lkpi_log(int level, const char *fmt, ...) __printf(2, 3);
extern int lkpi_log_level;
#define LKPI_ERR 3
#define LKPI_WARN 4
#define LKPI_NOTICE 5
#define LKPI_INFO 6
#define LKPI_DEBUG 7
#define KERN_ERR ""
#define KERN_WARNING ""
#define KERN_INFO ""
#define KERN_DEBUG ""
#define KERN_NOTICE ""
#define KERN_CONT ""
#define printk(fmt, ...) lkpi_log(LKPI_INFO, fmt, ##__VA_ARGS__)
#define pr_err(fmt, ...) lkpi_log(LKPI_ERR, fmt, ##__VA_ARGS__)
#define pr_warn(fmt, ...) lkpi_log(LKPI_WARN, fmt, ##__VA_ARGS__)
#define pr_warn_once pr_warn
#define pr_info(fmt, ...) lkpi_log(LKPI_INFO, fmt, ##__VA_ARGS__)
#define pr_notice(fmt, ...) lkpi_log(LKPI_NOTICE, fmt, ##__VA_ARGS__)
#define pr_debug(fmt, ...) lkpi_log(LKPI_DEBUG, fmt, ##__VA_ARGS__)
#define pr_cont(fmt, ...) lkpi_log(LKPI_INFO, fmt, ##__VA_ARGS__)
struct device { const char *name; void *driver_data; };
#define dev_err(d, fmt, ...) lkpi_log(LKPI_ERR, fmt, ##__VA_ARGS__)
#define dev_warn(d, fmt, ...) lkpi_log(LKPI_WARN, fmt, ##__VA_ARGS__)
#define dev_info(d, fmt, ...) lkpi_log(LKPI_INFO, fmt, ##__VA_ARGS__)
#define dev_notice(d, fmt, ...) lkpi_log(LKPI_NOTICE, fmt, ##__VA_ARGS__)
#define dev_dbg(d, fmt, ...) lkpi_log(LKPI_DEBUG, fmt, ##__VA_ARGS__)
#define dev_err_once dev_err
#define dev_warn_once dev_warn
#define dev_info_once dev_info
#define dev_err_ratelimited dev_err
#define dev_warn_ratelimited dev_warn
#define dev_info_ratelimited dev_info
#define dev_dbg_ratelimited dev_dbg
#define printk_ratelimit() 1
#define __printk_ratelimit(x) 1
#define dump_stack() ((void)0)
#define WARN(cond, fmt, ...) ({ int _c = !!(cond); if (_c) lkpi_log(LKPI_WARN, "WARNING %s:%d: " fmt, __FILE__, __LINE__, ##__VA_ARGS__); _c; })
#define WARN_ON(cond) ({ int _c = !!(cond); if (_c) lkpi_log(LKPI_WARN, "WARNING %s:%d: %s\n", __FILE__, __LINE__, #cond); _c; })
#define WARN_ON_ONCE WARN_ON
#define WARN_ONCE WARN
#define BUG() do { lkpi_log(LKPI_ERR, "BUG at %s:%d\n", __FILE__, __LINE__); abort(); } while (0)
#define BUG_ON(c) do { if (c) BUG(); } while (0)
#define might_sleep() ((void)0)
#define lockdep_assert_held(l) ((void)0)
#define in_interrupt() 0
#define in_task() 1

/* ---- memory ---- */
#define GFP_KERNEL 0u
#define GFP_ATOMIC 0u
#define GFP_NOWAIT 0u
#define GFP_DMA32 0u
#define __GFP_ZERO 1u
#define __GFP_NOWARN 0u
#define __GFP_RETRY_MAYFAIL 0u
static inline void *kmalloc(size_t n, gfp_t f) { return (f & __GFP_ZERO) ? calloc(1, n ? n : 1) : malloc(n ? n : 1); }
static inline void *kzalloc(size_t n, gfp_t f) { (void)f; return calloc(1, n ? n : 1); }
static inline void *kcalloc(size_t n, size_t s, gfp_t f) { (void)f; return calloc(n ? n : 1, s ? s : 1); }
static inline void *kmalloc_array(size_t n, size_t s, gfp_t f) { return kmalloc(n * s, f); }
static inline void *krealloc(void *p, size_t n, gfp_t f) { (void)f; return realloc(p, n); }
static inline void kfree(const void *p) { free((void *)p); }
static inline void *kmemdup(const void *src, size_t n, gfp_t f) { void *p = kmalloc(n, f); if (p) memcpy(p, src, n); return p; }
#define kvmalloc kmalloc
#define kvzalloc kzalloc
#define kvcalloc kcalloc
#define kvmalloc_array kmalloc_array
#define kvfree kfree
#define vmalloc(n) malloc(n)
#define vzalloc(n) calloc(1, (n))
#define vfree(p) free(p)
/* slab.h's typed allocators: kzalloc_obj(TYPE_OR_VAR[, gfp]) and the array forms */
#define kmalloc_obj(P, ...) ((typeof(P) *)kmalloc(sizeof(typeof(P)), 0))
#define kzalloc_obj(P, ...) ((typeof(P) *)kzalloc(sizeof(typeof(P)), 0))
#define kmalloc_objs(P, n, ...) ((typeof(P) *)kmalloc_array((n), sizeof(typeof(P)), 0))
#define kzalloc_objs(P, n, ...) ((typeof(P) *)kcalloc((n), sizeof(typeof(P)), 0))
#define kfree_sensitive kfree
static inline void memset32(u32 *s, u32 v, size_t n) { while (n--) *s++ = v; }
static inline void memset64(u64 *s, u64 v, size_t n) { while (n--) *s++ = v; }
size_t strscpy(char *dst, const char *src, size_t n);
#define sized_strscpy strscpy
#define strscpy_pad strscpy
int scnprintf(char *buf, size_t size, const char *fmt, ...) __printf(3, 4);
#define sysfs_emit(buf, fmt, ...) sprintf(buf, fmt, ##__VA_ARGS__)

/* ---- MMIO ---- */
static inline u32 readl(const volatile void __iomem *a) { return *(const volatile u32 *)a; }
static inline void writel(u32 v, volatile void __iomem *a) { *(volatile u32 *)a = v; }
static inline u8 readb(const volatile void __iomem *a) { return *(const volatile u8 *)a; }
static inline void writeb(u8 v, volatile void __iomem *a) { *(volatile u8 *)a = v; }
static inline u64 readq(const volatile void __iomem *a) { return *(const volatile u64 *)a; }
static inline void writeq(u64 v, volatile void __iomem *a) { *(volatile u64 *)a = v; }
#define ioread32 readl
#define iowrite32 writel
#define memcpy_fromio(d, s, n) memcpy((d), (const void *)(s), (n))
#define memcpy_toio(d, s, n) memcpy((void *)(d), (s), (n))
#define memset_io(d, c, n) memset((void *)(d), (c), (n))

/* ---- atomics ---- */
typedef struct { int counter; } atomic_t;
typedef struct { s64 counter; } atomic64_t;
typedef atomic64_t atomic_long_t;
#define ATOMIC_INIT(i) { (i) }
#define atomic_read(v) __atomic_load_n(&(v)->counter, __ATOMIC_SEQ_CST)
#define atomic_set(v, i) __atomic_store_n(&(v)->counter, (i), __ATOMIC_SEQ_CST)
#define atomic_inc(v) ((void)__atomic_add_fetch(&(v)->counter, 1, __ATOMIC_SEQ_CST))
#define atomic_dec(v) ((void)__atomic_sub_fetch(&(v)->counter, 1, __ATOMIC_SEQ_CST))
#define atomic_add(i, v) ((void)__atomic_add_fetch(&(v)->counter, (i), __ATOMIC_SEQ_CST))
#define atomic_sub(i, v) ((void)__atomic_sub_fetch(&(v)->counter, (i), __ATOMIC_SEQ_CST))
#define atomic_inc_return(v) __atomic_add_fetch(&(v)->counter, 1, __ATOMIC_SEQ_CST)
#define atomic_dec_return(v) __atomic_sub_fetch(&(v)->counter, 1, __ATOMIC_SEQ_CST)
#define atomic_add_return(i, v) __atomic_add_fetch(&(v)->counter, (i), __ATOMIC_SEQ_CST)
#define atomic_dec_and_test(v) (atomic_dec_return(v) == 0)
#define atomic_xchg(v, n) __atomic_exchange_n(&(v)->counter, (n), __ATOMIC_SEQ_CST)
#define atomic_cmpxchg(v, o, n) ({ __typeof__((v)->counter) _o = (o); __atomic_compare_exchange_n(&(v)->counter, &_o, (n), 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST); _o; })
#define atomic64_read atomic_read
#define atomic64_set atomic_set
#define atomic64_inc atomic_inc
#define atomic64_add atomic_add
#define atomic64_sub atomic_sub
#define atomic64_inc_return atomic_inc_return
#define atomic64_add_return atomic_add_return
#define atomic64_xchg atomic_xchg
#define cmpxchg(p, o, n) ({ __typeof__(*(p)) _o = (o); __atomic_compare_exchange_n((p), &_o, (n), 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST); _o; })
#define xchg(p, n) __atomic_exchange_n((p), (n), __ATOMIC_SEQ_CST)

/* ---- bitmaps ---- */
#define DECLARE_BITMAP(name, bits) unsigned long name[BITS_TO_LONGS(bits)]
static inline void set_bit(long nr, volatile unsigned long *a) { __atomic_or_fetch(&a[nr / 64], 1UL << (nr % 64), __ATOMIC_SEQ_CST); }
static inline void clear_bit(long nr, volatile unsigned long *a) { __atomic_and_fetch(&a[nr / 64], ~(1UL << (nr % 64)), __ATOMIC_SEQ_CST); }
static inline bool test_bit(long nr, const volatile unsigned long *a) { return (a[nr / 64] >> (nr % 64)) & 1; }
static inline bool test_and_set_bit(long nr, volatile unsigned long *a) { return (__atomic_fetch_or(&a[nr / 64], 1UL << (nr % 64), __ATOMIC_SEQ_CST) >> (nr % 64)) & 1; }
#define __set_bit set_bit
#define __clear_bit clear_bit
static inline unsigned long find_next_zero_bit(const unsigned long *a, unsigned long size, unsigned long off)
{
	for (unsigned long i = off; i < size; i++)
		if (!test_bit(i, a))
			return i;
	return size;
}
static inline unsigned long find_next_bit(const unsigned long *a, unsigned long size, unsigned long off)
{
	for (unsigned long i = off; i < size; i++)
		if (test_bit(i, a))
			return i;
	return size;
}
#define find_first_zero_bit(a, size) find_next_zero_bit((a), (size), 0)
#define find_first_bit(a, size) find_next_bit((a), (size), 0)
#define for_each_set_bit(bit, addr, size) \
	for ((bit) = find_first_bit((addr), (size)); (bit) < (size); (bit) = find_next_bit((addr), (size), (bit) + 1))
#define for_each_inst(i, mask) for ((i) = ffs(mask) - 1; (i) >= 0 && (mask) >> (i); (i) = ffs((mask) & (~0U << ((i) + 1))) - 1)
static inline void bitmap_zero(unsigned long *dst, unsigned int nbits) { memset(dst, 0, BITS_TO_LONGS(nbits) * sizeof(long)); }
static inline void bitmap_set(unsigned long *map, unsigned int start, unsigned int n) { while (n--) set_bit(start++, map); }

/* ---- locking (pthread) ---- */
struct mutex { pthread_mutex_t m; };
#define DEFINE_MUTEX(name) struct mutex name = { PTHREAD_MUTEX_INITIALIZER }
static inline void mutex_init(struct mutex *l) { pthread_mutex_init(&l->m, NULL); }
static inline void mutex_destroy(struct mutex *l) { pthread_mutex_destroy(&l->m); }
static inline void mutex_lock(struct mutex *l) { pthread_mutex_lock(&l->m); }
static inline int mutex_lock_interruptible(struct mutex *l) { pthread_mutex_lock(&l->m); return 0; }
static inline int mutex_trylock(struct mutex *l) { return pthread_mutex_trylock(&l->m) == 0; }
static inline void mutex_unlock(struct mutex *l) { pthread_mutex_unlock(&l->m); }
#define mutex_lock_nested(l, s) mutex_lock(l)
/* Spinlocks are mutexes here: interrupt work runs on a thread, never in a signal handler. */
typedef struct { pthread_mutex_t m; } spinlock_t;
typedef spinlock_t raw_spinlock_t;
#define DEFINE_SPINLOCK(name) spinlock_t name = { PTHREAD_MUTEX_INITIALIZER }
#define __SPIN_LOCK_UNLOCKED(name) { PTHREAD_MUTEX_INITIALIZER }
static inline void spin_lock_init(spinlock_t *l) { pthread_mutex_init(&l->m, NULL); }
static inline void spin_lock(spinlock_t *l) { pthread_mutex_lock(&l->m); }
static inline void spin_unlock(spinlock_t *l) { pthread_mutex_unlock(&l->m); }
#define spin_lock_irqsave(l, flags) do { (flags) = 0; spin_lock(l); } while (0)
#define spin_unlock_irqrestore(l, flags) do { (void)(flags); spin_unlock(l); } while (0)
#define spin_lock_irq spin_lock
#define spin_unlock_irq spin_unlock
#define spin_lock_bh spin_lock
#define spin_unlock_bh spin_unlock
struct rw_semaphore { pthread_rwlock_t l; };
static inline void init_rwsem(struct rw_semaphore *s) { pthread_rwlock_init(&s->l, NULL); }
static inline void down_read(struct rw_semaphore *s) { pthread_rwlock_rdlock(&s->l); }
static inline void up_read(struct rw_semaphore *s) { pthread_rwlock_unlock(&s->l); }
static inline void down_write(struct rw_semaphore *s) { pthread_rwlock_wrlock(&s->l); }
static inline void up_write(struct rw_semaphore *s) { pthread_rwlock_unlock(&s->l); }
static inline int down_read_trylock(struct rw_semaphore *s) { return pthread_rwlock_tryrdlock(&s->l) == 0; }
#define rcu_read_lock() ((void)0)
#define rcu_read_unlock() ((void)0)
#define rcu_dereference(p) (p)
#define rcu_dereference_protected(p, c) (p)
#define rcu_assign_pointer(p, v) ((p) = (v))
#define RCU_INIT_POINTER(p, v) ((p) = (v))
#define synchronize_rcu() ((void)0)
struct kref { atomic_t refcount; };
static inline void kref_init(struct kref *k) { atomic_set(&k->refcount, 1); }
static inline void kref_get(struct kref *k) { atomic_inc(&k->refcount); }
static inline int kref_put(struct kref *k, void (*release)(struct kref *))
{
	if (atomic_dec_and_test(&k->refcount)) {
		release(k);
		return 1;
	}
	return 0;
}
static inline unsigned int kref_read(const struct kref *k) { return atomic_read(&k->refcount); }

/* ---- time ---- */
u64 lkpi_ktime_ns(void);
void lkpi_udelay(unsigned long us);
void lkpi_msleep(unsigned long ms);
#define udelay(us) lkpi_udelay(us)
#define ndelay(ns) lkpi_udelay(((ns) + 999) / 1000)
#define mdelay(ms) lkpi_udelay((ms) * 1000UL)
#define msleep(ms) lkpi_msleep(ms)
#define msleep_interruptible(ms) (lkpi_msleep(ms), 0)
#define usleep_range(lo, hi) lkpi_udelay(lo)
#define fsleep(us) lkpi_udelay(us)
#if defined(__x86_64__)
#define cpu_relax() __asm__ __volatile__("pause" ::: "memory")
#else
#define cpu_relax() __asm__ __volatile__("" ::: "memory")
#endif
#define jiffies ((unsigned long)(lkpi_ktime_ns() / 1000000))
#define jiffies_64 ((u64)jiffies)
#define msecs_to_jiffies(ms) ((unsigned long)(ms))
#define usecs_to_jiffies(us) ((unsigned long)DIV_ROUND_UP((us), 1000))
#define jiffies_to_msecs(j) ((unsigned int)(j))
#define jiffies_to_usecs(j) ((unsigned int)(j) * 1000)
#define nsecs_to_jiffies(ns) ((unsigned long)((ns) / 1000000))
#define time_after(a, b) ((long)((b) - (a)) < 0)
#define time_before(a, b) time_after(b, a)
#define time_after_eq(a, b) ((long)((a) - (b)) >= 0)
#define time_is_before_jiffies(a) time_after(jiffies, a)
static inline ktime_t ktime_get(void) { return (ktime_t)lkpi_ktime_ns(); }
#define ktime_get_ns() lkpi_ktime_ns()
#define ktime_get_raw ktime_get
#define ktime_to_ns(t) ((s64)(t))
#define ktime_to_us(t) ((s64)(t) / 1000)
#define ktime_to_ms(t) ((s64)(t) / 1000000)
#define ktime_us_delta(a, b) (((s64)(a) - (s64)(b)) / 1000)
#define ktime_sub(a, b) ((a) - (b))
#define ktime_add_us(t, us) ((t) + (s64)(us) * 1000)
#define ns_to_ktime(ns) ((ktime_t)(ns))
#define ktime_after(a, b) ((a) > (b))
#define read_poll_timeout(op, val, cond, sleep_us, timeout_us, sleep_before, args...) ({ \
	u64 _deadline = lkpi_ktime_ns() + (u64)(timeout_us) * 1000; \
	int _r = 0; \
	for (;;) { \
		(val) = op(args); \
		if (cond) break; \
		if ((timeout_us) && lkpi_ktime_ns() > _deadline) { (val) = op(args); _r = (cond) ? 0 : -ETIMEDOUT; break; } \
		if (sleep_us) lkpi_udelay(sleep_us); \
	} \
	_r; })

/* ---- lists ---- */
struct list_head { struct list_head *next, *prev; };
struct hlist_node { struct hlist_node *next, **pprev; };
struct hlist_head { struct hlist_node *first; };
#define LIST_HEAD_INIT(name) { &(name), &(name) }
#define LIST_HEAD(name) struct list_head name = LIST_HEAD_INIT(name)
static inline void INIT_LIST_HEAD(struct list_head *l) { l->next = l->prev = l; }
static inline void __list_add(struct list_head *n, struct list_head *prev, struct list_head *next)
{
	next->prev = n;
	n->next = next;
	n->prev = prev;
	prev->next = n;
}
static inline void list_add(struct list_head *n, struct list_head *h) { __list_add(n, h, h->next); }
static inline void list_add_tail(struct list_head *n, struct list_head *h) { __list_add(n, h->prev, h); }
static inline void list_del(struct list_head *e)
{
	e->next->prev = e->prev;
	e->prev->next = e->next;
	e->next = e->prev = NULL;
}
static inline void list_del_init(struct list_head *e) { list_del(e); INIT_LIST_HEAD(e); }
static inline bool list_empty(const struct list_head *h) { return h->next == h; }
#define list_entry(p, t, m) container_of(p, t, m)
#define list_first_entry(h, t, m) list_entry((h)->next, t, m)
#define list_next_entry(p, m) list_entry((p)->m.next, __typeof__(*(p)), m)
#define list_for_each(p, h) for ((p) = (h)->next; (p) != (h); (p) = (p)->next)
#define list_for_each_entry(p, h, m) \
	for ((p) = list_entry((h)->next, __typeof__(*(p)), m); &(p)->m != (h); (p) = list_entry((p)->m.next, __typeof__(*(p)), m))
#define list_for_each_entry_safe(p, n, h, m) \
	for ((p) = list_entry((h)->next, __typeof__(*(p)), m), (n) = list_entry((p)->m.next, __typeof__(*(p)), m); \
	     &(p)->m != (h); (p) = (n), (n) = list_entry((n)->m.next, __typeof__(*(n)), m))

/* ---- wait queues, work, timers: interrupt handling runs on gpud's IRQ thread ---- */
typedef struct { pthread_mutex_t m; pthread_cond_t c; } wait_queue_head_t;
void init_waitqueue_head(wait_queue_head_t *q);
void wake_up_all(wait_queue_head_t *q);
#define wake_up wake_up_all
#define wait_event_interruptible_timeout(q, cond, t) ({ long _t = (t); while (!(cond) && _t > 0) { lkpi_msleep(1); _t--; } (cond) ? (_t ? _t : 1) : 0; })
#define wait_event_timeout wait_event_interruptible_timeout
struct work_struct;
typedef void (*work_func_t)(struct work_struct *);
/* Work items run on one worker thread, in order (si_dpm's thermal handler is one); delayed work is not used. */
struct work_struct {
	work_func_t func;
	struct work_struct *next;
	bool pending;
};
struct delayed_work { struct work_struct work; };
#define INIT_WORK(w, f) ((w)->func = (f), (w)->next = NULL, (w)->pending = false)
#define INIT_DELAYED_WORK(w, f) ((w)->work.func = (f))
bool schedule_work(struct work_struct *w);
#define queue_work(q, w) schedule_work(w)
bool cancel_work_sync(struct work_struct *w);
bool flush_work(struct work_struct *w);
#define schedule_delayed_work(w, d) ((void)(w), true)
#define cancel_delayed_work_sync(w) ((void)(w), false)
#define to_delayed_work(w) container_of(w, struct delayed_work, work)
struct timer_list { void (*function)(struct timer_list *); };
#define timer_setup(t, f, flags) ((t)->function = (f))
#define mod_timer(t, e) 0
#define del_timer_sync(t) 0
#define timer_delete_sync(t) 0
#define from_timer(var, t, field) container_of(t, __typeof__(*var), field)
struct completion { int done; };
struct notifier_block { int dummy; };

/* ---- firmware ---- */
struct firmware { size_t size; const u8 *data; };
int request_firmware(const struct firmware **fw, const char *name, struct device *dev);
void release_firmware(const struct firmware *fw);
#define firmware_request_nowarn request_firmware

/* ---- PCI ---- */
struct pci_bus { unsigned char number; struct pci_dev *self; };
struct pci_dev {
	struct device dev;
	struct pci_bus *bus;
	unsigned int devfn;
	unsigned short vendor, device, subsystem_vendor, subsystem_device;
	u8 revision;
	unsigned int class;
	resource_size_t bar_start[6], bar_len[6];
	const struct gpud_platform *platform; /* gpud: config space access */
};
struct gpud_platform;
#define PCI_SLOT(devfn) (((devfn) >> 3) & 0x1f)
#define PCI_FUNC(devfn) ((devfn) & 0x07)
#define to_pci_dev(d) container_of(d, struct pci_dev, dev)
int pci_read_config_byte(const struct pci_dev *dev, int where, u8 *val);
int pci_read_config_word(const struct pci_dev *dev, int where, u16 *val);
int pci_read_config_dword(const struct pci_dev *dev, int where, u32 *val);
int pci_write_config_byte(const struct pci_dev *dev, int where, u8 val);
int pci_write_config_word(const struct pci_dev *dev, int where, u16 val);
int pci_write_config_dword(const struct pci_dev *dev, int where, u32 val);
void pci_set_master(struct pci_dev *dev);
void pci_clear_master(struct pci_dev *dev);
int pcie_capability_read_word(struct pci_dev *dev, int pos, u16 *val);
int pcie_capability_read_dword(struct pci_dev *dev, int pos, u32 *val);
int pcie_capability_write_word(struct pci_dev *dev, int pos, u16 val);
int pcie_capability_clear_and_set_word_locked(struct pci_dev *dev, int pos, u16 clear, u16 set);
#define pcie_capability_clear_and_set_word pcie_capability_clear_and_set_word_locked
#define pcie_capability_set_word(d, p, s) pcie_capability_clear_and_set_word_locked(d, p, 0, s)
#define pcie_capability_clear_word(d, p, c) pcie_capability_clear_and_set_word_locked(d, p, c, 0)
int pcie_get_readrq(struct pci_dev *dev);
int pcie_set_readrq(struct pci_dev *dev, int rq);
static inline bool pci_is_root_bus(struct pci_bus *b) { (void)b; return true; }
static inline bool pci_is_pcie(struct pci_dev *d) { (void)d; return true; }
/* The root port belongs to the host: gpud never reaches upstream of its own function. */
static inline struct pci_dev *pci_upstream_bridge(struct pci_dev *d) { (void)d; return NULL; }
int pci_find_capability(struct pci_dev *dev, int cap);
int pci_pcie_cap(struct pci_dev *dev);
#define pci_dev_put(d) ((void)0)
#define pci_resource_start(d, b) ((d)->bar_start[b])
#define pci_resource_len(d, b) ((d)->bar_len[b])

/* ---- DMA ---- */
#define DMA_BIT_MASK(n) (((n) == 64) ? ~0ULL : ((1ULL << (n)) - 1))
static inline int dma_set_mask(struct device *d, u64 m) { (void)d; (void)m; return 0; }
static inline int dma_set_coherent_mask(struct device *d, u64 m) { (void)d; (void)m; return 0; }
static inline int dma_set_mask_and_coherent(struct device *d, u64 m) { (void)d; (void)m; return 0; }
void *dma_alloc_coherent(struct device *dev, size_t size, dma_addr_t *dma, gfp_t gfp);
void dma_free_coherent(struct device *dev, size_t size, void *cpu, dma_addr_t dma);

/* ---- misc ---- */
struct seq_file;
struct irq_domain;
struct dentry;
struct file;
struct module;
struct kobject;
struct attribute;
struct dev_pm_ops;
struct dma_buf;
struct dma_resv;
struct drm_file;
typedef int irqreturn_t;
#define IRQ_NONE 0
#define IRQ_HANDLED 1
#define seq_printf(m, fmt, ...) ((void)0)
#define trace_printk(fmt, ...) ((void)0)
#define num_online_cpus() 1
#define smp_processor_id() 0
#define current_cpu_type() 0
#define capable(c) true
#define hash_init(h) ((void)0)
#define DECLARE_HASHTABLE(name, bits) struct hlist_head name[1 << (bits)]
#define __stringify_1(x) #x
#define __stringify(x) __stringify_1(x)

/* ---- string_helpers.h ---- */
static inline const char *str_yes_no(bool v) { return v ? "yes" : "no"; }
static inline const char *str_enable_disable(bool v) { return v ? "enable" : "disable"; }
static inline const char *str_enabled_disabled(bool v) { return v ? "enabled" : "disabled"; }
static inline const char *str_on_off(bool v) { return v ? "on" : "off"; }

/* ---- sysfs: attribute types for code that declares attributes; gpud registers none ---- */
typedef unsigned short umode_t;
struct attribute {
	const char *name;
	umode_t mode;
};
struct device_attribute {
	struct attribute attr;
	ssize_t (*show)(struct device *dev, struct device_attribute *attr, char *buf);
	ssize_t (*store)(struct device *dev, struct device_attribute *attr, const char *buf, size_t count);
};
#define DEVICE_ATTR(_name, _mode, _show, _store) \
	struct device_attribute dev_attr_##_name = { .attr = { .name = #_name, .mode = (_mode) }, .show = (_show), .store = (_store) }
struct attribute_group {
	const char *name;
	umode_t (*is_visible)(struct kobject *kobj, struct attribute *attr, int index);
	struct attribute **attrs;
};
static inline struct device *kobj_to_dev(struct kobject *kobj) { (void)kobj; return NULL; }
static inline void *dev_get_drvdata(const struct device *dev) { (void)dev; return NULL; }
static inline int devm_device_add_group(struct device *dev, const struct attribute_group *grp) { (void)dev; (void)grp; return 0; }

/* ---- i2c: the types power management embeds; gpud registers no i2c buses ---- */
struct i2c_algorithm;
struct i2c_adapter {
	const struct i2c_algorithm *algo;
	void *algo_data;
	char name[48];
};
struct i2c_client;
struct i2c_board_info {
	char type[20];
	unsigned short flags;
	unsigned short addr;
};
static inline struct i2c_client *i2c_new_client_device(struct i2c_adapter *adap, const struct i2c_board_info *info)
{
	(void)adap;
	(void)info;
	return NULL;
}

#endif
