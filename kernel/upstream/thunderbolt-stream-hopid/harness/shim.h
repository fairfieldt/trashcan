/* Just enough of the kernel for stream.c's HopID bookkeeping, with a model of the XDomain HopID IDAs. */
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>

#define EINVAL 22
#define ENOSPC 28
#define EBUSY 16
#define ENOMEM 12
#define TB_PATH_MIN_HOPID 8
#define MAX_HOPID 15
#define container_of(p, t, m) ((t *)((char *)(p) - offsetof(t, m)))
#define pr_warn(...) (printf("    pr_warn: " __VA_ARGS__))
#define scoped_guard(type, lock) for (int _once = 1; _once; _once = 0)
#define guard(type) (void)
#define wake_up_interruptible(w) ((void)(w))
#define wake_up_interruptible_poll(w, m) ((void)(w))
#define kfree(p) ((void)(p))
#define ida_free(ida, id) ((void)(id))
#define EPOLLHUP 0
#define EPOLLERR 0

struct kref { int refcount; };
struct mutex { int dummy; };
struct config_item { const char *name; };
struct config_group { struct config_item cg_item; };
struct tb_service { int dummy; };
struct tb_xdomain { int dummy; };
struct tbstream { struct tb_service *svc; };
struct tbstream_dev {
	struct config_group group;
	struct kref kref;
	struct { const char *name; } misc;
	struct tbstream *stream;
	struct mutex lock;
	int index, in_hopid, out_hopid, wait;
};

enum { FREE, STREAM, OTHER };
extern int in_owner[MAX_HOPID + 1], out_owner[MAX_HOPID + 1], fail_next_alloc;
extern int warn_unallocated, freed_other_users, peer_in, peer_out;
extern struct tb_xdomain the_xd;

static inline const char *config_item_name(struct config_item *i) { return i->name; }
static inline struct tb_xdomain *tb_service_parent(struct tb_service *svc) { (void)svc; return &the_xd; }
static inline struct tbstream *tbstream_get(struct tbstream *s) { return s; }
static inline void tbstream_put(struct tbstream *s) { (void)s; }
static inline void tb_service_properties_changed(struct tb_service *s) { (void)s; }
static inline int service_update_properties(struct tb_service *svc, const char *name, int in, int out)
{ (void)svc; (void)name; (void)in; (void)out; return 0; }
static inline void service_get_hopids(struct tb_service *svc, const char *name, int *in, int *out)
{ (void)svc; (void)name; if (peer_out) *in = peer_out; if (peer_in) *out = peer_in; }

/* ida_alloc_range(): the lowest free ID in [min, max] */
static inline int model_alloc(int *owner, int hopid)
{
	if (hopid < 0)
		hopid = TB_PATH_MIN_HOPID;
	if (hopid < TB_PATH_MIN_HOPID || hopid > MAX_HOPID)
		return -EINVAL;
	if (fail_next_alloc) {
		fail_next_alloc = 0;
		return -ENOMEM;
	}
	for (int id = hopid; id <= MAX_HOPID; id++)
		if (owner[id] == FREE) {
			owner[id] = STREAM;
			return id;
		}
	return -ENOSPC;
}

/* ida_free(): WARNs for an ID that isn't allocated */
static inline void model_release(int *owner, int id, const char *dir)
{
	if (owner[id] == FREE) {
		printf("    WARNING: ida_free: %s HopID %d is not allocated\n", dir, id);
		warn_unallocated++;
	} else if (owner[id] == OTHER) {
		printf("    BUG: released %s HopID %d, which another user holds\n", dir, id);
		freed_other_users++;
	}
	owner[id] = FREE;
}

static inline int tb_xdomain_alloc_in_hopid(struct tb_xdomain *xd, int h) { (void)xd; return model_alloc(in_owner, h); }
static inline int tb_xdomain_alloc_out_hopid(struct tb_xdomain *xd, int h) { (void)xd; return model_alloc(out_owner, h); }
static inline void tb_xdomain_release_in_hopid(struct tb_xdomain *xd, int h) { (void)xd; model_release(in_owner, h, "in"); }
static inline void tb_xdomain_release_out_hopid(struct tb_xdomain *xd, int h) { (void)xd; model_release(out_owner, h, "out"); }
