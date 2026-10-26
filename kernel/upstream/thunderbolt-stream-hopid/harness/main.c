/*
 * Drives stream.c's HopID code (compiled unchanged, see run.sh) through attach, reconfiguration, detach and
 * release with allocations that fail, and counts ida_free() of IDs the stream doesn't hold.
 */
#include "shim.h"

int in_owner[MAX_HOPID + 1], out_owner[MAX_HOPID + 1], fail_next_alloc;
int warn_unallocated, freed_other_users, peer_in, peer_out;
struct tb_xdomain the_xd;
static struct tb_service svc;
static struct tbstream stream = { .svc = &svc };
struct tbstream_group { struct tbstream *stream; };

#include "stream_funcs.inc"

static struct tbstream_dev sdev;
static struct tbstream_group sg = { .stream = &stream };

static void reset(int in, int out)
{
	for (int i = 0; i <= MAX_HOPID; i++)
		in_owner[i] = out_owner[i] = FREE;
	sdev = (struct tbstream_dev){ .group.cg_item.name = "data", .in_hopid = in, .out_hopid = out };
	fail_next_alloc = peer_in = peer_out = 0;
}

static int check(const char *name, int warn0, int other0)
{
	int leaked = 0;

	for (int i = 0; i <= MAX_HOPID; i++)
		leaked += (in_owner[i] == STREAM) + (out_owner[i] == STREAM);
	printf("  %-60s ida_free of unallocated %d, other user's HopID freed %d, left allocated %d\n", name,
	       warn_unallocated - warn0, freed_other_users - other0, leaked);
	return (warn_unallocated - warn0) || (freed_other_users - other0) || leaked;
}

int main(void)
{
	int bad = 0, w, o;

	/* 1. HopIDs configured, both free: the normal path */
	reset(8, 9); w = warn_unallocated; o = freed_other_users;
	tbstream_dev_attach_stream(&sdev, &sg);
	tbstream_dev_detach_stream(&sdev);
	bad |= check("attach + detach, HopIDs free", w, o);

	/* 2. configured in HopID 8 held by another user of the XDomain */
	reset(8, 9); in_owner[8] = OTHER; w = warn_unallocated; o = freed_other_users;
	tbstream_dev_attach_stream(&sdev, &sg);
	tbstream_dev_detach_stream(&sdev);
	if (in_owner[8] == OTHER)
		in_owner[8] = FREE;	/* the other user's own release */
	bad |= check("attach with in HopID held elsewhere, detach", w, o);

	/* 3. allocation fails (out of memory) when attaching, then the configfs item is removed */
	reset(8, 9); w = warn_unallocated; o = freed_other_users;
	fail_next_alloc = 1;
	tbstream_dev_attach_stream(&sdev, &sg);
	sdev.kref.refcount = 0;
	tbstream_dev_release(&sdev.kref);
	bad |= check("attach with failing allocation, release", w, o);

	/* 4. automatic HopIDs from the peer's properties, one of them taken */
	reset(0, 0); peer_in = 10; peer_out = 11; out_owner[10] = OTHER; w = warn_unallocated; o = freed_other_users;
	tbstream_dev_attach_stream(&sdev, &sg);
	tbstream_dev_detach_stream(&sdev);
	if (out_owner[10] == OTHER)
		out_owner[10] = FREE;
	bad |= check("attach with peer HopIDs, one taken, detach", w, o);

	/* 5. attached; in_hopid changed to one that can't be allocated, then detach */
	reset(8, 9); in_owner[12] = OTHER; w = warn_unallocated; o = freed_other_users;
	tbstream_dev_attach_stream(&sdev, &sg);
	printf("    in_hopid store 12: %d\n", tbstream_dev_alloc_in_hopid(&sdev, 12));
	tbstream_dev_detach_stream(&sdev);
	if (in_owner[12] == OTHER)
		in_owner[12] = FREE;
	bad |= check("change in HopID to a taken one, detach", w, o);

	/* 6. attached; out_hopid change fails to allocate, then the item is released */
	reset(8, 9); w = warn_unallocated; o = freed_other_users;
	tbstream_dev_attach_stream(&sdev, &sg);
	fail_next_alloc = 1;
	printf("    out_hopid store 13: %d\n", tbstream_dev_alloc_out_hopid(&sdev, 13));
	sdev.kref.refcount = 0;
	tbstream_dev_release(&sdev.kref);
	bad |= check("change out HopID with failing allocation, release", w, o);

	printf("%s\n", bad ? "RESULT: HopID bookkeeping errors" : "RESULT: no HopID bookkeeping errors");
	return bad;
}
