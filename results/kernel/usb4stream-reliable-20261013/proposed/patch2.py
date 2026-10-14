import sys
p = sys.argv[1]
s = open(p).read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:60], s.count(old))
    s = s.replace(old, new)


rep("""		/*
		 * CLOSE tunneled packet. If userspace already read""", """		/*
		 * A frame the NHI flagged (packets lost, buffer overrun)
		 * holds bytes that are not the sender's: drop it, so the
		 * reader sees a gap instead of wrong data.
		 */
		if (sf->frame.flags & (RING_DESC_CRC_ERROR | RING_DESC_BUFFER_OVERRUN)) {
			sdev->rx_dropped++;
			ret = tbstream_dev_consume_rx(sdev);
			if (ret)
				break;
			continue;
		}
		/*
		 * CLOSE tunneled packet. If userspace already read""")
rep(""" * @closed: CLOSE packet was received""", """ * @rx_dropped: Received frames dropped as damaged (CRC error, overrun)
 * @closed: CLOSE packet was received""")
rep("""	int users;
	bool closed;""", """	int users;
	unsigned long rx_dropped;
	bool closed;""")
rep("""	if (sf->frame.flags & RING_DESC_CRC_ERROR)
		pr_warn("RX CRC error\\n");
	else if (sf->frame.flags & RING_DESC_BUFFER_OVERRUN)
		pr_warn("RX buffer overrun\\n");""", """	if (sf->frame.flags & RING_DESC_CRC_ERROR)
		pr_warn_ratelimited("RX CRC error\\n");
	else if (sf->frame.flags & RING_DESC_BUFFER_OVERRUN)
		pr_warn_ratelimited("RX buffer overrun\\n");""")
rep("""CONFIGFS_ATTR_RO(tbstream_dev_, index);""", """CONFIGFS_ATTR_RO(tbstream_dev_, index);

static ssize_t tbstream_dev_rx_dropped_show(struct config_item *item, char *buf)
{
	struct config_group *group = to_config_group(item);
	struct tbstream_dev *sdev = tbstream_dev_from_group(group);

	return sysfs_emit(buf, "%lu\\n", READ_ONCE(sdev->rx_dropped));
}
CONFIGFS_ATTR_RO(tbstream_dev_, rx_dropped);""")
rep("""	&tbstream_dev_attr_throttling,
	NULL,""", """	&tbstream_dev_attr_throttling,
	&tbstream_dev_attr_rx_dropped,
	NULL,""")
open(p, 'w').write(s)
