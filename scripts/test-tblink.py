#!/usr/bin/env python3
"""tblink's links, addresses and streams against a fake sysfs/configfs tree; never touches the host.

The script runs unchanged except that /sys, /etc and /run point into the fixture; `ip`, `modprobe`, `hostname`,
`flock` and `sleep` are mocks (one pass of `tblink run`), and `mkdir` creates a stream's configfs attributes as
the driver does. Temporary files go under $TMPDIR.
"""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

TBLINK = Path(__file__).resolve().parent.parent / 'nodes/macpro/root/usr/local/sbin/tblink'
# The fixture's nodes.tsv: mac0 is node 1 and mac2 node 3; the UUIDs are made up.
MAC2_BUS0 = '00000000-0000-0008-0000-000000000300'
NODES = ('# index, hostname, controller UUIDs by Apple bus\n'
         '1\tmac0\t00000000-0000-0008-0000-000000000100\t00000000-0000-0008-0000-000000000101\t'
         '00000000-0000-0008-0000-000000000102\n'
         f'3\tmac2\t{MAC2_BUS0}\t00000000-0000-0008-0000-000000000301\t00000000-0000-0008-0000-000000000302\n')

MOCKS = {
    'ip': '#!/bin/sh\n'
          'if [ "$1" = -4 ]; then cat "$MOCK_ROOT/ip-$6" 2>/dev/null; exit 0; fi\n'
          'echo "ip $*" >> "$MOCK_ROOT/calls"\n',
    'modprobe': '#!/bin/sh\necho "modprobe $*" >> "$MOCK_ROOT/calls"\nmkdir -p "$MOCK_ROOT/sys/module/thunderbolt_stream"\n',
    'hostname': '#!/bin/sh\necho mac0\n',
    'flock': '#!/bin/sh\nexit 0\n',
    # One pass of `tblink run`: the first sleep ends it.
    'sleep': '#!/bin/sh\nkill -TERM $PPID\n',
    # configfs: a stream directory comes with its attributes.
    # configfs: a stream directory comes with its attributes. With MOCK_AUTOCONFIG the driver has already copied the
    # peer's HopIDs (15) from its XDomain properties, and rewriting a HopID the stream holds fails (read-only here).
    'mkdir': '#!/bin/bash\n/bin/mkdir "$@" || exit $?\n'
             'd=${@: -1}\n'
             'if [[ $d == */kernel/config/thunderbolt/stream/*/trashcan ]]; then\n'
             '  n=$(cat "$MOCK_ROOT/next-index" 2>/dev/null || echo 0); echo $((n + 1)) > "$MOCK_ROOT/next-index"\n'
             '  h=0; [ -n "${MOCK_AUTOCONFIG:-}" ] && h=15\n'
             '  echo "$n" > "$d/index"; echo $h > "$d/in_hopid"; echo $h > "$d/out_hopid"; echo 256 > "$d/ring_size"\n'
             '  echo 0 > "$d/busy_poll"\n'
             '  if [ -n "${MOCK_AUTOCONFIG:-}" ]; then chmod 444 "$d/in_hopid" "$d/out_hopid"; fi\n'
             'fi\n'
             'exit 0\n',
    # configfs: removing a stream directory takes its attributes along.
    'rmdir': '#!/bin/bash\n'
             'for d in "$@"; do [[ $d == */kernel/config/thunderbolt/stream/*/trashcan ]] && '
             'rm -f "$d/index" "$d/in_hopid" "$d/out_hopid" "$d/ring_size" "$d/busy_poll"; done\n'
             'exec /bin/rmdir "$@"\n',
}


class Tblink(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='tblink-test-')
        self.root = Path(self.tmp.name)
        r = self.root
        mock = r / 'bin'
        mock.mkdir()
        for name, text in MOCKS.items():
            (mock / name).write_text(text)
            (mock / name).chmod(0o755)
        (r / 'etc/tblink').mkdir(parents=True)
        (r / 'etc/tblink/nodes.tsv').write_text(NODES)
        (r / 'etc/hosts').write_text('127.0.0.1 localhost\n')
        (r / 'run').mkdir()
        (r / 'sys/module/thunderbolt').mkdir(parents=True)
        (r / 'sys/kernel/config/thunderbolt/stream').mkdir(parents=True)
        self.devs = r / 'sys/bus/thunderbolt/devices'
        self.devs.mkdir(parents=True)
        # mac0's bus 0 controller, domain 0 (NHI 16:00.0), and its own router 0-0.
        nhi = r / 'sys/devices/pci/0000:16:00.0/domain0'
        nhi.mkdir(parents=True)
        (self.devs / 'domain0').symlink_to(nhi)
        (self.devs / '0-0').mkdir()
        script = TBLINK.read_text().replace('/sys/', f'{r}/sys/').replace('/etc/', f'{r}/etc/').replace('/run/', f'{r}/run/')
        self.script = r / 'tblink'
        self.script.write_text(script)
        self.script.chmod(0o755)
        self.env = dict(os.environ, MOCK_ROOT=str(r), PATH=f'{mock}:{os.environ["PATH"]}')

    def tearDown(self):
        self.tmp.cleanup()

    def link(self, xd='0-3', peer_uuid=MAC2_BUS0, stream_index=1, maxhopid=15, carrier=1):
        """An XDomain link to mac2's bus 0 controller, its network service and (with stream_index) the peer's
        stream service."""
        d = self.devs / xd
        d.mkdir(exist_ok=True)
        (d / 'unique_id').write_text(peer_uuid + '\n')
        (d / 'vendor').write_text('0x1\n')
        (d / 'maxhopid').write_text(f'{maxhopid}\n')
        (d / 'rx_lanes').write_text('2\n')
        for s in d.glob(f'{xd}.*'):
            for f in s.rglob('*'):
                if f.is_file():
                    f.unlink()
            for f in sorted(s.rglob('*'), reverse=True):
                f.rmdir()
            s.rmdir()
        net = d / f'{xd}.{1 - stream_index if stream_index is not None else 0}'
        (net / 'net/thunderbolt0').mkdir(parents=True)
        (net / 'key').write_text('network\n')
        if stream_index is not None:
            st = d / f'{xd}.{stream_index}'
            st.mkdir()
            (st / 'key').write_text('stream\n')
        net_dir = self.root / 'sys/class/net/thunderbolt0'
        net_dir.mkdir(parents=True, exist_ok=True)
        (net_dir / 'carrier').write_text(f'{carrier}\n')
        # mac0 is node 1, mac2 node 3, both on bus 0: 10.99.19.1/30.
        (self.root / 'ip-thunderbolt0').write_text('5: thunderbolt0    inet 10.99.19.1/30 scope global thunderbolt0\n')

    def run_pass(self):
        r = subprocess.run([str(self.script), 'run'], env=self.env, capture_output=True, text=True, timeout=30)
        return r.stdout + r.stderr

    def links(self):
        out = subprocess.run([str(self.script), 'links'], env=self.env, capture_output=True, text=True, check=True).stdout
        rows = [l.split('\t') for l in out.strip().split('\n')]
        return [dict(zip(rows[0], row)) for row in rows[1:]]

    def cfg(self):
        c = self.root / 'sys/kernel/config/thunderbolt/stream'
        return sorted(str(p.relative_to(c)) for p in c.rglob('*') if p.is_dir())

    def test_creates_the_stream_with_fixed_hopids(self):
        self.link()
        log = self.run_pass()
        self.assertEqual(self.cfg(), ['0-3.1', '0-3.1/trashcan'], log)
        d = self.root / 'sys/kernel/config/thunderbolt/stream/0-3.1/trashcan'
        self.assertEqual((d / 'in_hopid').read_text().strip(), '15')
        self.assertEqual((d / 'out_hopid').read_text().strip(), '15')
        self.assertIn('stream 0-3.1/trashcan, /dev/tbstream0', log)
        [row] = self.links()
        self.assertEqual((row['peer'], row['state'], row['stream']), ('mac2', 'up', '/dev/tbstream0'))
        # A second pass changes nothing.
        log = self.run_pass()
        self.assertNotIn('stream 0-3', log)
        self.assertEqual(self.cfg(), ['0-3.1', '0-3.1/trashcan'])

    def test_busy_poll_follows_its_setting(self):
        self.link()
        d = self.root / 'sys/kernel/config/thunderbolt/stream/0-3.1/trashcan'
        log = self.run_pass()
        self.assertEqual((d / 'busy_poll').read_text().strip(), '0', log)
        self.assertNotIn('busy_poll', log)
        (self.root / 'etc/tblink/busy_poll').write_text('1\n')
        log = self.run_pass()
        self.assertEqual((d / 'busy_poll').read_text().strip(), '1', log)
        self.assertIn('stream 0-3.1/trashcan busy_poll 1', log)
        # An open device refuses the write (read-only here): nothing is said, and the next pass sets it.
        (self.root / 'etc/tblink/busy_poll').write_text('0\n')
        (d / 'busy_poll').chmod(0o444)
        log = self.run_pass()
        self.assertEqual((d / 'busy_poll').read_text().strip(), '1', log)
        self.assertNotIn('busy_poll 0', log)
        (d / 'busy_poll').chmod(0o644)
        log = self.run_pass()
        self.assertEqual((d / 'busy_poll').read_text().strip(), '0', log)

    def test_loads_the_module_when_missing(self):
        self.link()
        self.run_pass()
        self.assertIn('modprobe thunderbolt_stream', (self.root / 'calls').read_text())

    def test_follows_a_service_index_swap(self):
        self.link(stream_index=1)
        self.run_pass()
        self.link(stream_index=0)
        log = self.run_pass()
        self.assertEqual(self.cfg(), ['0-3.0', '0-3.0/trashcan'], log)
        self.assertIn('0-3.1: removed stale stream', log)
        self.assertEqual(self.links()[0]['stream'], '/dev/tbstream1')

    def test_no_stream_without_the_peer_service_or_carrier(self):
        self.link(stream_index=None)
        self.run_pass()
        self.assertEqual(self.cfg(), [])
        self.assertEqual(self.links()[0]['stream'], '-')
        self.link(stream_index=1)
        self.run_pass()
        self.assertEqual(self.cfg(), ['0-3.1', '0-3.1/trashcan'])
        # The link loses its carrier: its stream goes.
        self.link(stream_index=1, carrier=0)
        self.run_pass()
        self.assertEqual(self.cfg(), [])
        self.assertEqual(self.links()[0]['state'], 'no_carrier')

    def test_keeps_hopids_copied_from_the_peer(self):
        self.link()
        self.env['MOCK_AUTOCONFIG'] = '1'
        log = self.run_pass()
        self.assertEqual(self.cfg(), ['0-3.1', '0-3.1/trashcan'], log)
        self.assertNotIn('cannot set', log)
        self.assertEqual(self.links()[0]['stream'], '/dev/tbstream0')

    def test_empty_table(self):
        (self.root / 'etc/tblink/nodes.tsv').write_text('# index, hostname, controller UUIDs by Apple bus\n')
        self.link()
        log = self.run_pass()
        self.assertEqual(self.cfg(), [], log)
        self.assertNotIn('ip addr add', (self.root / 'calls').read_text() if (self.root / 'calls').exists() else '')
        [row] = self.links()
        self.assertEqual((row['peer'], row['state']), ('?', 'unknown'))

    def test_peer_without_hopid_15(self):
        self.link(maxhopid=12)
        log = self.run_pass()
        self.assertEqual(self.cfg(), [])
        self.assertIn("peer's highest HopID is 12", log)


if __name__ == '__main__':
    unittest.main()
