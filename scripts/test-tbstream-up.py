#!/usr/bin/env python3
"""Local configfs/SSH mocks for tbstream-up.sh; never contacts a host or uses sudo.

The remote Bash runs unchanged except for redirecting sysfs/device paths into a
fixture and accepting fixture regular files in place of character devices.
"""
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).with_name('tbstream-up.sh').resolve()
MOCK = r'''#!/usr/bin/env python3
import json, os, pathlib, shlex, subprocess, sys
root=pathlib.Path(os.environ['MOCK_ROOT'])
kind=pathlib.Path(sys.argv[0]).name
args=sys.argv[1:];host=os.environ.get('MOCK_HOST','')
def log(op,values):
 with (root/'calls.jsonl').open('a') as f:f.write(json.dumps([host,op,values])+'\n')
if kind=='ssh':
 host=args.pop(0);command=shlex.split(' '.join(args));log('ssh',command)
 if command[:3]==['sudo','-n','modprobe']:sys.exit(0)
 if command==['/usr/local/sbin/tblink','links']:
  peer='b' if host=='a' else 'a'
  print('xdomain\tpeer\tlocal_bus\tpeer_bus\tlanes\tstate\tiface\taddress\tpeer_address')
  print('0-3\t'+peer+'\t0\t0\t2\tup\ttb0\t1\t2');sys.exit(0)
 assert command[:3]==['bash','-s','--'],command
 script=sys.stdin.read()
 if os.environ.get('MOCK_FAIL_HOST')==host and 'want_in=$3' in script:sys.exit(23)
 script=script.replace('/sys/',str(root/host/'sys')+'/').replace('/dev/tbstream',str(root/host/'dev/tbstream'))
 script=script.replace('[ -c "$device" ]','[ -f "$device" ]').replace('[ ! -c "$device" ]','[ ! -f "$device" ]')
 env=dict(os.environ,MOCK_HOST=host)
 sys.exit(subprocess.run(['/bin/bash',*command[1:]],input=script,text=True,env=env).returncode)
if kind=='readlink':
 path=root/host/'discovery-count';count=int(path.read_text())+1 if path.exists() else 1;path.write_text(str(count))
 if count<=int(os.environ.get('MOCK_DISCOVERY_DELAY','0')):sys.exit(1)
 print('/drivers/thunderbolt_stream');sys.exit(0)
if kind=='sleep':log('sleep',args);sys.exit(0)
assert kind=='sudo',kind
assert args.pop(0)=='-n'
op=args.pop(0);log(op,args)
if op=='test':sys.exit(subprocess.run(['/bin/test',*args]).returncode)
if op=='mkdir':
 path=pathlib.Path(args[-1]);path.mkdir(parents='-p' in args,exist_ok='-p' in args)
 if path.name=='xmac':
  for attr in ['in_hopid','out_hopid']:(path/attr).write_text('0\n')
  index='0' if host=='a' else '3'
  if os.environ.get('MOCK_BAD_INDEX')==host:index='-1'
  (path/'index').write_text(index+'\n')
  if os.environ.get('MOCK_MISSING_DEVICE')!=host:
   dev=root/host/'dev'/('tbstream'+index);dev.parent.mkdir(exist_ok=True);dev.touch();dev.chmod(0)
elif op=='cat':sys.stdout.write(pathlib.Path(args[0]).read_text())
elif op=='tee':
 value=sys.stdin.read().strip();path=pathlib.Path(args[0])
 if value=='-1':value='5' if path.name=='in_hopid' else '7'
 path.write_text(value+'\n');print(value)
elif op=='chgrp':pass
elif op=='chmod':pathlib.Path(args[-1]).chmod(int(args[0],8))
else:raise AssertionError((op,args))
'''


class StreamSetupTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='tbstream-mock-')
        self.root = Path(self.temp.name)
        mockbin = self.root / 'bin'
        mockbin.mkdir()
        for command in ['ssh', 'sudo', 'readlink', 'sleep']:
            path = mockbin / command
            path.write_text(MOCK)
            path.chmod(0o755)
        for host in ['a', 'b']:
            (self.root / host / 'sys/bus/thunderbolt/devices/0-3.1').mkdir(parents=True)
            (self.root / host / 'dev').mkdir()
        self.env = dict(os.environ, MOCK_ROOT=str(self.root), PATH=str(mockbin) + os.pathsep + os.environ['PATH'])

    def tearDown(self):
        self.temp.cleanup()

    def run_setup(self, **extra):
        return subprocess.run(['/bin/bash', str(SCRIPT), 'a', 'b'], capture_output=True, text=True, env=dict(self.env, **extra), timeout=20)

    def calls(self, op):
        return [r for r in map(json.loads, (self.root / 'calls.jsonl').read_text().splitlines()) if r[1] == op]

    def clear_calls(self):
        (self.root / 'calls.jsonl').write_text('')

    def group(self, host):
        return self.root / host / 'sys/kernel/config/thunderbolt/stream/0-3.1/xmac'

    def initialize(self):
        result = self.run_setup()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('/dev/tbstream0 (in 5 out 7) <-> b 0-3.1/xmac /dev/tbstream3', result.stdout)
        return result

    def test_new_stream_gets_complementary_hopids_and_exact_device_permissions(self):
        self.initialize()
        self.assertEqual((self.group('b') / 'in_hopid').read_text().strip(), '7')
        self.assertEqual((self.group('b') / 'out_hopid').read_text().strip(), '5')
        changes = self.calls('chmod')
        self.assertEqual([r[2][0] for r in changes], ['660', '660'])
        self.assertEqual([r[2][-1] for r in changes], [str(self.root / 'a/dev/tbstream0'), str(self.root / 'b/dev/tbstream3')])
        self.assertEqual(len(self.calls('chgrp')), 2)

    def test_delayed_service_advertisement_is_retried(self):
        result = self.run_setup(MOCK_DISCOVERY_DELAY='2')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.calls('sleep')), 4)

    def test_missing_service_has_bounded_timeout(self):
        result = self.run_setup(MOCK_DISCOVERY_DELAY='100')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('after 15 seconds', result.stderr)
        self.assertEqual(len(self.calls('sleep')), 15)
        self.assertFalse(self.calls('mkdir'))

    def test_rerun_preserves_hopids_and_existing_user_access(self):
        self.initialize()
        # Existing owner/ACL access need not be changed to the default group mode.
        device = self.root / 'a/dev/tbstream0'
        device.chmod(0o600)
        self.clear_calls()
        result = self.run_setup()
        self.assertEqual(result.returncode, 0, result.stderr)
        for op in ['tee', 'chmod', 'chgrp', 'mkdir']:
            self.assertFalse(self.calls(op), op)
        self.assertEqual(device.stat().st_mode & 0o777, 0o600)

    def test_existing_peer_mismatch_fails_without_writes(self):
        self.initialize()
        (self.group('b') / 'in_hopid').write_text('99\n')
        self.clear_calls()
        result = self.run_setup()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('expected (7, 5); left unchanged', result.stderr)
        self.assertFalse(self.calls('tee'))
        self.assertEqual((self.group('b') / 'in_hopid').read_text(), '99\n')

    def test_existing_incomplete_allocator_is_not_rewritten(self):
        self.initialize()
        (self.group('a') / 'out_hopid').write_text('0\n')
        self.clear_calls()
        result = self.run_setup()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('incomplete HopIDs', result.stderr)
        self.assertFalse(self.calls('tee'))

    def test_remote_failure_propagates(self):
        result = self.run_setup(MOCK_FAIL_HOST='b')
        self.assertEqual(result.returncode, 23)
        self.assertNotIn('bus 0:', result.stdout)

    def test_invalid_device_index_never_changes_permissions(self):
        result = self.run_setup(MOCK_BAD_INDEX='a')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('valid HopIDs and device index', result.stderr)
        self.assertFalse(self.calls('chmod'))
        self.assertFalse(self.calls('chgrp'))

    def test_missing_device_has_bounded_timeout(self):
        result = self.run_setup(MOCK_MISSING_DEVICE='a')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('missing character device', result.stderr)
        self.assertEqual(len(self.calls('sleep')), 15)
        self.assertFalse(self.calls('chmod'))


if __name__ == '__main__':
    unittest.main()
