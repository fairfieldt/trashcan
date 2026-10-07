#!/usr/bin/env python3
"""Bounded local fan-control tests using files and mock cat/sleep, never sysfs."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).with_name('fancurve.sh').resolve()
CAT = r'''#!/usr/bin/env python3
import os,pathlib,sys
path=pathlib.Path(sys.argv[1])
try:value=path.read_text()
except OSError:sys.exit(1)
if os.environ.get('FAULT_TARGET')=='1' and path.name=='fan1_output' and value.strip()=='2360':value='2359\n'
if os.environ.get('FAULT_MANUAL')=='1' and path.name=='fan1_manual' and value.strip()=='1':value='0\n'
sys.stdout.write(value)
'''
SLEEP = r'''#!/usr/bin/env python3
import json,os,pathlib,signal,time
root=pathlib.Path(os.environ['SMC_PATH'])
state=root/'polls.jsonl'
rows=state.read_text().splitlines() if state.exists() else []
record={k:(root/k).read_text().strip() if (root/k).exists() else None for k in ['fan1_max','fan1_manual','fan1_output']}
with state.open('a') as f:f.write(json.dumps(record)+'\n')
actions=json.loads(os.environ['FAN_ACTIONS'])
action=actions[len(rows)] if len(rows)<len(actions) else {'stop':True}
for key,value in action.items():
 if key=='stop':continue
 path=root/key
 if value is None:path.unlink(missing_ok=True)
 else:path.write_text(str(value)+'\n')
if action.get('stop'):os.kill(os.getppid(),signal.SIGTERM)
else:time.sleep(.01)
'''


class FanPolicyTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='fan-policy-test-')
        self.root = Path(self.temp.name)
        self.smc = self.root / 'smc'
        self.smc.mkdir()
        for name, value in {'fan1_min': 790, 'fan1_max': 2360, 'fan1_manual': 0, 'fan1_output': 790}.items():
            (self.smc / name).write_text(str(value) + '\n')
        mockbin = self.root / 'bin'
        mockbin.mkdir()
        for name, source in [('cat', CAT), ('sleep', SLEEP)]:
            file = mockbin / name
            file.write_text(source)
            file.chmod(0o755)
        self.env = dict(os.environ, SMC_PATH=str(self.smc), PATH=str(mockbin) + os.pathsep + os.environ['PATH'])

    def tearDown(self):
        self.temp.cleanup()

    def run_policy(self, actions=None, **extra):
        env = dict(self.env, FAN_ACTIONS=json.dumps(actions or [{'stop': True}]), **extra)
        result = subprocess.run(['/bin/bash', str(SCRIPT)], env=env, capture_output=True, text=True, timeout=5)
        path = self.smc / 'polls.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        self.assertEqual((self.smc / 'fan1_manual').read_text().strip(), '0', 'every stop/failure must restore automatic SMC')
        return result, rows

    def test_always_uses_maximum_without_temperature_sensors(self):
        result, rows = self.run_policy()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(rows[0]['fan1_output'], '2360')
        self.assertEqual(rows[0]['fan1_manual'], '1')

    def test_follows_live_limit_increase_and_decrease(self):
        (self.smc / 'fan1_max').write_text('1900\n')
        result, rows = self.run_policy([{'fan1_max': 2360}, {'fan1_max': 2200}, {'stop': True}])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([r['fan1_output'] for r in rows], ['1900', '2360', '2200'])

    def test_reasserts_manual_mode_and_target_independently(self):
        result, rows = self.run_policy([{'fan1_manual': 0, 'fan1_output': 900}, {'fan1_output': 1900}, {'stop': True}])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(all(r['fan1_manual'] == '1' and r['fan1_output'] == '2360' for r in rows))

    def test_two_transient_max_read_failures_retain_valid_target(self):
        result, rows = self.run_policy([{'fan1_max': None}, {}, {'fan1_max': 2362}, {'stop': True}])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([r['fan1_output'] for r in rows], ['2360', '2360', '2360', '2362'])
        self.assertEqual(result.stderr.count('temporarily unreadable'), 1)
        self.assertEqual(result.stdout.count('readable again'), 1)

    def test_persistent_max_read_failure_exits_for_service_retry(self):
        result, rows = self.run_policy([{'fan1_max': None}, {}, {}, {'stop': True}])
        self.assertEqual(result.returncode, 1)
        self.assertEqual(len(rows), 3)
        self.assertIn('unreadable for three polls', result.stderr)

    def test_startup_read_failure_has_no_cached_fallback(self):
        (self.smc / 'fan1_max').unlink()
        result, rows = self.run_policy()
        self.assertEqual(result.returncode, 1)
        self.assertFalse(rows)
        self.assertIn('unavailable at startup', result.stderr)

    def test_oversized_interval_cannot_wrap_into_allowed_range(self):
        result, rows = self.run_policy(INTERVAL='18446744073709551617')
        self.assertEqual(result.returncode, 1)
        self.assertFalse(rows)
        self.assertIn('INTERVAL must be 1..10 seconds', result.stderr)

    def test_malformed_limit_returns_to_automatic_immediately(self):
        result, rows = self.run_policy([{'fan1_max': 'broken'}, {'stop': True}])
        self.assertEqual(result.returncode, 1)
        self.assertEqual(len(rows), 1)
        self.assertIn('invalid SMC maximum', result.stderr)

    def test_limit_below_minimum_returns_to_automatic(self):
        result, rows = self.run_policy([{'fan1_max': 0}, {'stop': True}])
        self.assertEqual(result.returncode, 1)
        self.assertEqual(len(rows), 1)
        self.assertIn('below minimum', result.stderr)

    def test_failed_target_readback_is_not_ignored(self):
        result, rows = self.run_policy(FAULT_TARGET='1')
        self.assertEqual(result.returncode, 1)
        self.assertFalse(rows)
        self.assertIn('target readback failed', result.stderr)

    def test_failed_manual_readback_is_not_ignored(self):
        result, rows = self.run_policy(FAULT_MANUAL='1')
        self.assertEqual(result.returncode, 1)
        self.assertFalse(rows)
        self.assertIn('manual control readback failed', result.stderr)

    def test_small_maximum_jitter_does_not_spam_logs(self):
        result, rows = self.run_policy([{'fan1_max': 2362}, {'fan1_max': 2359}, {'fan1_max': 2361}, {'stop': True}])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([r['fan1_output'] for r in rows], ['2360', '2362', '2359', '2361'])
        self.assertEqual(result.stdout.count('fan target '), 1)


if __name__ == '__main__':
    unittest.main()
