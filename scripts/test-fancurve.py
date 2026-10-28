#!/usr/bin/env python3
"""Bounded local fan-control tests using files and mock cat/sleep, never sysfs. Every test runs on both applesmc
layouts: before Linux 7.3 (fan files in the device directory, fan1_manual 1/0, fan1_output) and from 7.3 (hwmon
attributes in hwmon/hwmon0, pwm1_enable 1 manual / 2 automatic, fan1_target). Records and actions use the old names
and the old 1/0 for manual mode; the mocks translate them to the layout under test. GPUs (drm/cardN with an amdgpu
driver link, temperature and DPM level), their debugfs fence files and coretemp are files under the test's root; an
action key with a slash writes (or with None removes, or with {'link': TARGET} symlinks) that file relative to the
root."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).with_name('fancurve.sh').resolve()
LAYOUTS = {
    'legacy': dict(FAN_SUBDIR='', FAN_MODE='fan1_manual', FAN_TARGET='fan1_output', FAN_ON='1', FAN_OFF='0'),
    'hwmon': dict(FAN_SUBDIR='hwmon/hwmon0', FAN_MODE='pwm1_enable', FAN_TARGET='fan1_target', FAN_ON='1', FAN_OFF='2'),
}
CAT = r'''#!/usr/bin/env python3
import os,pathlib,sys
e=os.environ
path=pathlib.Path(sys.argv[1])
try:value=path.read_text()
except OSError:sys.exit(1)
if e.get('FAULT_TARGET')=='1' and path.name==e['FAN_TARGET'] and value.strip()=='2360':value='2359\n'
if e.get('FAULT_MANUAL')=='1' and path.name==e['FAN_MODE'] and value.strip()==e['FAN_ON']:value=e['FAN_OFF']+'\n'
sys.stdout.write(value)
'''
SLEEP = r'''#!/usr/bin/env python3
import json,os,pathlib,signal,time
e=os.environ
root=pathlib.Path(e['SMC_PATH'])
fan=root/e['FAN_SUBDIR']
names={'fan1_max':'fan1_max','fan1_manual':e['FAN_MODE'],'fan1_output':e['FAN_TARGET']}
to_old={e['FAN_ON']:'1',e['FAN_OFF']:'0'}
to_new={'1':e['FAN_ON'],'0':e['FAN_OFF']}
def record_value(key):
 path=fan/names[key]
 if not path.exists():return None
 value=path.read_text().strip()
 return to_old.get(value,value) if key=='fan1_manual' else value
state=root/'polls.jsonl'
rows=state.read_text().splitlines() if state.exists() else []
record={k:record_value(k) for k in names}
with state.open('a') as f:f.write(json.dumps(record)+'\n')
actions=json.loads(e['FAN_ACTIONS'])
action=actions[len(rows)] if len(rows)<len(actions) else {'stop':True}
for key,value in action.items():
 if key=='stop':continue
 path=root.parent/key if '/' in key else fan/names[key]
 if value is None:path.unlink(missing_ok=True)
 elif isinstance(value,dict):path.symlink_to(value['link'])
 else:path.write_text((to_new.get(str(value),str(value)) if key=='fan1_manual' else str(value))+'\n')
if action.get('stop'):os.kill(os.getppid(),signal.SIGTERM)
else:time.sleep(.01)
'''


def fences(n, gfx=0x35):
    """amdgpu_fence_info with the gfx ring at fence gfx and a compute ring at fence n."""
    return (f'--- ring 0 (gfx) ---\nLast signaled fence          0x{gfx:08x}\nLast emitted                 0x{gfx:08x}\n'
            f'--- ring 1 (comp_1.0.0) ---\nLast signaled fence          0x{n:08x}\nLast emitted                 0x{n:08x}\n'
            '--- ring 3 (sdma0) ---\nLast signaled fence          0x00000007\nLast emitted                 0x00000007')


class FanPolicyTest(unittest.TestCase):
    LAYOUT = 'legacy'

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='fan-policy-test-')
        self.root = Path(self.temp.name)
        self.smc = self.root / 'smc'
        self.layout = LAYOUTS[self.LAYOUT]
        self.fan = self.smc / self.layout['FAN_SUBDIR']
        self.fan.mkdir(parents=True)
        for name, value in {'fan1_min': 790, 'fan1_max': 2360, self.layout['FAN_MODE']: self.layout['FAN_OFF'],
                            self.layout['FAN_TARGET']: 790}.items():
            (self.fan / name).write_text(str(value) + '\n')
        mockbin = self.root / 'bin'
        mockbin.mkdir()
        for name, source in [('cat', CAT), ('sleep', SLEEP)]:
            file = mockbin / name
            file.write_text(source)
            file.chmod(0o755)
        for name in ['drm', 'dri', 'hwmon']:
            (self.root / name).mkdir()
        self.env = dict(os.environ, SMC_PATH=str(self.smc), PATH=str(mockbin) + os.pathsep + os.environ['PATH'],
                        DRM_PATH=str(self.root / 'drm'), DRI_DEBUG_PATH=str(self.root / 'dri'),
                        HWMON_PATH=str(self.root / 'hwmon'), **self.layout)

    def write(self, rel, value):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(value) + '\n')

    def sensors(self, gpus=(40, 40), cpu=45, level='auto', debugfs=True):
        """Two amdgpu GPUs at these temperatures and DPM level, their fence files, and coretemp."""
        for i, t in enumerate(gpus):
            dev = self.root / f'drm/card{i}/device'
            dev.mkdir(parents=True)
            (dev / 'driver').symlink_to('../../../bus/pci/drivers/amdgpu')
            self.write(f'drm/card{i}/device/hwmon/hwmon{i + 1}/temp1_input', t * 1000)
            self.write(f'drm/card{i}/device/power_dpm_force_performance_level', level)
            if debugfs:
                self.write(f'dri/{i}/amdgpu_fence_info', fences(10))
        (self.root / 'drm/card0-DP-1').mkdir()
        self.write('hwmon/hwmon5/name', 'coretemp')
        self.write('hwmon/hwmon5/temp1_input', cpu * 1000)

    def tearDown(self):
        self.temp.cleanup()

    def run_policy(self, actions=None, **extra):
        env = dict(self.env, FAN_ACTIONS=json.dumps(actions or [{'stop': True}]), **extra)
        result = subprocess.run(['/bin/bash', str(SCRIPT)], env=env, capture_output=True, text=True, timeout=30)
        path = self.smc / 'polls.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        self.assertEqual((self.fan / self.layout['FAN_MODE']).read_text().strip(), self.layout['FAN_OFF'],
                         'every stop/failure must restore automatic SMC')
        return result, rows

    def test_always_uses_maximum_without_temperature_sensors(self):
        result, rows = self.run_policy()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(rows[0]['fan1_output'], '2360')
        self.assertEqual(rows[0]['fan1_manual'], '1')

    def test_follows_live_limit_increase_and_decrease(self):
        (self.fan / 'fan1_max').write_text('1900\n')
        result, rows = self.run_policy([{'fan1_max': 2360}, {'fan1_max': 2200}, {'stop': True}], FAN_RPM='0')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([r['fan1_output'] for r in rows], ['1900', '2360', '2200'])

    def test_floor_applies_below_the_live_limit_and_yields_above_it(self):
        (self.fan / 'fan1_max').write_text('1900\n')
        result, rows = self.run_policy([{'fan1_max': 2382}, {'fan1_max': 2200}, {'stop': True}])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([r['fan1_output'] for r in rows], ['2350', '2382', '2350'])
        self.assertIn('fan target 2350 rpm (temperature; live SMC maximum 1900)', result.stdout)

    def test_floor_is_configurable(self):
        (self.fan / 'fan1_max').write_text('1900\n')
        result, rows = self.run_policy(FAN_RPM='2300')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(rows[0]['fan1_output'], '2300')

    def test_floor_above_cap_is_rejected(self):
        for value in ['2401', '-1', '23x0', '02350']:
            result, rows = self.run_policy(FAN_RPM=value)
            self.assertEqual(result.returncode, 1, value)
            self.assertFalse(rows)
            self.assertIn('FAN_RPM must be 0..2400', result.stderr)

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
        (self.fan / 'fan1_max').unlink()
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

    def test_idle_and_cool_runs_at_idle_speed(self):
        self.sensors()
        result, rows = self.run_policy()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(rows[0]['fan1_output'], '1200')
        self.assertIn('fan target 1200 rpm (temperature; live SMC maximum 2360)', result.stdout)

    def test_temperatures_between_cool_and_hot_scale_linearly(self):
        self.sensors(gpus=(40, 60), cpu=70)  # GPU 10 of 15 degrees (666 per mille), CPU 10 of 20 (500)
        result, rows = self.run_policy()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(rows[0]['fan1_output'], str(1200 + (2360 - 1200) * 666 // 1000))

    def test_hot_cpu_alone_runs_full_speed(self):
        self.sensors(cpu=85)
        result, rows = self.run_policy()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(rows[0]['fan1_output'], '2360')

    def test_gpu_work_runs_full_speed_for_the_hold_then_steps_down(self):
        self.sensors()
        # INTERVAL 10: the 120 s hold is 12 polls. Fences advance before poll 2 only.
        actions = [{'dri/1/amdgpu_fence_info': fences(11)}] + [{}] * 24 + [{'stop': True}]
        result, rows = self.run_policy(actions, INTERVAL='10')
        self.assertEqual(result.returncode, 0, result.stderr)
        out = [int(r['fan1_output']) for r in rows]
        self.assertEqual(out[0], 1200)
        self.assertEqual(out[1:13], [2360] * 12)
        self.assertEqual(out[13:17], [2260, 2160, 2060, 1960])
        self.assertEqual(out[-1], 1200)
        # Start, work, back to the temperatures, settled.
        self.assertEqual(result.stdout.count('fan target '), 4, result.stdout)
        self.assertIn('fan target 1200 rpm (temperature;', [l for l in result.stdout.splitlines() if 'fan target' in l][-1])

    def test_gfx_work_alone_does_not_count(self):
        self.sensors()
        actions = [{'dri/0/amdgpu_fence_info': fences(10, gfx=0x40)}, {'dri/0/amdgpu_fence_info': fences(10, gfx=0x50)},
                   {'stop': True}]
        result, rows = self.run_policy(actions)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([r['fan1_output'] for r in rows], ['1200', '1200', '1200'])

    def test_gpus_bound_after_start_are_found(self):
        # At boot the service can start before amdgpu binds the GPUs: their files exist, the driver link not yet.
        self.sensors()
        for i in (0, 1):
            (self.root / f'drm/card{i}/device/driver').unlink()
        link = {'link': '../../../bus/pci/drivers/amdgpu'}
        actions = [{}, {'drm/card0/device/driver': link, 'drm/card1/device/driver': link}] + [{}] * 13 + [{'stop': True}]
        result, rows = self.run_policy(actions)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('no amdgpu GPU yet', result.stderr)
        self.assertIn('found 2 amdgpu GPU(s) and the CPU sensor', result.stdout)
        out = [int(r['fan1_output']) for r in rows]
        self.assertEqual(out[:3], [2360, 2360, 2260])   # full speed until found, then down in steps
        self.assertEqual(out[-1], 1200)
        self.assertNotIn('(work;', result.stdout)        # finding the GPUs is not GPU work

    def test_dpm_level_high_counts_as_work(self):
        self.sensors(level='high', debugfs=False)
        result, rows = self.run_policy([{}, {'stop': True}])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([r['fan1_output'] for r in rows], ['2360', '2360'])
        self.assertIn('(work;', result.stdout)

    def test_without_debugfs_the_temperatures_still_count(self):
        self.sensors(debugfs=False)
        result, rows = self.run_policy()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(rows[0]['fan1_output'], '1200')

    def test_a_temperature_that_cannot_be_read_runs_full_speed(self):
        self.sensors()
        result, rows = self.run_policy([{'drm/card1/device/hwmon/hwmon2/temp1_input': None}, {'stop': True}])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([r['fan1_output'] for r in rows], ['1200', '2360'])

    def test_idle_speed_never_below_the_smc_minimum(self):
        self.sensors()
        (self.fan / 'fan1_min').write_text('1300\n')
        result, rows = self.run_policy()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(rows[0]['fan1_output'], '1300')


class HwmonFanPolicyTest(FanPolicyTest):
    """The same tests on applesmc's hwmon layout (Linux 7.3 and later)."""
    LAYOUT = 'hwmon'


if __name__ == '__main__':
    unittest.main()
