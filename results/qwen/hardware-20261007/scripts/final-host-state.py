#!/usr/bin/env python3
"""Read-only endpoint state after this task's jobs and leases finish."""
import collections,hashlib,json,os,pathlib,subprocess,time
root=pathlib.Path('/home/fairfieldt/trashcan-exp/tp4-offline-20261007')
smc=pathlib.Path('/sys/devices/platform/applesmc.768')
read=lambda p:p.read_text().strip()
procs=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  argv=p.joinpath('cmdline').read_bytes().rstrip(b'\0').split(b'\0')
  if argv and argv[0].decode(errors='replace') in [str(root/n) for n in ['trashcan','baseline','trashcan-ready','trashcan-final']]:
   procs.append(dict(pid=int(p.name),argv=[v.decode(errors='replace') for v in argv]))
 except (PermissionError,FileNotFoundError,ProcessLookupError):pass
state=dict(unix=time.time(),boot_id=read(pathlib.Path('/proc/sys/kernel/random/boot_id')),
 fan={k:int(read(smc/k)) for k in ['fan1_min','fan1_max','fan1_manual','fan1_output','fan1_input']},
 fan_script_sha256=hashlib.sha256(pathlib.Path('/usr/local/sbin/fancurve.sh').read_bytes()).hexdigest(),
 fan_service_sha256=hashlib.sha256(pathlib.Path('/etc/systemd/system/fancurve.service').read_bytes()).hexdigest(),
 fan_service=subprocess.check_output(['systemctl','show','fancurve.service','-p','ActiveState','-p','SubState','-p','NRestarts','-p','ExecMainStatus'],text=True).splitlines(),
 fan_enabled=subprocess.check_output(['systemctl','is-enabled','fancurve.service'],text=True).strip(),
 cpu_governors=dict(collections.Counter(read(p) for p in pathlib.Path('/sys/devices/system/cpu/cpufreq').glob('policy*/scaling_governor'))),
 experiment_model_processes=procs,
 streams={f'/dev/tbstream{i}':dict(exists=os.path.exists(f'/dev/tbstream{i}'),readable=os.access(f'/dev/tbstream{i}',os.R_OK),writable=os.access(f'/dev/tbstream{i}',os.W_OK)) for i in range(3)},
 final_binary_sha256=hashlib.sha256((root/'trashcan-final').read_bytes()).hexdigest())
print(json.dumps(state,indent=2))
