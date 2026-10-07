#!/usr/bin/env python3
"""Acquire paired leases only after cooling; restore governors and release after every measurement."""
import argparse,json,pathlib,re,shlex,subprocess,time
p=argparse.ArgumentParser();p.add_argument('plan',type=pathlib.Path);p.add_argument('--max-c',type=int,default=82);p.add_argument('--binary',default='clean-candidates');a=p.parse_args()
root='/home/fairfieldt/trashcan-exp/improvements-20261007';scripts=pathlib.Path(__file__).resolve().parent
owner='Qwen27B improvements 1-5'
def shell(args,check=True):return subprocess.run(args,check=check,capture_output=True,text=True)
def remote(host,cmd):return shell(['ssh',host,cmd]).stdout
for case in json.loads(a.plan.read_text()):
 while True:
  temps=[int(v)/1000 for host in ['incite','invite'] for v in remote(host,'cat /sys/class/drm/card0/device/hwmon/hwmon*/temp1_input /sys/class/drm/card1/device/hwmon/hwmon*/temp1_input').split()]
  if max(temps)<=a.max_c:break
  print(case['name'],'cooling',temps,flush=True);time.sleep(20)
 tickets={}
 success=False
 try:
  for host in ['incite','invite']:
   result=shell(['gpuq','--on',host,'request',case['name'],'--est',str(case.get('minutes',4)),'--owner',owner]).stdout
   print(result,flush=True);tickets[host]=int(re.search(r'Ticket #(\d+)',result)[1])
  for host,ticket in tickets.items():print(shell(['gpuq','--on',host,'wait',str(ticket)]).stdout,flush=True)
  for host,ticket in tickets.items():
   script='cd '+root+'; for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do printf "%s %s\\n" "$f" "$(cat "$f")"; done > governors.before; for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance | sudo -n tee "$f" >/dev/null; done'
   remote(host,'gpuq run '+str(ticket)+' -- bash -c '+shlex.quote(script))
  cmd=['python3',str(scripts/'cooled.py'),'--incite',str(tickets['incite']),'--invite',str(tickets['invite']),'--binary',a.binary,'--name',case['name'],'--env',case.get('env',''),'--command',case.get('command','qwen-bench'),'--args',case['args']]
  if case.get('mtp'):cmd.append('--mtp')
  if case.get('profile'):cmd.append('--profile')
  print(case['name'],flush=True);subprocess.run(cmd,check=True);success=True
 finally:
  for host,ticket in tickets.items():
   script='cd '+root+'; while read -r f gov; do echo "$gov" | sudo -n tee "$f" >/dev/null; done < governors.before'
   shell(['ssh',host,'gpuq run '+str(ticket)+' -- bash -c '+shlex.quote(script)],False)
   print(shell(['gpuq','--on',host,'release',str(ticket),'-r',case['name']+(' completed; clocks and outputs saved' if success else ' failed; diagnostics saved')]).stdout,flush=True)
