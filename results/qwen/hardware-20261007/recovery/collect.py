#!/usr/bin/env python3
"""Read-only recovery collection after the October 7 shared power interruption."""
import concurrent.futures, datetime, json, pathlib, shlex, subprocess, sys, tarfile
ROOT=pathlib.Path(__file__).resolve().parent
OLD={'incite':'c7692794-b9bd-44fe-bfe8-2401ddd8f04d','invite':'17105e70-09ac-496f-bda1-6d03e2a72fca'}
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','-o','ConnectionAttempts=1','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=2','-o','ControlMaster=no','-o','ControlPath=none']
def collect(host):
 out=ROOT/host;out.mkdir(exist_ok=True)
 commands=[
 ('state.txt','date -u; hostname; cat /proc/sys/kernel/random/boot_id; uname -a; uptime; for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do printf "%s %s\\n" "$f" "$(cat "$f")"; done; systemctl show system-crash-capture.service system-platform-trace.service -p ActiveState -p SubState -p UnitFileState -p NRestarts -p ExecMainStatus -p ActiveEnterTimestamp; ls -l /dev/tbstream*; for f in /sys/bus/thunderbolt/devices/*/rx_speed /sys/bus/thunderbolt/devices/*/tx_speed /sys/bus/thunderbolt/devices/*/rx_lanes /sys/bus/thunderbolt/devices/*/tx_lanes; do test ! -f "$f" || { printf "%s " "$f"; cat "$f"; }; done; sha256sum /home/fairfieldt/trashcan-exp/prefill-lds-20261007/trashcan'),
 ('boots.txt','sudo -n journalctl --list-boots --no-pager'),
 ('interrupted-workspace.tar.gz','tar -czf - -C /home/fairfieldt/trashcan-exp/prefill-lds-20261007 results slab-governors.before results-micro.err results-micro.jsonl results-slab.err results-slab.jsonl scripts'),
 ('prior-kernel-journal.log','sudo -n journalctl -k -b '+OLD[host].replace('-','')+' --no-pager -o short-iso-precise'),
 ('prior-system-journal.log','sudo -n journalctl -b '+OLD[host].replace('-','')+' --since "2026-10-07 12:05:00" --no-pager -o short-iso-precise'),
 ('current-kernel-journal.log','sudo -n journalctl -k -b --no-pager -o short-iso-precise'),
 ('recorder-journal.log','sudo -n journalctl -b -u system-crash-capture.service -u system-platform-trace.service --no-pager -o short-iso-precise'),
 ('recorders.tar.gz','sudo -n tar -czf - -C /var/log trashcan-crash trashcan-platform'),
 ('pstore-and-dumps.txt','sudo -n ls -laR /sys/fs/pstore /var/log/trashcan-crash; sudo -n tail -n 2 /var/log/trashcan-crash/telemetry.jsonl; sudo -n tail -n 5 /var/log/trashcan-platform/events.jsonl')]
 for filename,command in commands:
  start=datetime.datetime.now(datetime.timezone.utc).isoformat()
  argv=SSH+[host,command]
  try:
   result=subprocess.run(argv,capture_output=True,timeout=45)
   (out/filename).write_bytes(result.stdout)
   if result.stderr:(out/(filename+'.stderr')).write_bytes(result.stderr)
   record=dict(utc=start,file=filename,command=shlex.join(argv),returncode=result.returncode,bytes=len(result.stdout),stderr=result.stderr.decode(errors='replace')[:500])
  except subprocess.TimeoutExpired as error:
   record=dict(utc=start,file=filename,command=shlex.join(argv),error='timeout after 45s')
  with (out/'commands.jsonl').open('a') as log:log.write(json.dumps(record)+'\n')
  print(host,filename,record.get('returncode',record.get('error')),record.get('bytes',0),flush=True)
  if filename=='state.txt' and record.get('returncode')!=0:break
 for filename in ['interrupted-workspace.tar.gz']:
  path=out/filename
  if not path.exists() or not path.stat().st_size:continue
  target=out/filename.removesuffix('.tar.gz');target.mkdir(exist_ok=True)
  with tarfile.open(path) as archive:
   for member in archive.getmembers():
    dest=(target/member.name).resolve()
    if not dest.is_relative_to(target.resolve()) or member.issym() or member.islnk():raise RuntimeError('unsafe archive member')
   archive.extractall(target,filter='data')
 return host
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  print(list(pool.map(collect,sys.argv[1:] or ['incite','invite'])))
