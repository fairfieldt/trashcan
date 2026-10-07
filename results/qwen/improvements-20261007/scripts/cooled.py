#!/usr/bin/env python3
"""Measure one configuration on leased machines, unload between configs, and preserve exact settings."""
import argparse, re, shlex, subprocess, time
ROOT='/home/fairfieldt/trashcan-exp/improvements-20261007'
MODEL='/home/fairfieldt/models/qwen3.8-27b/Qwen3.8-27B-trashcan-Q4_0.gguf'
MTP='/home/fairfieldt/models/qwen3.8-27b/mtp-Qwen3.8-27B-trashcan-Q4_0.gguf'
STREAM='/dev/tbstream0,/dev/tbstream1,/dev/tbstream2'
p=argparse.ArgumentParser();p.add_argument('--incite',type=int,required=True);p.add_argument('--invite',type=int,required=True)
p.add_argument('--binary',default='clean-candidates');p.add_argument('--name',required=True);p.add_argument('--env',default='');p.add_argument('--command',default='qwen-bench');p.add_argument('--args',default='--pp 512 --tg 128 --reps 2');p.add_argument('--mtp',action='store_true');p.add_argument('--profile',action='store_true')
a=p.parse_args()
if not re.fullmatch(r"[A-Za-z0-9_-]+",a.name): p.error("name must contain letters, digits, underscores or hyphens")
def run(host,cmd,check=True):return subprocess.run(['ssh',host,cmd],check=check)
def leased(ticket,cmd):return 'gpuq run '+str(ticket)+' -- bash -c '+shlex.quote('cd '+ROOT+'; '+cmd)
def stop():run('invite','pkill -TERM -f '+shlex.quote('^'+re.escape(ROOT+'/'+a.binary)+' qwen-tp-node '),False)
nodeenv=a.env
if a.profile:nodeenv+=' TRASHCAN_TP_PROFILE='+ROOT+'/results/profile-node-'+a.name+'.jsonl TRASHCAN_TP_PROFILE_DISPATCH=1'
node=shlex.join(['scripts/with-clocks.sh','results/clocks/node-'+a.name+'.txt','sudo','-n','env',*shlex.split(nodeenv),'python3','scripts/latency-run.py',ROOT+'/'+a.binary,'qwen-tp-node',MODEL,'--mtp',MTP,'--stream',STREAM])+' > '+shlex.quote('results/node-'+a.name+'.log')+' 2>&1'
child=subprocess.Popen(['ssh','invite',leased(a.invite,node)])
try:
 time.sleep(2)
 env=a.env
 if a.profile:env+=' TRASHCAN_TP_PROFILE='+ROOT+'/results/profile-driver-'+a.name+'.jsonl TRASHCAN_TP_PROFILE_DISPATCH=1'
 args=shlex.split(a.args)+(['--mtp',MTP] if a.mtp else [])
 command=shlex.join(['scripts/run.sh',a.name,'./'+a.binary,a.command,MODEL,'--tp-stream',STREAM,*args])
 run('incite',leased(a.incite,'ENV='+shlex.quote(env)+' '+command))
finally:
 stop(); child.wait(timeout=20)
