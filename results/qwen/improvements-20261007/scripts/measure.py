#!/usr/bin/env python3
"""Run paired 27B measurements on leased Macs; all GPU commands stay inside gpuq run."""
import argparse, shlex, subprocess, time
ROOT='/home/fairfieldt/trashcan-exp/improvements-20261007'
MODEL='/home/fairfieldt/models/qwen3.8-27b/Qwen3.8-27B-trashcan-Q4_0.gguf'
MTP='/home/fairfieldt/models/qwen3.8-27b/mtp-Qwen3.8-27B-trashcan-Q4_0.gguf'
STREAM='/dev/tbstream0,/dev/tbstream1,/dev/tbstream2'
p=argparse.ArgumentParser(); p.add_argument('--incite',type=int,required=True);p.add_argument('--invite',type=int,required=True)
p.add_argument('--resident',action='store_true');p.add_argument('--binary',default='workers-kv');p.add_argument('--modes',default='old,pool,kv,both');p.add_argument('--tag',default='ab');p.add_argument('--reps',type=int,default=2)
a=p.parse_args()
def ssh(host,cmd,**kw): return subprocess.run(['ssh',host,cmd],check=True,**kw)
def leased(host,ticket,cmd): return 'gpuq run '+str(ticket)+' -- bash -c '+shlex.quote('cd '+ROOT+'; '+cmd)
def stop():
    subprocess.run(['ssh','invite','pkill -TERM -f '+shlex.quote('^'+ROOT+'/'+a.binary+' qwen-tp-node ')],check=False)
stop()
for phase,mode in enumerate(a.modes.split(',')):
    switches={'old':'TRASHCAN_TP_WORKERS=0 TRASHCAN_TP_KV_ONLY=0','pool':'TRASHCAN_TP_WORKERS=1 TRASHCAN_TP_KV_ONLY=0','kv':'TRASHCAN_TP_WORKERS=0 TRASHCAN_TP_KV_ONLY=1','both':'TRASHCAN_TP_WORKERS=1 TRASHCAN_TP_KV_ONLY=1','pinned':'TRASHCAN_TP_WORKERS=1 TRASHCAN_TP_KV_ONLY=1 TRASHCAN_TP_CPUS=2,4,6,8,10,0','prefetch':'TRASHCAN_QWEN_MV=prefetch','rows2':'TRASHCAN_QWEN_VERIFY=rows2','tiled':'TRASHCAN_QWEN_ATTN=tiled','compact':'TRASHCAN_QWEN_Q4=compact','all':'TRASHCAN_QWEN_Q4=compact TRASHCAN_QWEN_ATTN=tiled TRASHCAN_TP_CPUS=2,4,6,8,10,0'}.get(mode,mode)
    name=a.tag+'-'+str(phase)+'-'+mode.replace('=','').replace(' ','-')
    node='scripts/with-clocks.sh results/clocks/node-'+name+'.txt sudo -n env '+switches+' python3 scripts/latency-run.py '+ROOT+'/'+a.binary+' qwen-tp-node '+MODEL+' --mtp '+MTP+' --stream '+STREAM+' > results/node-'+name+'.log 2>&1'
    child=subprocess.Popen(['ssh','invite',leased('invite',a.invite,node)])
    try:
        time.sleep(2)
        for rep in range(1 if a.resident else a.reps):
            for label,args in [('pp512','--pp 512 --tg 128'),('pp2048','--pp 2048 --tg 32'),('mtp','--pp 512 --tg 128 --mtp '+MTP)]:
                run=name+'-'+label+'-'+str(rep)
                print(run,flush=True)
                cmd='ENV='+shlex.quote(switches)+' scripts/run.sh '+run+' ./'+a.binary+' qwen-bench '+MODEL+' --tp-stream '+STREAM+' '+args+(' --reps '+str(a.reps) if a.resident else '')
                ssh('incite',leased('incite',a.incite,cmd))
    finally:
        stop()
        child.wait(timeout=15)
