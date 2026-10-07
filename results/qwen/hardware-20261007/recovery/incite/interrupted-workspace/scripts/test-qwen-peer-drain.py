#!/usr/bin/env python3
"""Test a lost peer on two local ranks of a TP4-sized model, under a GPU lease."""
import argparse, json, pathlib, signal, socket, struct, subprocess, time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--binary',default='./drain-worker')
p.add_argument('--model',default='/home/fairfieldt/models/qwen3.8-27b/Qwen3.8-27B-trashcan-Q4_0.gguf')
p.add_argument('--out',default='drain-results')
a=p.parse_args();out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
with socket.socket() as listener:
    listener.bind(('127.0.0.1',0));port=listener.getsockname()[1]
log=out/'node.log'
with log.open('w') as stderr:
    proc=subprocess.Popen([str(pathlib.Path(a.binary).resolve()),'qwen-tp-node',a.model,
                           '--gpus','0,1','--listen',f'127.0.0.1:{port}'],stderr=stderr)

def wait_for(predicate,timeout=60):
    deadline=time.monotonic()+timeout
    while not predicate():
        assert proc.poll() is None, proc.returncode
        assert time.monotonic()<deadline, log.read_text()
        time.sleep(.01)

def recv(sock,n):
    buf=bytearray()
    while len(buf)<n:
        chunk=sock.recv(n-len(buf));assert chunk;buf.extend(chunk)
    return bytes(buf)

def send(sock,data):
    data=json.dumps(data).encode();sock.sendall(struct.pack('<I',len(data))+data)

records=[]
try:
    wait_for(lambda:'tp-node:' in log.read_text())
    for n,micro in [(1,1),(128,2),(1,1)]:
        prior=log.read_text().count('session ended')
        with socket.create_connection(('127.0.0.1',port),timeout=30) as sock:
            sock.sendall(struct.pack('<4I',0x454e414c,1,0,1))
            send(sock,dict(setup=dict(first=2,ranks=4,context=1024,batch=128,verify=0,mtp=False,draft_vocab=0)))
            reply=json.loads(recv(sock,struct.unpack('<I',recv(sock,4))[0]));assert reply.get('ok'),reply
            send(sock,dict(forward=dict(tokens=[1]*n,pos=0,verify=False,dump=False,micro=micro,threads=1)))
            recv(sock,1)
        begin=time.monotonic()
        wait_for(lambda:log.read_text().count('session ended')>prior,timeout=5)
        elapsed=time.monotonic()-begin
        assert elapsed<3 and 'cleanup failed' not in log.read_text()
        result=dict(tokens=n,micro=micro,seconds=elapsed,unix=time.time());records.append(result)
        print(json.dumps(result),flush=True)
    print('PASS: three lost-peer sessions drained and node accepted fresh sessions',flush=True)
finally:
    proc.send_signal(signal.SIGTERM)
    code=proc.wait(timeout=5)
    (out/'events.json').write_text(json.dumps(records,indent=2)+'\n')
    assert code==-signal.SIGTERM,code
