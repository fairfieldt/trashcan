#!/usr/bin/env python3
"""Boot a separate ESP and exercise real OpenSSH against Dropbear inside dross."""
from pathlib import Path
import os, sys, subprocess, time, tempfile, socket, shutil, pty, fcntl, termios, struct, select, signal, hashlib
here=Path(__file__).resolve().parent; out=here/'target'; os.chdir(here)
port=int(os.environ.get('SSH_PORT','2222'))
subprocess.run([sys.executable,'stage.py'],check=True)
firmware=None
for root in ['/opt/homebrew/share/qemu','/usr/local/share/qemu','/usr/share/qemu','/usr/share/OVMF']:
 for code,variables in [('edk2-x86_64-code.fd','edk2-i386-vars.fd'),('OVMF_CODE.fd','OVMF_VARS.fd')]:
  if (Path(root)/code).is_file() and (Path(root)/variables).is_file():firmware=(Path(root)/code,Path(root)/variables);break
 if firmware:break
if not firmware:raise SystemExit('No OVMF firmware found')
shutil.copy2(firmware[1],out/'vars.fd')
args=['qemu-system-x86_64','-machine','q35,i8042=off','-accel','tcg,thread=multi','-cpu','IvyBridge,+x2apic','-smp',os.environ.get('SMP','4'),'-m','2G','-device','intel-iommu,intremap=off',
 '-drive',f'if=pflash,format=raw,readonly=on,file={firmware[0]}','-drive',f'if=pflash,format=raw,file={out}/vars.fd','-drive',f'format=raw,file=fat:rw:{out}/esp',
 '-device','isa-debug-exit,iobase=0xf4,iosize=4','-serial',f'file:{out}/qemu.log','-display','none','-no-reboot','-netdev',f'user,id=net0,hostfwd=tcp:127.0.0.1:{port}-:22','-device','e1000,netdev=net0']
known=out/'known_hosts';known.write_text(f'[127.0.0.1]:{port} '+(out/'test-hostkey.pub').read_text())
ssh=['ssh','-F','/dev/null','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','IdentityAgent=none','-o','AddKeysToAgent=no','-o','StrictHostKeyChecking=yes','-o',f'UserKnownHostsFile={known}','-o','ConnectTimeout=3','-i',str(out/'test-client'),'-p',str(port),'root@127.0.0.1']
def run(command, data=None, expected=0):
 p=subprocess.run(ssh+[command],input=data,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=45)
 if p.returncode!=expected:raise AssertionError((command,p.returncode,p.stdout,p.stderr))
 return p.stdout
(out/'qemu.log').write_text('')
with open(out/'qemu-stderr.log','wb') as err:
 qemu=subprocess.Popen(args,stdout=err,stderr=err)
 try:
  deadline=time.monotonic()+30
  while True:
   if qemu.poll() is not None:raise AssertionError(('QEMU exited during startup',(out/'qemu-stderr.log').read_text()))
   log=(out/'qemu.log').read_text(errors='replace') if (out/'qemu.log').exists() else ''
   if 'Not backgrounding' in log:break
   if time.monotonic()>deadline:raise TimeoutError('SSH startup; inspect target/qemu.log')
   time.sleep(.1)
  assert run("printf ssh-ok; uname -a").startswith(b'ssh-okLinux dross')
  print('ok OpenSSH public-key authentication and remote execution',flush=True)
  run('exit 37',expected=37);run('exit 143',expected=143)
  assert run('sleep 1 & p=$!; wait $p; echo background-ok').strip()==b'background-ok'
  assert run('printf out; printf err >&2; exit 0')==b'out'
  print('ok exit status and separate stdout/stderr',flush=True)
  payload=bytes(range(256))*1024
  result=run('cat > /tmp/upload; sha256sum /tmp/upload',payload)
  assert result.startswith(hashlib.sha256(payload).hexdigest().encode()),result
  assert run('cat /tmp/upload')==payload
  print('ok 256 KiB upload/download and hash',flush=True)
  big=os.urandom(8<<20);t0=time.time()
  assert run('cat > /tmp/big; sha256sum /tmp/big',big).startswith(hashlib.sha256(big).hexdigest().encode())
  t1=time.time();assert run('cat /tmp/big; rm /tmp/big')==big;t2=time.time()
  print(f'ok 8 MiB upload {8/(t1-t0):.2f} MiB/s, download {8/(t2-t1):.2f} MiB/s',flush=True)
  run('cat > /tmp/too-big', os.urandom(10<<20), expected=1)
  assert run('wc -c < /tmp/too-big; rm /tmp/too-big').strip()==b'8388608'
  print('ok file limit returns ENOSPC at 8 MiB and SSH remains usable',flush=True)
  # A real local PTY gives OpenSSH meaningful dimensions and keeps its input open.
  master,slave=pty.openpty();fcntl.ioctl(slave,termios.TIOCSWINSZ,struct.pack('HHHH',31,107,0,0))
  child=subprocess.Popen(ssh[:-1]+['-tt',ssh[-1],'stty size; read x; stty size; printf "TTY:%s\\n" "$x"; exit 7'],stdin=slave,stdout=slave,stderr=slave,close_fds=True)
  os.close(slave); transcript=bytearray();deadline=time.monotonic()+10
  def read_until(needle):
   while needle not in transcript:
    if time.monotonic()>deadline:raise TimeoutError(bytes(transcript))
    if select.select([master],[],[],.2)[0]:
     try:transcript.extend(os.read(master,4096))
     except OSError:raise AssertionError(bytes(transcript))
  read_until(b'31 107')
  fcntl.ioctl(master,termios.TIOCSWINSZ,struct.pack('HHHH',43,119,0,0));child.send_signal(signal.SIGWINCH)
  time.sleep(.1);os.write(master,b'input-ok\n');read_until(b'TTY:input-ok')
  assert child.wait(timeout=5)==7;assert b'43 119' in transcript,bytes(transcript);os.close(master)
  (out/'ssh-pty.log').write_bytes(transcript);print('ok PTY allocation, input, resize and exit status',flush=True)
  # A shell must receive Ctrl-C through its foreground process group.
  master,slave=pty.openpty();child=subprocess.Popen(ssh[:-1]+['-tt',ssh[-1]],stdin=slave,stdout=slave,stderr=slave,close_fds=True);os.close(slave)
  transcript=bytearray();deadline=time.monotonic()+10;os.write(master,b'printf \"SHELL_%s\\n\" READY\n');read_until(b'SHELL_READY\r\n')
  os.write(master,b'sleep 30\n');time.sleep(.3);os.write(master,b'\x1a');read_until(b'Stopped');os.write(master,b'fg\n');time.sleep(.2);os.write(master,b'\x03');time.sleep(.2);os.write(master,b'printf \"AFTER_%s\\n\" INTERRUPT\n');read_until(b'AFTER_INTERRUPT\r\n')
  os.write(master,b'exit\n')
  finish=time.monotonic()+5
  while child.poll() is None and time.monotonic()<finish:
   if select.select([master],[],[],.1)[0]:
    try:transcript.extend(os.read(master,4096))
    except OSError:break
  (out/'ssh-interactive.log').write_bytes(transcript)
  assert child.wait(timeout=1)==0;os.close(master)
  (out/'ssh-interactive.log').write_bytes(transcript);print('ok interactive ash and foreground Ctrl-C',flush=True)
  bad=ssh.copy();bad[bad.index('-i')+1]=str(out/'test-stranger')
  rejected=subprocess.run(bad+['true'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=10);assert rejected.returncode==255
  print('ok unauthorized key rejected',flush=True)
  for i in range(8):assert run(f'echo reconnect-{i}').strip()==f'reconnect-{i}'.encode()
  print('ok reconnect cleanup',flush=True)
  # The guest may see the marker and exit before SSH delivers the command status.
  done=subprocess.run(ssh+['touch /tmp/ssh-done'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=20)
  assert done.returncode in (0,255),(done.returncode,done.stdout,done.stderr)
  assert qemu.wait(timeout=15)==33
  assert 'unix: ALL PASSED' in (out/'qemu.log').read_text();print('ALL UNIX AND SSH TESTS PASSED',flush=True)
 finally:
  if qemu.poll() is None:qemu.terminate()
  try:qemu.wait(timeout=5)
  except subprocess.TimeoutExpired:qemu.kill();qemu.wait()
