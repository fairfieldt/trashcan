#!/usr/bin/env python3
"""Pinned upstream tools plus an isolated, opt-in dross Unix image."""
from pathlib import Path
import subprocess, os, hashlib, urllib.request, tarfile, shutil, re, json
here=Path(__file__).resolve().parent;dross=here.parent;out=here/'target';cache=here/'vendor-cache';sources=out/'src'
out.mkdir(exist_ok=True);cache.mkdir(exist_ok=True);sources.mkdir(exist_ok=True)
for tool in ['cargo','zig','make','cc','ssh-keygen']:
 if not shutil.which(tool):raise SystemExit(f'Required tool not found: {tool}')
packages=[('busybox','1.37.0','https://busybox.net/downloads/busybox-1.37.0.tar.bz2','3311dff32e746499f4df0d5df04d7eb396382d7e108bb9250e7b519b837043a4'),
 ('dropbear','2026.94','https://dropbear.nl/mirror/releases/dropbear-2026.94.tar.bz2','e098034a843699200c8c977a991fff73159735bf795d5f72ef672c41a6b1ae81')]
def run(args,cwd=here,env=None,log=None,input=None):
 print('+',' '.join(map(str,args)),flush=True)
 if log:
  with open(out/log,'w') as f:subprocess.run(list(map(str,args)),cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,input=input,text=isinstance(input,str),check=True)
 else:subprocess.run(list(map(str,args)),cwd=cwd,env=env,check=True)
for name,version,url,sha in packages:
 archive=cache/f'{name}-{version}.tar.bz2'
 if not archive.exists():
  temporary=archive.with_suffix('.download');urllib.request.urlretrieve(url,temporary);temporary.replace(archive)
 if hashlib.sha256(archive.read_bytes()).hexdigest()!=sha:raise SystemExit(f'Checksum mismatch: {archive}')
 source=sources/f'{name}-{version}'
 if not source.exists():
  with tarfile.open(archive) as tar:tar.extractall(sources,filter='data')
source=sources/'busybox-1.37.0'
(source/'allno.config').unlink(missing_ok=True)
run(['make','allnoconfig','HOSTCC=cc'],source,log='busybox-config-base.log')
config=(source/'.config').read_text()
for line in (here/'busybox.config').read_text().splitlines():
 if '=' not in line:continue
 key,value=line.split('=',1);pattern=re.compile(r'^(?:'+key+r'=.*|# '+key+r' is not set)$',re.M)
 if pattern.search(config):config=pattern.sub(line,config)
for key in ['CONFIG_SH_IS_NONE','CONFIG_BASH_IS_ASH','CONFIG_BASH_IS_HUSH']:
 config=re.sub(r'^'+key+r'=y$','# '+key+' is not set',config,flags=re.M)
(source/'.config').write_text(config)
run(['make','oldconfig','HOSTCC=cc'],source,log='busybox-config.log',input='\n'*10000)
run(['make',f'-j{min(os.cpu_count() or 2,8)}','HOSTCC=cc',f'CC={here}/cc','ARCH=x86_64','AR=zig ar','SKIP_STRIP=y'],source,log='busybox-build.log')
run(['make','busybox.links','HOSTCC=cc',f'CC={here}/cc','ARCH=x86_64'],source,log='busybox-links.log')
source=sources/'dropbear-2026.94';shutil.copy2(here/'dropbear-localoptions.h',source/'localoptions.h')
env=os.environ.copy();env.update(CC=str(here/'cc'),AR='zig ar',RANLIB='zig ar s')
run(['./configure','--host=x86_64-linux-musl','--disable-zlib','--disable-syslog','--disable-lastlog','--disable-utmp','--disable-utmpx','--disable-wtmp','--disable-wtmpx','--enable-static'],source,env,log='dropbear-config.log')
run(['make',f'-j{min(os.cpu_count() or 2,8)}','PROGRAMS=dropbear dropbearkey dropbearconvert dbclient','AR=zig ar','RANLIB=zig ar s'],source,env,log='dropbear-build.log')
env=os.environ.copy();env['CARGO_TARGET_DIR']=str(out/'kernel')
run(['cargo','build','--locked','-p','dross','-p','pcid','--features','dross/unix','--target','x86_64-unknown-none','--release'],dross,env)
run(['cargo','build','--locked','-p','dross-boot','--target','x86_64-unknown-uefi','--release'],dross,env)
env['CARGO_TARGET_DIR']=str(out/'native')
run(['cargo','build','--locked','--manifest-path',here/'Cargo.toml','-p','unix-init','-p','unix-netd','-p','unix-nic','--target','x86_64-unknown-none','--release'],dross,env)
env['CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_LINKER']='rust-lld';env['CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_RUSTFLAGS']='-C link-self-contained=yes'
run(['cargo','build','--locked','--manifest-path',here/'Cargo.toml','-p','unix-terminal-bridge','--target','x86_64-unknown-linux-musl','--release'],dross,env)
run([here/'cc','-O2','-static',here/'probe/main.c','-o',out/'unix-probe'])
for name in ['test-hostkey','test-client','test-stranger']:
 if not (out/name).exists():run(['ssh-keygen','-q','-t','ed25519','-N','','-C','oscar-test','-f',out/name])
manifest={'packages':[{'name':name,'version':version,'url':url,'sha256':sha}for name,version,url,sha in packages]}
(out/'build-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
run(['python3',here/'stage.py'])
