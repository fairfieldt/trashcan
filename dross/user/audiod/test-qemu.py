#!/usr/bin/env python3
"""Isolated HDA playback test. Writes only target/audio-qemu (or --out)."""
import argparse, json, math, shutil, struct, subprocess, wave
from pathlib import Path
HERE=Path(__file__).resolve().parent
DROSS=HERE.parent.parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out',type=Path,default=DROSS/'target/audio-qemu')
parser.add_argument('--esp',type=Path,default=DROSS/'target/hardware-memory-final-esp')
parser.add_argument('--smp',type=int,default=4)
args=parser.parse_args()
out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
def run(command,**kwargs): subprocess.run([str(x) for x in command],cwd=DROSS,check=True,**kwargs)
run(['cargo','test','-p','audiod','--lib','--target-dir',out/'cargo'])
run(['cargo','build','-p','audiod','--features','native','--bins','--release','--target','x86_64-unknown-none','--target-dir',out/'cargo'])
# Package the exact self-contained EFI artifact that can later be selected for
# a coordinated physical acceptance boot. The tested kernel stays unchanged.
import tarfile
(out/'init.conf').write_text('# Dedicated native audio acceptance boot.\n')
with tarfile.open(out/'rootfs.tar','w',format=tarfile.USTAR_FORMAT):pass
native=out/'cargo/x86_64-unknown-none/release'
run(['python3',DROSS/'stage-image.py','--out',out,'--esp',args.esp.resolve(),
     '--init-conf',out/'init.conf','--rootfs',out/'rootfs.tar',
     '--module',f'init={native}/audio-test-init','--module',f'audiod={native}/audiod',
     '--module',f'audiotest={native}/audiotest'])
esp=out/'esp'
if esp.exists():shutil.rmtree(esp)
(esp/'EFI/BOOT').mkdir(parents=True)
shutil.copy2(out/'BOOTX64.EFI',esp/'EFI/BOOT/BOOTX64.EFI')
firmware=None
for directory in ['/opt/homebrew/share/qemu','/usr/share/qemu','/usr/share/OVMF']:
    directory=Path(directory)
    for code,variables in [('edk2-x86_64-code.fd','edk2-i386-vars.fd'),('OVMF_CODE.fd','OVMF_VARS.fd')]:
        if (directory/code).exists():firmware=directory/code;shutil.copy2(directory/variables,out/'vars.fd');break
    if firmware:break
if not firmware:raise SystemExit('OVMF not found')
cmd=['qemu-system-x86_64','-machine','q35','-accel','tcg,thread=multi','-cpu','IvyBridge,+x2apic','-smp',str(args.smp),'-m','2G','-device','intel-iommu,intremap=off','-drive',f'if=pflash,format=raw,readonly=on,file={firmware}','-drive',f'if=pflash,format=raw,file={out}/vars.fd','-drive',f'format=raw,file=fat:rw:{esp}','-device','isa-debug-exit,iobase=0xf4,iosize=4','-serial','stdio','-display','none','-no-reboot','-net','none','-audiodev',f'wav,id=audio,path={out}/capture.wav,out.frequency=48000,out.channels=2,out.format=s16','-device','intel-hda,id=hda','-device','hda-output,bus=hda.0,audiodev=audio']
(out/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
with (out/'qemu.log').open('w') as log:
    process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
    try:status=process.wait(timeout=120)
    except subprocess.TimeoutExpired:process.kill();process.wait();raise SystemExit(f'QEMU timed out: {out}/qemu.log')
text=(out/'qemu.log').read_text(errors='replace')
print(text[text.find('init started'):])
if status!=33 or 'audio test: ok' not in text or 'DMA unmapped' not in text:raise SystemExit(f'audio boot failed({status}); see {out}/qemu.log')
# isa-debug-exit bypasses QEMU audio cleanup, leaving WAV lengths zero.
# Finalize only those header lengths; retain the captured PCM bytes verbatim.
path=out/'capture.wav';raw=bytearray(path.read_bytes())
if raw[:4]!=b'RIFF' or raw[8:12]!=b'WAVE' or raw[36:40]!=b'data':raise SystemExit('unexpected WAV capture header')
struct.pack_into('<I',raw,4,len(raw)-8);struct.pack_into('<I',raw,40,len(raw)-44);path.write_bytes(raw)
with wave.open(str(path),'rb') as wav:
    assert wav.getframerate()==48000 and wav.getnchannels()==2 and wav.getsampwidth()==2
    frames=wav.getnframes();values=struct.unpack('<'+'h'*(frames*2),wav.readframes(frames))
# Require both submitted blocks byte-for-byte in order, including every frame.
sine=[0,293,574,833,1061,1247,1386,1471,1500,1471,1386,1247,1061,833,574,293,0,-293,-574,-833,-1061,-1247,-1386,-1471,-1500,-1471,-1386,-1247,-1061,-833,-574,-293]
search=0;blocks=[]
for count in [12000,24000]:
    left=right=0;expected=[]
    for frame in range(count):
        expected.extend([sine[left>>27],sine[right>>27]])
        left=(left+((440<<32)//48000))&0xffffffff;right=(right+((660<<32)//48000))&0xffffffff
    pcm=struct.pack('<'+'h'*len(expected),*expected)
    location=raw.find(pcm,44+search)
    if location<0 or (location-44)%4:raise SystemExit(f'missing or corrupted {count}-frame PCM block')
    blocks.append({'frames':count,'capture_frame':(location-44)//4,'exact_pcm':True})
    search=location-44+len(pcm)
channels=[values[0::2],values[1::2]]
metrics={'sample_rate':48000,'frames':frames,'exact_blocks':blocks,'channels':[]}
for signal,frequency in zip(channels,[440,660]):
    active=[i for i,x in enumerate(signal) if abs(x)>100]
    if len(active)<24000:raise SystemExit('too few audible samples')
    start=active[0]+2400;segment=signal[start:start+7200]
    rms=math.sqrt(sum(x*x for x in segment)/len(segment))
    power=abs(sum(x*complex(math.cos(2*math.pi*frequency*i/48000),-math.sin(2*math.pi*frequency*i/48000)) for i,x in enumerate(segment)))**2
    other=1100 if frequency==440 else 990
    noise=abs(sum(x*complex(math.cos(2*math.pi*other*i/48000),-math.sin(2*math.pi*other*i/48000)) for i,x in enumerate(segment)))**2
    if not 500<rms<1500 or power<noise*100:raise SystemExit('wrong level or tone in captured PCM')
    metrics['channels'].append({'expected_hz':frequency,'rms':round(rms,2),'tone_to_other_db':round(10*math.log10(power/max(noise,1)),2),'active_samples':len(active)})
(out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
print(json.dumps(metrics,indent=2))
