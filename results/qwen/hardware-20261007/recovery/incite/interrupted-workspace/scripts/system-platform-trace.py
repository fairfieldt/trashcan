#!/usr/bin/env python3
"""Record PCIe hotplug/power control and software poweroff calls without changing them."""
import argparse
import datetime
import json
import logging.handlers
import os
from pathlib import Path
import socket

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--target',required=True)
args=parser.parse_args()
base=Path('/sys/kernel/tracing')
instance=base/'instances/trashcan_platform'
instance.mkdir(exist_ok=True)
instance.joinpath('current_tracer').write_text('nop')
instance.joinpath('trace_clock').write_text('boot')
instance.joinpath('buffer_size_kb').write_text('64')

probes={
    'slot_power_off':'pciehp_power_off_slot ctrl=$arg1:x64',
    'slot_power_on':'pciehp_power_on_slot ctrl=$arg1:x64',
    'slot_disable':'pciehp_disable_slot ctrl=$arg1:x64 safe=$arg2:u8',
    'slot_indicators':'pciehp_set_indicators ctrl=$arg1:x64 power=$arg2:x32 attention=$arg3:x32',
    'slot_command':'pcie_do_write_cmd ctrl=$arg1:x64 command=$arg2:x16 mask=$arg3:x16',
    'slot_read':'pciehp_get_power_status ctrl=$arg1:x64',
    'kernel_poweroff':'kernel_power_off',
    'machine_poweroff':'machine_power_off',
    'acpi_poweroff':'acpi_power_off',
    'gpu_config_reset':'amdgpu_device_pci_config_reset device=$arg1:x64',
}
active=[]
for name,definition in probes.items():
    event=instance/'events/trashcan_platform'/name
    if not event.exists():
        try:
            # Python's append-mode FileIO seeks to EOF; tracefs requires command
            # writes at offset zero. Open the descriptor without that seek.
            fd=os.open(base/'kprobe_events',os.O_WRONLY|os.O_APPEND)
            try:os.write(fd,('p:trashcan_platform/'+name+' '+definition+'\n').encode())
            finally:os.close(fd)
        except OSError as error:
            print('probe unavailable:',name,error,flush=True)
            continue
    event.joinpath('enable').write_text('1');active.append(name)
    if name in ['slot_power_off','slot_power_on','slot_disable','kernel_poweroff','machine_poweroff','gpu_config_reset']:
        trigger=event/'trigger'
        if not any(line.startswith('stacktrace') for line in trigger.read_text().splitlines()):trigger.write_text('stacktrace')
for name in ['pci/pci_hp_event','pci/pcie_link_event','ras/aer_event','ras/mc_event','ras/memory_failure_event']:
    event=instance/'events'/name/'enable'
    if event.exists():event.write_text('1');active.append(name)
instance.joinpath('tracing_on').write_text('1')

root=Path('/var/log/trashcan-platform');root.mkdir(mode=0o700,exist_ok=True)
handler=logging.handlers.RotatingFileHandler(root/'events.jsonl',maxBytes=8*1024**2,backupCount=3)
sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
host=socket.gethostname();boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()

def emit(event,**values):
    record=dict(event=event,host=host,boot=boot,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**values)
    line=json.dumps(record,separators=(',',':'))
    handler.emit(logging.LogRecord('platform',20,'',0,line,(),None))
    handler.flush();os.fsync(handler.stream.fileno())
    try:sock.sendto(('CRASHCAPTURE '+line).encode(),(args.target.split('/')[0],6666))
    except OSError:pass

emit('platform_trace_start',probes=active)
with instance.joinpath('trace_pipe').open() as source:
    for line in source:
        emit('platform_trace',trace=line.rstrip())
