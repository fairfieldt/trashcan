#!/usr/bin/env python3
"""Read-only Mac Pro telemetry and automatic preservation of kernel device dumps.

Run as root through systemd. UDP records go to the same external receiver as
netconsole; local files are flushed every sample. No GPU commands or resets.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import logging.handlers
import os
from pathlib import Path
import socket
import threading
import time


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


class Recorder:
    def __init__(self, root, target):
        self.host = socket.gethostname()
        self.boot = read('/proc/sys/kernel/random/boot_id')
        self.directory = Path(root) / self.boot
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.handler = logging.handlers.RotatingFileHandler(
            Path(root) / 'telemetry.jsonl', maxBytes=16 * 1024**2, backupCount=3)
        self.lock = threading.Lock()
        self.target = (target.split('/')[0], 6666)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.smc = Path('/sys/devices/platform/applesmc.768')
        self.key_indices = {}
        self.smc_lock = open('/run/lock/trashcan-smc-capture.lock', 'w')
        self.discover_keys()

    def emit(self, event, **values):
        record = dict(event=event, host=self.host, boot=self.boot,
                      utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      uptime=read('/proc/uptime'), **values)
        line = json.dumps(record, separators=(',', ':'))
        with self.lock:
            self.handler.emit(logging.LogRecord('capture', 20, '', 0, line, (), None))
            self.handler.flush()
            os.fsync(self.handler.stream.fileno())
            try:
                self.sock.sendto(('CRASHCAPTURE ' + line).encode(), self.target)
            except OSError:
                pass  # Local capture continues when the receiver is unavailable.

    def discover_keys(self):
        selector = self.smc / 'key_at_index'
        previous = read(selector)
        if previous is None:
            return
        wanted = {'PCPC', 'PG0R', 'PG1R', 'PSTR', 'MSSD', 'MSRC', 'MSAL',
                  'MSSP', 'MSSR', 'MSPS', 'MSF0', 'MSF1', 'MSF2', 'MSEE',
                  'TG0D', 'TG1D', 'TC0P', 'Tp0F', 'TPCD', 'TA0P'}
        fcntl.flock(self.smc_lock, fcntl.LOCK_EX)
        try:
            for index in range(int(read(self.smc / 'key_count') or '0')):
                selector.write_text(str(index))
                name = read(self.smc / 'key_at_index_name')
                if name in wanted:
                    self.key_indices[name] = index
        finally:
            selector.write_text(previous)
            fcntl.flock(self.smc_lock, fcntl.LOCK_UN)

    def fan_file(self, name):
        # applesmc keeps its fan files in the device directory before Linux 7.3 and in hwmon/hwmonN from 7.3.
        if (self.smc / name).exists():
            return self.smc / name
        return next((self.smc / 'hwmon').glob('hwmon*/' + name), self.smc / name)

    def smc_values(self):
        result = {}
        selector = self.smc / 'key_at_index'
        fcntl.flock(self.smc_lock, fcntl.LOCK_EX)
        previous = read(selector)
        try:
            for name, index in self.key_indices.items():
                try:
                    selector.write_text(str(index))
                    kind = read(self.smc / 'key_at_index_type')
                    data = (self.smc / 'key_at_index_data').read_bytes()
                    value = dict(type=kind, hex=data.hex())
                    if kind == 'spa5' and len(data) == 2:
                        value['watts'] = int.from_bytes(data, 'big', signed=True) / 32
                    elif kind == 'sp78' and len(data) == 2:
                        value['celsius'] = int.from_bytes(data, 'big', signed=True) / 256
                    elif kind == 'si8' and len(data) == 1:
                        value['signed'] = int.from_bytes(data, 'big', signed=True)
                    result[name] = value
                except OSError as error:
                    result[name] = dict(error=str(error))
        finally:
            if previous is not None:
                selector.write_text(previous)
            fcntl.flock(self.smc_lock, fcntl.LOCK_UN)
        return result

    def sample(self):
        started = time.monotonic()
        gpu = {}
        for address in ('0000:02:00.0', '0000:06:00.0'):
            device = Path('/sys/bus/pci/devices') / address
            values = {}
            for name in ('power/runtime_status', 'current_link_speed', 'current_link_width'):
                values[name] = read(device / name)
            for hwmon in (device / 'hwmon').glob('hwmon*'):
                for sensor in ('temp1_input', 'fan1_input', 'power1_average'):
                    values[sensor] = read(hwmon / sensor)
            gpu[address] = values
        aer = {}
        for port in ('0000:00:02.0', '0000:00:03.0'):
            counters = {}
            for path in (Path('/sys/bus/pci/devices') / port).glob('aer_*'):
                if not path.is_file():
                    continue
                value = read(path)
                if value is None:
                    counters[path.name] = None
                elif path.name.startswith('aer_rootport_total'):
                    counters[path.name] = value
                else:
                    # Preserve totals and nonzero errors without repeating the
                    # full table of zero counters in every UDP packet.
                    counters[path.name] = {name: int(count) for name, count in
                        (line.split() for line in value.splitlines())
                        if int(count) != 0 or name.startswith('TOTAL_')}
            aer[port] = counters
        smc = self.smc_values()
        self.emit('telemetry', gpu=gpu, aer=aer, smc=smc,
                  fan_rpm=read(self.fan_file('fan1_input')),
                  load=read('/proc/loadavg'), sample_ms=round((time.monotonic()-started)*1000, 2))

    def dumps(self, source_root='/sys/class/devcoredump'):
        # The installed kernel's devcoredump reader copies a preformatted buffer;
        # it does not issue GPU commands. Keep this separate from the sampler.
        seen = set()
        while True:
            for data in Path(source_root).glob('devcd*/data'):
                try:
                    identity = (str(data), data.parent.stat().st_ino)
                except OSError:
                    continue  # The kernel can expire a dump between glob and stat.
                if identity in seen:
                    continue
                try:
                    source_device = str((data.parent / 'failing_device').resolve())
                    destination = self.directory / (data.parent.name + '-' + str(time.time_ns()) + '.dump')
                    self.emit('dump_started', source=str(data), device=source_device)
                    digest = hashlib.sha256()
                    size = 0
                    with data.open('rb', buffering=0) as source, destination.open('xb') as output:
                        os.chmod(destination, 0o600)
                        while chunk := source.read(64 * 1024):
                            if size + len(chunk) > 256 * 1024**2:
                                raise ValueError('device dump exceeded 256 MiB limit')
                            output.write(chunk)
                            digest.update(chunk)
                            size += len(chunk)
                        output.flush()
                        os.fsync(output.fileno())
                    directory_fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
                    seen.add(identity)
                    # Repeat announcements so a lost UDP packet need not lose the dump.
                    for _ in range(3):
                        self.emit('dump_ready', path=str(destination), bytes=size,
                                  sha256=digest.hexdigest(), device=source_device)
                        time.sleep(0.2)
                except (OSError, ValueError) as error:
                    self.emit('dump_error', source=str(data), error=str(error))
                    seen.add(identity)
            time.sleep(0.2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', required=True, help='external netconsole receiver IPv4 address')
    parser.add_argument('--root', default='/var/log/trashcan-crash')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    recorder = Recorder(args.root, args.target)
    recorder.emit('start', kernel=read('/proc/sys/kernel/osrelease'),
                  cmdline=read('/proc/cmdline'), keys=list(recorder.key_indices))
    if args.once:
        recorder.sample()
        return
    threading.Thread(target=recorder.dumps, daemon=True).start()
    while True:
        started = time.monotonic()
        try:
            recorder.sample()
        except OSError as error:
            recorder.emit('sample_error', error=str(error))
        time.sleep(max(0, 1 - (time.monotonic() - started)))


if __name__ == '__main__':
    main()
