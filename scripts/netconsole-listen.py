#!/usr/bin/env python3
"""Receive incite's netconsole (UDP 6666) and append each line, timestamped, to a log.

usage: scripts/netconsole-listen.py [LOG]   (default ~/dev/claude-work/trashcan-kbuild/netconsole-incite.log)
"""
import datetime, hashlib, json, logging.handlers, os, pathlib, re, socket, subprocess, sys, threading

log = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/dev/claude-work/trashcan-kbuild/netconsole-incite.log")
os.makedirs(os.path.dirname(log), exist_ok=True)
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("0.0.0.0", 6666))
handler = logging.handlers.RotatingFileHandler(log, maxBytes=64 * 1024**2, backupCount=5)
dump_root = pathlib.Path(log).parent / 'crash-dumps'
dump_root.mkdir(exist_ok=True, mode=0o700)
seen = set()

def fetch_dump(record):
    # Accept only these machines and the recorder's fixed directory/file layout.
    host, path = record.get('host'), record.get('path', '')
    if host not in ('incite', 'invite') or not re.fullmatch(
            r'/var/log/trashcan-crash/[0-9a-f-]{36}/devcd[0-9]+-[0-9]+\.dump', path):
        return
    key = (host, path)
    if key in seen:
        return
    seen.add(key)
    def copy():
        destination = dump_root / (host + '-' + pathlib.Path(path).parent.name + '-' + pathlib.Path(path).name)
        partial = destination.with_suffix('.partial')
        try:
            with partial.open('wb') as output:
                os.chmod(partial, 0o600)
                # launchd's background SSH cannot reach the LAN on this workstation;
                # use each node's established tailnet address for dump transfer.
                tailnet = {'incite': '100.108.0.1', 'invite': '100.97.51.99'}[host]
                trusted_alias = {'incite': '192.168.0.83', 'invite': '192.168.0.100'}[host]
                subprocess.run(['ssh', '-o', 'HostName=' + tailnet,
                                '-o', 'HostKeyAlias=' + trusted_alias,
                                '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=3',
                                host, 'sudo', '-n', 'cat', path], stdout=output,
                               stderr=subprocess.PIPE, check=True, timeout=20)
                output.flush()
                os.fsync(output.fileno())
            digest = hashlib.sha256()
            with partial.open('rb') as source:
                while chunk := source.read(64 * 1024):
                    digest.update(chunk)
            if partial.stat().st_size != record.get('bytes') or digest.hexdigest() != record.get('sha256'):
                raise ValueError('dump size/hash mismatch')
            partial.rename(destination)
            destination.with_suffix('.json').write_text(json.dumps(record, indent=2) + '\n')
            directory_fd = os.open(dump_root, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            print(f'dump capture failed for {host} {path}: {error}', file=sys.stderr, flush=True)
            if isinstance(error, subprocess.CalledProcessError):
                print(error.stderr.decode(errors='replace')[-2000:], file=sys.stderr, flush=True)
            seen.discard(key)  # A repeated announcement may retry after a transient failure.
    threading.Thread(target=copy, daemon=True).start()

while True:
    data, addr = s.recvfrom(65535)
    now = datetime.datetime.now().isoformat(timespec="milliseconds")
    for line in data.decode(errors="replace").splitlines():
        handler.emit(logging.LogRecord('netconsole', 20, '', 0, f"{now} {addr[0]} {line}", (), None))
        if line.startswith('CRASHCAPTURE '):
            try:
                record = json.loads(line.removeprefix('CRASHCAPTURE '))
                if record.get('event') == 'dump_ready':
                    fetch_dump(record)
            except (ValueError, AttributeError):
                pass
    handler.flush()
    os.fsync(handler.stream.fileno())
