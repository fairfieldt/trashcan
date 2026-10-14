#!/usr/bin/env python3
"""Single UDP packets there and back over a Thunderbolt IP link (thunderbolt_net), one at a time and nothing else on
the link: if a receiving controller held a lone frame until more arrived, as it does on the stream, a round trip would
take until the next packet (here: never, until the timeout).

usage:
  udpping.py server ADDR PORT SECONDS     echo every datagram back, for SECONDS
  udpping.py client ADDR PORT N SIZE      N round trips of SIZE-byte datagrams to ADDR:PORT, each after the last reply
                                          (1 s timeout); prints the round-trip times and how many timed out
"""
import json, os, socket, struct, sys, time


def main():
    role, addr, port = sys.argv[1], sys.argv[2], int(sys.argv[3])
    if role == 'server':
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind((addr, port))
        s.settimeout(0.5)
        end, n = time.time() + float(sys.argv[4]), 0
        while time.time() < end:
            try:
                b, peer = s.recvfrom(65536)
            except socket.timeout:
                continue
            s.sendto(b, peer)
            n += 1
        print(json.dumps({'role': 'server', 'addr': addr, 'echoed': n, 'host': os.uname().nodename}), flush=True)
        return
    n, size = int(sys.argv[4]), int(sys.argv[5])
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect((addr, port))
    s.settimeout(1.0)
    rtts, lost = [], 0
    for i in range(n):
        msg = struct.pack('<I', i) + bytes(size - 4)
        t0 = time.perf_counter()
        s.send(msg)
        try:
            while True:
                b = s.recv(65536)
                if struct.unpack('<I', b[:4])[0] == i:
                    break
            rtts.append((time.perf_counter() - t0) * 1e6)
        except socket.timeout:
            lost += 1
    rtts.sort()
    pct = lambda p: round(rtts[min(len(rtts) - 1, int(len(rtts) * p))], 1) if rtts else None
    print(json.dumps({'role': 'client', 'addr': addr, 'n': n, 'size': size, 'lost': lost,
                      'rtt_us': {'p50': pct(0.5), 'p90': pct(0.9), 'p99': pct(0.99), 'max': pct(1.0)},
                      'host': os.uname().nodename}), flush=True)


if __name__ == '__main__':
    main()
