# Path map: pathmap

## Path ends with the stream devices open

Tx: the hop in the NHI adapter's table (in HopID = Tx ring); Rx: the hop in the lane adapter's table (next HopID = Rx ring). cr = initial credits, fc = ingress/egress flow control.

| Mac | kernel | link | peer | stream Tx ring>hop | stream Tx hop | stream Rx hop>ring | stream Rx hop | tbnet Tx ring>hop | tbnet Tx hop | tbnet Rx hop>ring | tbnet Rx hop | lane adapter | NHI adapter |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| incite | 7.3.0-rc7-p2p | 0-3 port 3 | inflight | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2>8 | cr 14 fc 1/1 drop 0 p5/w1 | 8>2 | cr 14 fc 1/0 drop 0 p5/w1 | port 3: nfc 0 total 60 width 2 speed 0x8; port 4: nfc 0 total 60 width 2 speed 0x8 | port 5: nfc 0 total 15 |
| incite | 7.3.0-rc7-p2p | 1-3 port 3 | enlight | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2>8 | cr 14 fc 1/1 drop 0 p5/w1 | 8>2 | cr 14 fc 1/0 drop 0 p5/w1 | port 3: nfc 0 total 60 width 2 speed 0x8; port 4: nfc 0 total 60 width 2 speed 0x8 | port 5: nfc 0 total 15 |
| inflight | 7.3.0-rc7-p2p | 0-3 port 3 | incite | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2>8 | cr 14 fc 1/1 drop 0 p5/w1 | 8>2 | cr 14 fc 1/0 drop 0 p5/w1 | port 3: nfc 0 total 60 width 2 speed 0x8; port 4: nfc 0 total 60 width 2 speed 0x8 | port 5: nfc 0 total 15 |
| inflight | 7.3.0-rc7-p2p | 1-3 port 3 | alight | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2>8 | cr 14 fc 1/1 drop 0 p5/w1 | 8>2 | cr 14 fc 1/0 drop 0 p5/w1 | port 3: nfc 0 total 60 width 2 speed 0x8; port 4: nfc 0 total 60 width 2 speed 0x8 | port 5: nfc 0 total 15 |
| alight | 7.3.0-rc7-p2p | 0-3 port 3 | indite | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2>8 | cr 14 fc 1/1 drop 0 p5/w1 | 8>2 | cr 14 fc 1/0 drop 0 p5/w1 | port 3: nfc 0 total 60 width 2 speed 0x8; port 4: nfc 0 total 60 width 2 speed 0x8 | port 5: nfc 0 total 15 |
| alight | 7.3.0-rc7-p2p | 1-3 port 3 | inflight | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2>8 | cr 14 fc 1/1 drop 0 p5/w1 | 8>2 | cr 14 fc 1/0 drop 0 p5/w1 | port 3: nfc 0 total 60 width 2 speed 0x8; port 4: nfc 0 total 60 width 2 speed 0x8 | port 5: nfc 0 total 15 |
| indite | 7.3.0-rc7-p2p | 0-3 port 3 | alight | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2>8 | cr 14 fc 1/1 drop 0 p5/w1 | 8>2 | cr 14 fc 1/0 drop 0 p5/w1 | port 3: nfc 0 total 60 width 2 speed 0x8; port 4: nfc 0 total 60 width 2 speed 0x8 | port 5: nfc 0 total 15 |
| indite | 7.3.0-rc7-p2p | 1-3 port 3 | invite | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2>8 | cr 14 fc 1/1 drop 0 p5/w1 | 8>2 | cr 14 fc 1/0 drop 0 p5/w1 | port 3: nfc 0 total 60 width 2 speed 0x8; port 4: nfc 0 total 60 width 2 speed 0x8 | port 5: nfc 0 total 15 |
| invite | 7.3.0-rc7-p2p-s3 | 0-3 port 3 | enlight | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2>8 | cr 14 fc 1/1 drop 0 p5/w1 | 8>2 | cr 14 fc 1/0 drop 0 p5/w1 | port 3: nfc 0 total 60 width 2 speed 0x8; port 4: nfc 0 total 60 width 2 speed 0x8 | port 5: nfc 0 total 15 |
| invite | 7.3.0-rc7-p2p-s3 | 1-3 port 3 | indite | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2>8 | cr 14 fc 1/1 drop 0 p5/w1 | 8>2 | cr 14 fc 1/0 drop 0 p5/w1 | port 3: nfc 0 total 60 width 2 speed 0x8; port 4: nfc 0 total 60 width 2 speed 0x8 | port 5: nfc 0 total 15 |
| enlight | 7.3.0-rc7-p2p-s3 | 0-3 port 3 | invite | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2>8 | cr 14 fc 1/1 drop 0 p5/w1 | 8>2 | cr 14 fc 1/0 drop 0 p5/w1 | port 3: nfc 0 total 60 width 2 speed 0x8; port 4: nfc 0 total 60 width 2 speed 0x8 | port 5: nfc 0 total 15 |
| enlight | 7.3.0-rc7-p2p-s3 | 1-3 port 3 | incite | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2>8 | cr 14 fc 1/1 drop 0 p5/w1 | 8>2 | cr 14 fc 1/0 drop 0 p5/w1 | port 3: nfc 0 total 60 width 2 speed 0x8; port 4: nfc 0 total 60 width 2 speed 0x8 | port 5: nfc 0 total 15 |

## Changes between the snapshots

- invite 1-0 port 3 in HopID 15, closed -> open: 0x00000000 0x22000001 -> 0x801c2803 0x017ff501
- invite 1-0 port 5 in HopID 3, closed -> open: 0x00000000 0x22000001 -> 0x801c180f 0x037ff501
- 2 other changes; 46 stream hops only switched on or off with the devices (enable and flow control bits)

## The hold per direction (pos: fillers each lone frame needed into the other end, per open)

| direction | pos per open | pp per open (client = sender) | lost opens | sender Tx ring>hop | sender Tx hop | receiver Rx hop>ring | receiver Rx hop | tbnet rings (sender Tx, receiver Rx) | receiver lane adapter | sender lane adapter | kernels | receiver group |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| incite->enlight | 0 0 0 0 0 0 | 0 0 0 0 0 0 | - | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2, 2 | nfc 0 total 60 width 2 speed 0x8 | nfc 0 total 60 width 2 speed 0x8 | 7.3.0-rc7-p2p > 7.3.0-rc7-p2p-s3 | busy_poll=0 paths_last=0 poll_mode=0 ring_size=256 rx_e2e=1 rx_eof_close=1 rx_repost=0 throttling=8192 |
| enlight->incite | 0 0 0 0 0 0 | 0 0 0 0 0 0 | - | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2, 2 | nfc 0 total 60 width 2 speed 0x8 | nfc 0 total 60 width 2 speed 0x8 | 7.3.0-rc7-p2p-s3 > 7.3.0-rc7-p2p | busy_poll=0 ring_size=256 throttling=8192 |
| incite->inflight | x 2 2 2 2 2 | 2 x x x x x | run 1 no answer at incite (client), timeout at inflight (server); run 15 ENXIO at incite (client), ENXIO at inflight (server); run 17 ENXIO at incite (client), ENXIO at inflight (server); run 19 ENXIO at incite (client), ENXIO at inflight (server); run 21 ENXIO at incite (client), ENXIO at inflight (server); run 23 ENXIO at incite (client), ENXIO at inflight (server) | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2, 2 | nfc 0 total 60 width 2 speed 0x8 | nfc 0 total 60 width 2 speed 0x8 | 7.3.0-rc7-p2p > 7.3.0-rc7-p2p | busy_poll=0 ring_size=256 throttling=8192 |
| inflight->incite | x x x x x x | 2-10 2-10 2-10 2-10 2-10 2-10 | run 2 ENXIO at inflight (client), stopped at incite (server); run 4 ENXIO at inflight (client), stopped at incite (server); run 6 ENXIO at inflight (client), stopped at incite (server); run 8 ENXIO at inflight (client), stopped at incite (server); run 10 ENXIO at inflight (client), stopped at incite (server); run 12 ENXIO at inflight (client), stopped at incite (server) | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2, 2 | nfc 0 total 60 width 2 speed 0x8 | nfc 0 total 60 width 2 speed 0x8 | 7.3.0-rc7-p2p > 7.3.0-rc7-p2p | busy_poll=0 ring_size=256 throttling=8192 |
| inflight->alight | x 1 1 1 1 1 | 1 x x x x x | run 49 no answer at inflight (client), timeout at alight (server); run 63 ENXIO at inflight (client), ENXIO at alight (server); run 65 ENXIO at inflight (client), ENXIO at alight (server); run 67 ENXIO at inflight (client), ENXIO at alight (server); run 69 ENXIO at inflight (client), ENXIO at alight (server); run 71 ENXIO at inflight (client), ENXIO at alight (server) | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2, 2 | nfc 0 total 60 width 2 speed 0x8 | nfc 0 total 60 width 2 speed 0x8 | 7.3.0-rc7-p2p > 7.3.0-rc7-p2p | busy_poll=0 ring_size=256 throttling=8192 |
| alight->inflight | x x x x x x | 1-10 1-10 1-10 1-10 1-10 1-10 | run 50 ENXIO at alight (client), stopped at inflight (server); run 52 ENXIO at alight (client), stopped at inflight (server); run 54 ENXIO at alight (client), stopped at inflight (server); run 56 ENXIO at alight (client), stopped at inflight (server); run 58 ENXIO at alight (client), stopped at inflight (server); run 60 ENXIO at alight (client), stopped at inflight (server) | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2, 2 | nfc 0 total 60 width 2 speed 0x8 | nfc 0 total 60 width 2 speed 0x8 | 7.3.0-rc7-p2p > 7.3.0-rc7-p2p | busy_poll=0 ring_size=256 throttling=8192 |
| alight->indite | x 2 2 2 2 2 | 2 x x x x x | run 73 no answer at alight (client), timeout at indite (server); run 87 ENXIO at alight (client), ENXIO at indite (server); run 89 ENXIO at alight (client), ENXIO at indite (server); run 91 ENXIO at alight (client), ENXIO at indite (server); run 93 ENXIO at alight (client), ENXIO at indite (server); run 95 ENXIO at alight (client), ENXIO at indite (server) | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2, 2 | nfc 0 total 60 width 2 speed 0x8 | nfc 0 total 60 width 2 speed 0x8 | 7.3.0-rc7-p2p > 7.3.0-rc7-p2p | busy_poll=0 ring_size=256 throttling=8192 |
| indite->alight | x x x x x x | 2-10 2-10 2-10 2-10 2-10 2-10 | run 74 ENXIO at indite (client), stopped at alight (server); run 76 ENXIO at indite (client), stopped at alight (server); run 78 ENXIO at indite (client), stopped at alight (server); run 80 ENXIO at indite (client), stopped at alight (server); run 82 ENXIO at indite (client), stopped at alight (server); run 84 ENXIO at indite (client), stopped at alight (server) | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2, 2 | nfc 0 total 60 width 2 speed 0x8 | nfc 0 total 60 width 2 speed 0x8 | 7.3.0-rc7-p2p > 7.3.0-rc7-p2p | busy_poll=0 ring_size=256 throttling=8192 |
| indite->invite | 0 0 0 0 0 0 | 0 0 0 0 0 0 | - | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2, 2 | nfc 0 total 60 width 2 speed 0x8 | nfc 0 total 60 width 2 speed 0x8 | 7.3.0-rc7-p2p > 7.3.0-rc7-p2p-s3 | busy_poll=0 paths_last=0 poll_mode=0 ring_size=256 rx_e2e=1 rx_eof_close=1 rx_repost=0 throttling=8192 |
| invite->indite | 0 0 0 0 0 0 | 0 0 0 0 0 0 | - | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2, 2 | nfc 0 total 60 width 2 speed 0x8 | nfc 0 total 60 width 2 speed 0x8 | 7.3.0-rc7-p2p-s3 > 7.3.0-rc7-p2p | busy_poll=0 ring_size=256 throttling=8192 |
| invite->enlight | 0 0 0 0 0 0 | 0 0 0 0 0 0 | - | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2, 2 | nfc 0 total 60 width 2 speed 0x8 | nfc 0 total 60 width 2 speed 0x8 | 7.3.0-rc7-p2p-s3 > 7.3.0-rc7-p2p-s3 | busy_poll=0 paths_last=0 poll_mode=0 ring_size=256 rx_e2e=1 rx_eof_close=1 rx_repost=0 throttling=8192 |
| enlight->invite | 0 0 0 0 0 0 | 0 0 0 0 0 0 | - | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2, 2 | nfc 0 total 60 width 2 speed 0x8 | nfc 0 total 60 width 2 speed 0x8 | 7.3.0-rc7-p2p-s3 > 7.3.0-rc7-p2p-s3 | busy_poll=0 paths_last=0 poll_mode=0 ring_size=256 rx_e2e=1 rx_eof_close=1 rx_repost=0 throttling=8192 |

## Settings against the hold

Held (pos, any open with a filler): incite->inflight, inflight->alight, alight->indite. No hold: incite->enlight, enlight->incite, indite->invite, invite->indite, invite->enlight, enlight->invite. No answer: inflight->incite, alight->inflight, indite->alight.

| setting | in the directions that held | in the others | splits them |
|---|---|---|---|
| receiver_ctl | 0-0, 1-0 | 0-0, 1-0 |  |
| receiver_group | busy_poll=0 ring_size=256 throttling=8192 | busy_poll=0 paths_last=0 poll_mode=0 ring_size=256 rx_e2e=1 rx_eof_close=1 rx_repost=0 throttling=8192, busy_poll=0 ring_size=256 throttling=8192 |  |
| receiver_kernel | 7.3.0-rc7-p2p | 7.3.0-rc7-p2p, 7.3.0-rc7-p2p-s3 |  |
| rx_credits | 14 | 14 |  |
| rx_drop | 0 | 0 |  |
| rx_fc | 1/0 | 1/0 |  |
| rx_lane_buffers | nfc 0 total 60 width 2 speed 0x8 | nfc 0 total 60 width 2 speed 0x8 |  |
| rx_nhi_buffers | nfc 0 total 15 | nfc 0 total 15 |  |
| rx_ring | 3 | 3 |  |
| sender_ctl | 0-0, 1-0 | 0-0, 1-0 |  |
| sender_kernel | 7.3.0-rc7-p2p | 7.3.0-rc7-p2p, 7.3.0-rc7-p2p-s3 |  |
| stream_rx_before_tbnet | False | False |  |
| stream_tx_before_tbnet | False | False |  |
| tbnet_rx_ring | 2 | 2 |  |
| tbnet_tx_ring | 2 | 2 |  |
| tx_credits | 14 | 14 |  |
| tx_drop | 0 | 0 |  |
| tx_fc | 1/1 | 1/1 |  |
| tx_lane_buffers | nfc 0 total 60 width 2 speed 0x8 | nfc 0 total 60 width 2 speed 0x8 |  |
| tx_nhi_buffers | nfc 0 total 15 | nfc 0 total 15 |  |
| tx_prio | 5/1 | 5/1 |  |
| tx_ring | 3 | 3 |  |

## Devices held open for the open snapshot

incite: tbstream0 ok, tbstream1 ok; inflight: tbstream0 ok, tbstream1 ok; alight: tbstream0 ok, tbstream1 ok; indite: tbstream0 ok eofs 29, tbstream1 ok; invite: tbstream0 ok, tbstream1 ok; enlight: tbstream0 ok eofs 1, tbstream1 ok eofs 1

## Kernel log (last Thunderbolt lines per Mac, closed snapshot)

incite:
    [ 8460.738144] thunderbolt 1-3: failed to send properties changed notification
    [ 8460.738164] thunderbolt 0-3: failed to send properties changed notification
    [ 8462.786253] thunderbolt 0-3: failed to send properties changed notification
    [ 8462.788175] ACPI: bus type thunderbolt unregistered
    [ 8466.543087] ACPI: bus type thunderbolt registered
    [ 8466.543846] thunderbolt 0000:16:00.0: device link creation from 0000:15:00.0 failed
    [ 8466.549944] thunderbolt 0000:5d:00.0: device link creation from 0000:5c:00.0 failed
    [ 8466.559514] thunderbolt 0000:a4:00.0: device link creation from 0000:a3:00.0 failed
    [ 8471.875166] thunderbolt 1-3: new host found, vendor=0x1d6b device=0x4
    [ 8471.875187] thunderbolt 1-3: Linux enlight
    [ 8471.875305] thunderbolt 0-3: new host found, vendor=0x1d6b device=0x4
    [ 8471.875321] thunderbolt 0-3: Linux inflight
inflight:
    [10297.312079] thunderbolt 1-3: Linux alight
    [10298.336354] thunderbolt 0-3: new host found, vendor=0x1d6b device=0x4
    [10298.336374] thunderbolt 0-3: Linux incite
    [10322.395507] ACPI: bus type thunderbolt unregistered
    [10329.227337] ACPI: bus type thunderbolt registered
    [10329.227858] thunderbolt 0000:16:00.0: device link creation from 0000:15:00.0 failed
    [10329.231562] thunderbolt 0000:5d:00.0: device link creation from 0000:5c:00.0 failed
    [10329.235584] thunderbolt 0000:a4:00.0: device link creation from 0000:a3:00.0 failed
    [10333.537857] thunderbolt 1-3: new host found, vendor=0x1d6b device=0x4
    [10333.537883] thunderbolt 1-3: Linux alight
    [10333.538021] thunderbolt 0-3: new host found, vendor=0x1d6b device=0x4
    [10333.538042] thunderbolt 0-3: Linux incite
alight:
    [ 6217.194650] thunderbolt 1-3: Linux inflight
    [ 6217.198666] thunderbolt 0-3: new host found, vendor=0x1d6b device=0x4
    [ 6217.198685] thunderbolt 0-3: Linux indite
    [ 6240.235302] ACPI: bus type thunderbolt unregistered
    [ 6247.081994] ACPI: bus type thunderbolt registered
    [ 6247.082518] thunderbolt 0000:16:00.0: device link creation from 0000:15:00.0 failed
    [ 6247.088470] thunderbolt 0000:5d:00.0: device link creation from 0000:5c:00.0 failed
    [ 6247.093748] thunderbolt 0000:a4:00.0: device link creation from 0000:a3:00.0 failed
    [ 6252.397177] thunderbolt 0-3: new host found, vendor=0x1d6b device=0x4
    [ 6252.397205] thunderbolt 0-3: Linux indite
    [ 6252.432997] thunderbolt 1-3: new host found, vendor=0x1d6b device=0x4
    [ 6252.433024] thunderbolt 1-3: Linux inflight
indite:
    [ 8960.981181] thunderbolt 0-3: new host found, vendor=0x1d6b device=0x4
    [ 8960.981192] thunderbolt 0-3: Linux alight
    [ 8960.981196] thunderbolt 1-3: Linux invite
    [ 8985.056458] ACPI: bus type thunderbolt unregistered
    [ 8991.872285] ACPI: bus type thunderbolt registered
    [ 8991.872721] thunderbolt 0000:16:00.0: device link creation from 0000:15:00.0 failed
    [ 8991.878081] thunderbolt 0000:5d:00.0: device link creation from 0000:5c:00.0 failed
    [ 8991.881678] thunderbolt 0000:a4:00.0: device link creation from 0000:a3:00.0 failed
    [ 8996.183198] thunderbolt 0-3: new host found, vendor=0x1d6b device=0x4
    [ 8996.183205] thunderbolt 1-3: new host found, vendor=0x1d6b device=0x4
    [ 8996.183221] thunderbolt 0-3: Linux alight
    [ 8996.183231] thunderbolt 1-3: Linux invite
invite:
    [  777.641932] RIP: 0010:ring_interrupt_active+0x27f/0x310 [thunderbolt]
    [  777.642090]  tb_ring_stop+0x167/0x1d0 [thunderbolt]
    [  777.642145]  tbstream_dev_stop+0x131/0x170 [thunderbolt_stream]
    [  777.642158]  tbstream_dev_fops_release+0xbe/0x130 [thunderbolt_stream]
    [  777.978091] thunderbolt 0000:16:00.0: interrupt for RX ring 3 is already disabled
    [  777.978100] WARNING: drivers/thunderbolt/nhi.c:143 at ring_interrupt_active+0x26e/0x310 [thunderbolt], CPU#10: link-close/26736
    [  777.978162] Modules linked in: thunderbolt_stream thunderbolt_net thunderbolt netconsole xt_connmark xt_conntrack xt_MASQUERADE xt_tcpudp xt_mark nft_compat x_tables nft_chain_nat nf_nat nf_conntrack nf_defrag_ipv6 nf_defrag_ipv4 tun nf_tables amdgpu 8021q garp stp mrp llc binfmt_misc b43 nls_ascii nls_cp437 vfat cordic fat mac80211 intel_rapl_msr intel_rapl_common sb_edac x86_pkg_temp_thermal intel_powerclamp coretemp btusb drm_buddy amdxcp btmtk drm_panel_backlight_quirks gpu_sched btrtl btbcm snd_hda_codec_atihdmi snd_hda_codec_cs420x kvm_intel btintel radeon snd_hda_codec_generic snd_hda_codec_hdmi cfg80211 snd_hda_intel drm_ttm_helper joydev kvm bluetooth ttm snd_hda_codec ssb drm_exec drm_suballoc_helper mmc_core snd_hda_core drm_display_helper hid_generic irqbypass usbhid snd_intel_dspcfg aesni_intel snd_intel_sdw_acpi snd_hwdep hid cec gf128mul ecdh_generic pcmcia rapl snd_pcm rc_core rfkill intel_cstate pcmcia_core libarc4 drm_client_lib mei_me snd_timer apple_mfi_fastcharge intel_uncore applesmc evdev pcspkr
    [  777.978277]  drm_kms_helper snd mei apple_gmux i2c_algo_bit bcma soundcore video wmi button sg drm efi_pstore efivarfs autofs4 ext4 crc16 mbcache jbd2 sd_mod ahci libahci libata xhci_pci iTCO_wdt ehci_pci intel_pmc_bxt xhci_hcd ehci_hcd tg3 watchdog at24 scsi_mod usbcore libphy ioatdma i2c_i801 lpc_ich i2c_smbus scsi_common mdio_bus dca configfs usb_common [last unloaded: thunderbolt]
    [  777.978398] RIP: 0010:ring_interrupt_active+0x27f/0x310 [thunderbolt]
    [  777.978514]  tb_ring_stop+0x167/0x1d0 [thunderbolt]
    [  777.978557]  tbstream_dev_stop+0x131/0x170 [thunderbolt_stream]
    [  777.978567]  tbstream_dev_fops_release+0xbe/0x130 [thunderbolt_stream]
enlight:
    [  777.744576] RIP: 0010:ring_interrupt_active+0x27f/0x310 [thunderbolt]
    [  777.744661]  tb_ring_stop+0x167/0x1d0 [thunderbolt]
    [  777.744690]  tbstream_dev_stop+0x131/0x170 [thunderbolt_stream]
    [  777.744698]  tbstream_dev_fops_release+0xbe/0x130 [thunderbolt_stream]
    [  777.963853] thunderbolt 0000:16:00.0: interrupt for RX ring 3 is already disabled
    [  777.963858] WARNING: drivers/thunderbolt/nhi.c:143 at ring_interrupt_active+0x26e/0x310 [thunderbolt], CPU#20: trashcan-stream/26482
    [  777.963898] Modules linked in: thunderbolt_stream thunderbolt_net thunderbolt xt_connmark xt_conntrack xt_MASQUERADE xt_tcpudp xt_mark nft_compat x_tables nft_chain_nat nf_nat nf_conntrack nf_defrag_ipv6 nf_defrag_ipv4 netconsole tun nf_tables amdgpu 8021q garp stp mrp llc b43 cordic binfmt_misc mac80211 intel_rapl_msr intel_rapl_common nls_ascii nls_cp437 sb_edac vfat fat x86_pkg_temp_thermal intel_powerclamp coretemp btusb drm_buddy amdxcp btmtk drm_panel_backlight_quirks gpu_sched btrtl snd_hda_codec_atihdmi snd_hda_codec_cs420x btbcm kvm_intel btintel radeon snd_hda_codec_generic snd_hda_codec_hdmi snd_hda_intel drm_ttm_helper cfg80211 kvm joydev bluetooth snd_hda_codec ttm ssb drm_exec drm_suballoc_helper snd_hda_core mmc_core drm_display_helper hid_generic snd_intel_dspcfg snd_intel_sdw_acpi irqbypass usbhid snd_hwdep aesni_intel pcmcia cec hid snd_pcm gf128mul ecdh_generic rc_core pcmcia_core rapl rfkill intel_cstate libarc4 drm_client_lib snd_timer mei_me applesmc evdev apple_mfi_fastcharge intel_uncore
    [  777.963972]  drm_kms_helper snd pcspkr mei i2c_algo_bit bcma soundcore apple_gmux video wmi button sg drm efi_pstore efivarfs autofs4 ext4 crc16 mbcache jbd2 sd_mod ahci libahci xhci_pci libata xhci_hcd ehci_pci iTCO_wdt ehci_hcd intel_pmc_bxt tg3 watchdog at24 scsi_mod usbcore libphy i2c_i801 ioatdma i2c_smbus lpc_ich scsi_common mdio_bus configfs dca usb_common [last unloaded: thunderbolt]
    [  777.964055] RIP: 0010:ring_interrupt_active+0x27f/0x310 [thunderbolt]
    [  777.964139]  tb_ring_stop+0x167/0x1d0 [thunderbolt]
    [  777.964169]  tbstream_dev_stop+0x131/0x170 [thunderbolt_stream]
    [  777.964176]  tbstream_dev_fops_release+0xbe/0x130 [thunderbolt_stream]
