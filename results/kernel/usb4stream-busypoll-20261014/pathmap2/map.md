# Path map: pathmap2

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

- 0 other changes; 48 stream hops only switched on or off with the devices (enable and flow control bits)

## The hold per direction (pos: fillers each lone frame needed into the other end, per open)

| direction | pos per open | pp per open (client = sender) | lost opens | sender Tx ring>hop | sender Tx hop | receiver Rx hop>ring | receiver Rx hop | tbnet rings (sender Tx, receiver Rx) | receiver lane adapter | sender lane adapter | kernels | receiver group |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| incite->enlight | 0 0 0 0 0 0 | 0 0 0 0 0 0 | - | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2, 2 | nfc 0 total 60 width 2 speed 0x8 | nfc 0 total 60 width 2 speed 0x8 | 7.3.0-rc7-p2p > 7.3.0-rc7-p2p-s3 | busy_poll=0 paths_last=0 poll_mode=0 ring_size=256 rx_e2e=1 rx_eof_close=1 rx_repost=0 throttling=8192 |
| enlight->incite | 0 0 0 0 0 0 | 0 0 0 0 0 0 | - | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2, 2 | nfc 0 total 60 width 2 speed 0x8 | nfc 0 total 60 width 2 speed 0x8 | 7.3.0-rc7-p2p-s3 > 7.3.0-rc7-p2p | busy_poll=0 ring_size=256 throttling=8192 |
| incite->inflight | x x x x x x | x x x x x x | run 1 no answer at incite (client), stopped at inflight (server); run 4 no answer at incite (client), stopped at inflight (server); run 5 no answer at incite (client), stopped at inflight (server); run 8 no answer at incite (client), stopped at inflight (server); run 9 no answer at incite (client), stopped at inflight (server); run 12 no answer at incite (client), stopped at inflight (server); run 13 ENXIO at incite (client), ENXIO at inflight (server); run 16 ENXIO at incite (client), ENXIO at inflight (server); run 17 ENXIO at incite (client), ENXIO at inflight (server); run 20 ENXIO at incite (client), ENXIO at inflight (server); run 21 ENXIO at incite (client), ENXIO at inflight (server); run 24 ENXIO at incite (client), ENXIO at inflight (server) | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2, 2 | nfc 0 total 60 width 2 speed 0x8 | nfc 0 total 60 width 2 speed 0x8 | 7.3.0-rc7-p2p > 7.3.0-rc7-p2p | busy_poll=0 ring_size=256 throttling=8192 |
| inflight->incite | x 0 x 0 x 0 | 2-10 2-10 2-10 2-10 2-10 2-10 | run 2 ENXIO at inflight (client), stopped at incite (server); run 6 ENXIO at inflight (client), stopped at incite (server); run 10 ENXIO at inflight (client), stopped at incite (server) | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2, 2 | nfc 0 total 60 width 2 speed 0x8 | nfc 0 total 60 width 2 speed 0x8 | 7.3.0-rc7-p2p > 7.3.0-rc7-p2p | busy_poll=0 ring_size=256 throttling=8192 |
| inflight->alight | x x x x x x | x x x x x x | run 49 no answer at inflight (client), stopped at alight (server); run 52 no answer at inflight (client), stopped at alight (server); run 53 no answer at inflight (client), stopped at alight (server); run 56 no answer at inflight (client), stopped at alight (server); run 57 no answer at inflight (client), stopped at alight (server); run 60 no answer at inflight (client), stopped at alight (server); run 61 ENXIO at inflight (client), ENXIO at alight (server); run 64 ENXIO at inflight (client), ENXIO at alight (server); run 65 ENXIO at inflight (client), ENXIO at alight (server); run 68 ENXIO at inflight (client), ENXIO at alight (server); run 69 ENXIO at inflight (client), ENXIO at alight (server); run 72 ENXIO at inflight (client), ENXIO at alight (server) | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2, 2 | nfc 0 total 60 width 2 speed 0x8 | nfc 0 total 60 width 2 speed 0x8 | 7.3.0-rc7-p2p > 7.3.0-rc7-p2p | busy_poll=0 ring_size=256 throttling=8192 |
| alight->inflight | x 0 x 0 x 0 | 1-10 1-10 1-10 1-10 1-10 1-10 | run 50 ENXIO at alight (client), stopped at inflight (server); run 54 ENXIO at alight (client), stopped at inflight (server); run 58 ENXIO at alight (client), stopped at inflight (server) | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2, 2 | nfc 0 total 60 width 2 speed 0x8 | nfc 0 total 60 width 2 speed 0x8 | 7.3.0-rc7-p2p > 7.3.0-rc7-p2p | busy_poll=0 ring_size=256 throttling=8192 |
| alight->indite | x x x x x x | x x x x x x | run 73 no answer at alight (client), stopped at indite (server); run 76 no answer at alight (client), stopped at indite (server); run 77 no answer at alight (client), stopped at indite (server); run 80 no answer at alight (client), stopped at indite (server); run 81 no answer at alight (client), stopped at indite (server); run 84 no answer at alight (client), stopped at indite (server); run 85 ENXIO at alight (client), ENXIO at indite (server); run 88 ENXIO at alight (client), ENXIO at indite (server); run 89 ENXIO at alight (client), ENXIO at indite (server); run 92 ENXIO at alight (client), ENXIO at indite (server); run 93 ENXIO at alight (client), ENXIO at indite (server); run 96 ENXIO at alight (client), ENXIO at indite (server) | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2, 2 | nfc 0 total 60 width 2 speed 0x8 | nfc 0 total 60 width 2 speed 0x8 | 7.3.0-rc7-p2p > 7.3.0-rc7-p2p | busy_poll=0 ring_size=256 throttling=8192 |
| indite->alight | x 0 x 0 x 0 | 2-10 2-10 2-10 2-10 2-10 2-10 | run 74 ENXIO at indite (client), stopped at alight (server); run 78 ENXIO at indite (client), stopped at alight (server); run 82 ENXIO at indite (client), stopped at alight (server) | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2, 2 | nfc 0 total 60 width 2 speed 0x8 | nfc 0 total 60 width 2 speed 0x8 | 7.3.0-rc7-p2p > 7.3.0-rc7-p2p | busy_poll=0 ring_size=256 throttling=8192 |
| indite->invite | 0 0 0 0 0 0 | 0 0 0 0 0 0 | - | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2, 2 | nfc 0 total 60 width 2 speed 0x8 | nfc 0 total 60 width 2 speed 0x8 | 7.3.0-rc7-p2p > 7.3.0-rc7-p2p-s3 | busy_poll=0 paths_last=0 poll_mode=0 ring_size=256 rx_e2e=1 rx_eof_close=1 rx_repost=0 throttling=8192 |
| invite->indite | 0 0 0 0 0 0 | 0 0 0 0 0 0 | - | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2, 2 | nfc 0 total 60 width 2 speed 0x8 | nfc 0 total 60 width 2 speed 0x8 | 7.3.0-rc7-p2p-s3 > 7.3.0-rc7-p2p | busy_poll=0 ring_size=256 throttling=8192 |
| invite->enlight | 0 0 0 0 0 0 | 0 0 0 0 0 0 | - | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2, 2 | nfc 0 total 60 width 2 speed 0x8 | nfc 0 total 60 width 2 speed 0x8 | 7.3.0-rc7-p2p-s3 > 7.3.0-rc7-p2p-s3 | busy_poll=0 paths_last=0 poll_mode=0 ring_size=256 rx_e2e=1 rx_eof_close=1 rx_repost=0 throttling=8192 |
| enlight->invite | 0 0 0 0 0 0 | 0 0 0 0 0 0 | - | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2, 2 | nfc 0 total 60 width 2 speed 0x8 | nfc 0 total 60 width 2 speed 0x8 | 7.3.0-rc7-p2p-s3 > 7.3.0-rc7-p2p-s3 | busy_poll=0 paths_last=0 poll_mode=0 ring_size=256 rx_e2e=1 rx_eof_close=1 rx_repost=0 throttling=8192 |

## Settings against the hold

Held (pos, any open with a filler): none. No hold: incite->enlight, enlight->incite, inflight->incite, alight->inflight, indite->alight, indite->invite, invite->indite, invite->enlight, enlight->invite. No answer: incite->inflight, inflight->alight, alight->indite.

| setting | in the directions that held | in the others | splits them |
|---|---|---|---|
| receiver_ctl | - | 0-0, 1-0 |  |
| receiver_group | - | busy_poll=0 paths_last=0 poll_mode=0 ring_size=256 rx_e2e=1 rx_eof_close=1 rx_repost=0 throttling=8192, busy_poll=0 ring_size=256 throttling=8192 |  |
| receiver_kernel | - | 7.3.0-rc7-p2p, 7.3.0-rc7-p2p-s3 |  |
| rx_credits | - | 14 |  |
| rx_drop | - | 0 |  |
| rx_fc | - | 1/0 |  |
| rx_lane_buffers | - | nfc 0 total 60 width 2 speed 0x8 |  |
| rx_nhi_buffers | - | nfc 0 total 15 |  |
| rx_ring | - | 3 |  |
| sender_ctl | - | 0-0, 1-0 |  |
| sender_kernel | - | 7.3.0-rc7-p2p, 7.3.0-rc7-p2p-s3 |  |
| stream_rx_before_tbnet | - | False |  |
| stream_tx_before_tbnet | - | False |  |
| tbnet_rx_ring | - | 2 |  |
| tbnet_tx_ring | - | 2 |  |
| tx_credits | - | 14 |  |
| tx_drop | - | 0 |  |
| tx_fc | - | 1/1 |  |
| tx_lane_buffers | - | nfc 0 total 60 width 2 speed 0x8 |  |
| tx_nhi_buffers | - | nfc 0 total 15 |  |
| tx_prio | - | 5/1 |  |
| tx_ring | - | 3 |  |

## Devices held open for the open snapshot

incite: tbstream0 ok, tbstream1 ok; inflight: tbstream0 ok eofs 29, tbstream1 ok; alight: tbstream0 ok, tbstream1 ok; indite: tbstream0 ok eofs 29, tbstream1 ok; invite: tbstream0 ok, tbstream1 ok; enlight: tbstream0 ok eofs 1, tbstream1 ok eofs 1

## Kernel log (last Thunderbolt lines per Mac, closed snapshot)

incite:
    [ 8471.875187] thunderbolt 1-3: Linux enlight
    [ 8471.875305] thunderbolt 0-3: new host found, vendor=0x1d6b device=0x4
    [ 8471.875321] thunderbolt 0-3: Linux inflight
    [12004.801628] ACPI: bus type thunderbolt unregistered
    [12011.649904] ACPI: bus type thunderbolt registered
    [12011.650440] thunderbolt 0000:16:00.0: device link creation from 0000:15:00.0 failed
    [12011.654304] thunderbolt 0000:5d:00.0: device link creation from 0000:5c:00.0 failed
    [12011.659694] thunderbolt 0000:a4:00.0: device link creation from 0000:a3:00.0 failed
    [12016.969400] thunderbolt 0-3: new host found, vendor=0x1d6b device=0x4
    [12016.969420] thunderbolt 0-3: Linux inflight
    [12016.969448] thunderbolt 1-3: new host found, vendor=0x1d6b device=0x4
    [12016.969465] thunderbolt 1-3: Linux enlight
inflight:
    [13868.584447] thunderbolt 0-3: failed to send properties changed notification
    [13868.584448] thunderbolt 1-3: failed to send properties changed notification
    [13870.632516] thunderbolt 0-3: failed to send properties changed notification
    [13870.634175] ACPI: bus type thunderbolt unregistered
    [13874.391333] ACPI: bus type thunderbolt registered
    [13874.392216] thunderbolt 0000:16:00.0: device link creation from 0000:15:00.0 failed
    [13874.397479] thunderbolt 0000:5d:00.0: device link creation from 0000:5c:00.0 failed
    [13874.401519] thunderbolt 0000:a4:00.0: device link creation from 0000:a3:00.0 failed
    [13877.673219] thunderbolt 0-3: new host found, vendor=0x1d6b device=0x4
    [13877.673238] thunderbolt 1-3: new host found, vendor=0x1d6b device=0x4
    [13877.673244] thunderbolt 0-3: Linux incite
    [13877.673249] thunderbolt 1-3: Linux alight
alight:
    [ 8913.214865] tbstream: RX CRC error
    [ 8918.656960] tbstream: RX CRC error
    [ 9786.445810] thunderbolt 0-3: failed to send properties changed notification
    [ 9786.447240] ACPI: bus type thunderbolt unregistered
    [ 9792.200481] ACPI: bus type thunderbolt registered
    [ 9792.200998] thunderbolt 0000:16:00.0: device link creation from 0000:15:00.0 failed
    [ 9792.208434] thunderbolt 0000:5d:00.0: device link creation from 0000:5c:00.0 failed
    [ 9792.215529] thunderbolt 0000:a4:00.0: device link creation from 0000:a3:00.0 failed
    [ 9797.518747] thunderbolt 0-3: new host found, vendor=0x1d6b device=0x4
    [ 9797.518773] thunderbolt 0-3: Linux indite
    [ 9797.554705] thunderbolt 1-3: new host found, vendor=0x1d6b device=0x4
    [ 9797.554715] thunderbolt 1-3: Linux inflight
indite:
    [ 8996.183205] thunderbolt 1-3: new host found, vendor=0x1d6b device=0x4
    [ 8996.183221] thunderbolt 0-3: Linux alight
    [ 8996.183231] thunderbolt 1-3: Linux invite
    [12530.175474] ACPI: bus type thunderbolt unregistered
    [12537.030361] ACPI: bus type thunderbolt registered
    [12537.031241] thunderbolt 0000:16:00.0: device link creation from 0000:15:00.0 failed
    [12537.035197] thunderbolt 0000:5d:00.0: device link creation from 0000:5c:00.0 failed
    [12537.038951] thunderbolt 0000:a4:00.0: device link creation from 0000:a3:00.0 failed
    [12540.303879] thunderbolt 1-3: new host found, vendor=0x1d6b device=0x4
    [12540.303904] thunderbolt 1-3: Linux invite
    [12540.306812] thunderbolt 0-3: new host found, vendor=0x1d6b device=0x4
    [12540.306822] thunderbolt 0-3: Linux alight
invite:
    [  777.978557]  tbstream_dev_stop+0x131/0x170 [thunderbolt_stream]
    [  777.978567]  tbstream_dev_fops_release+0xbe/0x130 [thunderbolt_stream]
    [ 3610.919023] thunderbolt 1-3: failed to send properties changed notification
    [ 3610.921795] ACPI: bus type thunderbolt unregistered
    [ 3616.685505] ACPI: bus type thunderbolt registered
    [ 3616.686022] thunderbolt 0000:16:00.0: device link creation from 0000:15:00.0 failed
    [ 3616.689727] thunderbolt 0000:5d:00.0: device link creation from 0000:5c:00.0 failed
    [ 3616.693604] thunderbolt 0000:a4:00.0: device link creation from 0000:a3:00.0 failed
    [ 3619.975824] thunderbolt 0-3: new host found, vendor=0x1d6b device=0x4
    [ 3619.975849] thunderbolt 0-3: Linux enlight
    [ 3622.023550] thunderbolt 1-3: new host found, vendor=0x1d6b device=0x4
    [ 3622.023566] thunderbolt 1-3: Linux indite
enlight:
    [  777.964169]  tbstream_dev_stop+0x131/0x170 [thunderbolt_stream]
    [  777.964176]  tbstream_dev_fops_release+0xbe/0x130 [thunderbolt_stream]
    [ 3610.892366] thunderbolt 1-3: failed to send properties changed notification
    [ 3610.895148] ACPI: bus type thunderbolt unregistered
    [ 3616.669905] ACPI: bus type thunderbolt registered
    [ 3616.670442] thunderbolt 0000:16:00.0: device link creation from 0000:15:00.0 failed
    [ 3616.675838] thunderbolt 0000:5d:00.0: device link creation from 0000:5c:00.0 failed
    [ 3616.680611] thunderbolt 0000:a4:00.0: device link creation from 0000:a3:00.0 failed
    [ 3620.946329] thunderbolt 1-3: new host found, vendor=0x1d6b device=0x4
    [ 3620.946355] thunderbolt 1-3: Linux incite
    [ 3621.967237] thunderbolt 0-3: new host found, vendor=0x1d6b device=0x4
    [ 3621.967263] thunderbolt 0-3: Linux invite
