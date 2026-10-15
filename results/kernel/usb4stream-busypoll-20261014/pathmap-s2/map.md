# Path map: pathmap-s2

## Path ends with the stream devices open

Tx: the hop in the NHI adapter's table (in HopID = Tx ring); Rx: the hop in the lane adapter's table (next HopID = Rx ring). cr = initial credits, fc = ingress/egress flow control.

| Mac | kernel | link | peer | stream Tx ring>hop | stream Tx hop | stream Rx hop>ring | stream Rx hop | tbnet Tx ring>hop | tbnet Tx hop | tbnet Rx hop>ring | tbnet Rx hop | lane adapter | NHI adapter |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| incite | 7.3.0-rc7-p2p-s2 | 0-3 port 3 | inflight | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2>8 | cr 14 fc 1/1 drop 0 p5/w1 | 8>2 | cr 14 fc 1/0 drop 0 p5/w1 | port 3: nfc 0 total 60 width 2 speed 0x8; port 4: nfc 0 total 60 width 2 speed 0x8 | port 5: nfc 0 total 15 |
| incite | 7.3.0-rc7-p2p-s2 | 1-3 port 3 | enlight | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2>8 | cr 14 fc 1/1 drop 0 p5/w1 | 8>2 | cr 14 fc 1/0 drop 0 p5/w1 | port 3: nfc 0 total 60 width 2 speed 0x8; port 4: nfc 0 total 60 width 2 speed 0x8 | port 5: nfc 0 total 15 |
| inflight | 7.3.0-rc7-p2p-s2 | 0-3 port 3 | incite | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2>8 | cr 14 fc 1/1 drop 0 p5/w1 | 8>2 | cr 14 fc 1/0 drop 0 p5/w1 | port 3: nfc 0 total 60 width 2 speed 0x8; port 4: nfc 0 total 60 width 2 speed 0x8 | port 5: nfc 0 total 15 |
| inflight | 7.3.0-rc7-p2p-s2 | 1-3 port 3 | alight | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2>8 | cr 14 fc 1/1 drop 0 p5/w1 | 8>2 | cr 14 fc 1/0 drop 0 p5/w1 | port 3: nfc 0 total 60 width 2 speed 0x8; port 4: nfc 0 total 60 width 2 speed 0x8 | port 5: nfc 0 total 15 |
| alight | 7.3.0-rc7-p2p-s2 | 0-3 port 3 | indite | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2>8 | cr 14 fc 1/1 drop 0 p5/w1 | 8>2 | cr 14 fc 1/0 drop 0 p5/w1 | port 3: nfc 0 total 60 width 2 speed 0x8; port 4: nfc 0 total 60 width 2 speed 0x8 | port 5: nfc 0 total 15 |
| alight | 7.3.0-rc7-p2p-s2 | 1-3 port 3 | inflight | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2>8 | cr 14 fc 1/1 drop 0 p5/w1 | 8>2 | cr 14 fc 1/0 drop 0 p5/w1 | port 3: nfc 0 total 60 width 2 speed 0x8; port 4: nfc 0 total 60 width 2 speed 0x8 | port 5: nfc 0 total 15 |
| indite | 7.3.0-rc7-p2p-s2 | 0-3 port 3 | alight | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2>8 | cr 14 fc 1/1 drop 0 p5/w1 | 8>2 | cr 14 fc 1/0 drop 0 p5/w1 | port 3: nfc 0 total 60 width 2 speed 0x8; port 4: nfc 0 total 60 width 2 speed 0x8 | port 5: nfc 0 total 15 |
| indite | 7.3.0-rc7-p2p-s2 | 1-3 port 3 | invite | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2>8 | cr 14 fc 1/1 drop 0 p5/w1 | 8>2 | cr 14 fc 1/0 drop 0 p5/w1 | port 3: nfc 0 total 60 width 2 speed 0x8; port 4: nfc 0 total 60 width 2 speed 0x8 | port 5: nfc 0 total 15 |
| invite | 7.3.0-rc7-p2p-s2 | 0-3 port 3 | enlight | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2>8 | cr 14 fc 1/1 drop 0 p5/w1 | 8>2 | cr 14 fc 1/0 drop 0 p5/w1 | port 3: nfc 0 total 60 width 2 speed 0x8; port 4: nfc 0 total 60 width 2 speed 0x8 | port 5: nfc 0 total 15 |
| invite | 7.3.0-rc7-p2p-s2 | 1-3 port 3 | indite | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2>8 | cr 14 fc 1/1 drop 0 p5/w1 | 8>2 | cr 14 fc 1/0 drop 0 p5/w1 | port 3: nfc 0 total 60 width 2 speed 0x8; port 4: nfc 0 total 60 width 2 speed 0x8 | port 5: nfc 0 total 15 |
| enlight | 7.3.0-rc7-p2p-s2 | 0-3 port 3 | invite | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2>8 | cr 14 fc 1/1 drop 0 p5/w1 | 8>2 | cr 14 fc 1/0 drop 0 p5/w1 | port 3: nfc 0 total 60 width 2 speed 0x8; port 4: nfc 0 total 60 width 2 speed 0x8 | port 5: nfc 0 total 15 |
| enlight | 7.3.0-rc7-p2p-s2 | 1-3 port 3 | incite | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2>8 | cr 14 fc 1/1 drop 0 p5/w1 | 8>2 | cr 14 fc 1/0 drop 0 p5/w1 | port 3: nfc 0 total 60 width 2 speed 0x8; port 4: nfc 0 total 60 width 2 speed 0x8 | port 5: nfc 0 total 15 |

## Changes between the snapshots

- 0 other changes; 48 stream hops only switched on or off with the devices (enable and flow control bits)

## The hold per direction (pos: fillers each lone frame needed into the other end, per open)

| direction | pos per open | pp per open (client = sender) | lost opens | sender Tx ring>hop | sender Tx hop | receiver Rx hop>ring | receiver Rx hop | tbnet rings (sender Tx, receiver Rx) | receiver lane adapter | sender lane adapter | kernels | receiver group |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| incite->enlight | 0 0 0 0 | 0 0 0 0 | - | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2, 2 | nfc 0 total 60 width 2 speed 0x8 | nfc 0 total 60 width 2 speed 0x8 | 7.3.0-rc7-p2p-s2 > 7.3.0-rc7-p2p-s2 | busy_poll=0 ring_size=256 throttling=8192 |
| enlight->incite | 0 0 0 0 | 0 0 0 1-10 | - | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2, 2 | nfc 0 total 60 width 2 speed 0x8 | nfc 0 total 60 width 2 speed 0x8 | 7.3.0-rc7-p2p-s2 > 7.3.0-rc7-p2p-s2 | busy_poll=0 ring_size=256 throttling=8192 |
| incite->inflight | 0 0 0 0 | 0 0 0 0 | - | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2, 2 | nfc 0 total 60 width 2 speed 0x8 | nfc 0 total 60 width 2 speed 0x8 | 7.3.0-rc7-p2p-s2 > 7.3.0-rc7-p2p-s2 | busy_poll=0 ring_size=256 throttling=8192 |
| inflight->incite | 0 0 0 0 | 0 0 0 0 | - | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2, 2 | nfc 0 total 60 width 2 speed 0x8 | nfc 0 total 60 width 2 speed 0x8 | 7.3.0-rc7-p2p-s2 > 7.3.0-rc7-p2p-s2 | busy_poll=0 ring_size=256 throttling=8192 |
| inflight->alight | 0 0 0 0 | 0 0 0 0 | - | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2, 2 | nfc 0 total 60 width 2 speed 0x8 | nfc 0 total 60 width 2 speed 0x8 | 7.3.0-rc7-p2p-s2 > 7.3.0-rc7-p2p-s2 | busy_poll=0 ring_size=256 throttling=8192 |
| alight->inflight | 0 0 0 0 | 0 0 0 0 | - | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2, 2 | nfc 0 total 60 width 2 speed 0x8 | nfc 0 total 60 width 2 speed 0x8 | 7.3.0-rc7-p2p-s2 > 7.3.0-rc7-p2p-s2 | busy_poll=0 ring_size=256 throttling=8192 |
| alight->indite | 0 0 0 0 | 0 0 0 0 | - | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2, 2 | nfc 0 total 60 width 2 speed 0x8 | nfc 0 total 60 width 2 speed 0x8 | 7.3.0-rc7-p2p-s2 > 7.3.0-rc7-p2p-s2 | busy_poll=0 ring_size=256 throttling=8192 |
| indite->alight | 0 0 0 0 | 0 0 0 0 | - | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2, 2 | nfc 0 total 60 width 2 speed 0x8 | nfc 0 total 60 width 2 speed 0x8 | 7.3.0-rc7-p2p-s2 > 7.3.0-rc7-p2p-s2 | busy_poll=0 ring_size=256 throttling=8192 |
| indite->invite | 0 0 0 0 | 0 0 0 0 | - | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2, 2 | nfc 0 total 60 width 2 speed 0x8 | nfc 0 total 60 width 2 speed 0x8 | 7.3.0-rc7-p2p-s2 > 7.3.0-rc7-p2p-s2 | busy_poll=0 ring_size=256 throttling=8192 |
| invite->indite | 0 0 0 0 | 0 0 0 0 | - | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2, 2 | nfc 0 total 60 width 2 speed 0x8 | nfc 0 total 60 width 2 speed 0x8 | 7.3.0-rc7-p2p-s2 > 7.3.0-rc7-p2p-s2 | busy_poll=0 ring_size=256 throttling=8192 |
| invite->enlight | 0 0 0 0 | 0 0 0 0 | - | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2, 2 | nfc 0 total 60 width 2 speed 0x8 | nfc 0 total 60 width 2 speed 0x8 | 7.3.0-rc7-p2p-s2 > 7.3.0-rc7-p2p-s2 | busy_poll=0 ring_size=256 throttling=8192 |
| enlight->invite | 0 0 0 0 | 0 0 0 0 | - | 3>15 | cr 14 fc 1/1 drop 0 p5/w1 | 15>3 | cr 14 fc 1/0 drop 0 p5/w1 | 2, 2 | nfc 0 total 60 width 2 speed 0x8 | nfc 0 total 60 width 2 speed 0x8 | 7.3.0-rc7-p2p-s2 > 7.3.0-rc7-p2p-s2 | busy_poll=0 ring_size=256 throttling=8192 |

## Settings against the hold

Held (pos, any open with a filler): none. No hold: incite->enlight, enlight->incite, incite->inflight, inflight->incite, inflight->alight, alight->inflight, alight->indite, indite->alight, indite->invite, invite->indite, invite->enlight, enlight->invite. No answer: none.

| setting | in the directions that held | in the others | splits them |
|---|---|---|---|
| receiver_ctl | - | 0-0, 1-0 |  |
| receiver_group | - | busy_poll=0 ring_size=256 throttling=8192 |  |
| receiver_kernel | - | 7.3.0-rc7-p2p-s2 |  |
| rx_credits | - | 14 |  |
| rx_drop | - | 0 |  |
| rx_fc | - | 1/0 |  |
| rx_lane_buffers | - | nfc 0 total 60 width 2 speed 0x8 |  |
| rx_nhi_buffers | - | nfc 0 total 15 |  |
| rx_ring | - | 3 |  |
| sender_ctl | - | 0-0, 1-0 |  |
| sender_kernel | - | 7.3.0-rc7-p2p-s2 |  |
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

incite: tbstream0 ok, tbstream1 ok; inflight: tbstream0 ok eofs 1, tbstream1 ok; alight: tbstream0 ok eofs 1, tbstream1 ok eofs 1; indite: tbstream0 ok, tbstream1 ok; invite: tbstream0 ok, tbstream1 ok eofs 1; enlight: tbstream0 ok eofs 1, tbstream1 ok eofs 1

## Kernel log (last Thunderbolt lines per Mac, closed snapshot)

incite:
    [   94.778752] thunderbolt 0-3: failed to send properties changed notification
    [   94.778753] thunderbolt 1-3: failed to send properties changed notification
    [   96.826914] thunderbolt 0-3: failed to send properties changed notification
    [   96.828729] ACPI: bus type thunderbolt unregistered
    [  100.561330] ACPI: bus type thunderbolt registered
    [  100.561850] thunderbolt 0000:16:00.0: device link creation from 0000:15:00.0 failed
    [  100.565133] thunderbolt 0000:5d:00.0: device link creation from 0000:5c:00.0 failed
    [  100.569770] thunderbolt 0000:a4:00.0: device link creation from 0000:a3:00.0 failed
    [  105.883835] thunderbolt 1-3: new host found, vendor=0x1d6b device=0x4
    [  105.883860] thunderbolt 1-3: Linux enlight
    [  105.883868] thunderbolt 0-3: new host found, vendor=0x1d6b device=0x4
    [  105.883884] thunderbolt 0-3: Linux inflight
inflight:
    [   58.735371] thunderbolt 0-3: error waiting for link width to become 14
    [   59.740778] thunderbolt 0-3: new host found, vendor=0x1d6b device=0x4
    [   59.740798] thunderbolt 0-3: Linux incite
    [   74.403369] ACPI: bus type thunderbolt unregistered
    [   81.207729] ACPI: bus type thunderbolt registered
    [   81.208260] thunderbolt 0000:16:00.0: device link creation from 0000:15:00.0 failed
    [   81.211963] thunderbolt 0000:5d:00.0: device link creation from 0000:5c:00.0 failed
    [   81.217237] thunderbolt 0000:a4:00.0: device link creation from 0000:a3:00.0 failed
    [   84.504177] thunderbolt 1-3: new host found, vendor=0x1d6b device=0x4
    [   84.504204] thunderbolt 1-3: Linux alight
    [   86.552082] thunderbolt 0-3: new host found, vendor=0x1d6b device=0x4
    [   86.552108] thunderbolt 0-3: Linux incite
alight:
    [   50.962021] thunderbolt 0000:16:00.0: 0:4: failed to reach state TB_PORT_UP. Ignoring port...
    [   51.994222] thunderbolt 0-3: new host found, vendor=0x1d6b device=0x4
    [   51.994245] thunderbolt 0-3: Linux indite
    [   74.560567] ACPI: bus type thunderbolt unregistered
    [   81.348918] ACPI: bus type thunderbolt registered
    [   81.349437] thunderbolt 0000:16:00.0: device link creation from 0000:15:00.0 failed
    [   81.354012] thunderbolt 0000:5d:00.0: device link creation from 0000:5c:00.0 failed
    [   81.358524] thunderbolt 0000:a4:00.0: device link creation from 0000:a3:00.0 failed
    [   86.679030] thunderbolt 0-3: new host found, vendor=0x1d6b device=0x4
    [   86.679064] thunderbolt 0-3: Linux indite
    [   86.679122] thunderbolt 1-3: new host found, vendor=0x1d6b device=0x4
    [   86.679144] thunderbolt 1-3: Linux inflight
indite:
    [   75.457071] thunderbolt 0-3: error waiting for link width to become 14
    [   76.463491] thunderbolt 0-3: new host found, vendor=0x1d6b device=0x4
    [   76.463513] thunderbolt 0-3: Linux alight
    [   92.104059] ACPI: bus type thunderbolt unregistered
    [   98.903999] ACPI: bus type thunderbolt registered
    [   98.904443] thunderbolt 0000:16:00.0: device link creation from 0000:15:00.0 failed
    [   98.907560] thunderbolt 0000:5d:00.0: device link creation from 0000:5c:00.0 failed
    [   98.911873] thunderbolt 0000:a4:00.0: device link creation from 0000:a3:00.0 failed
    [  103.215774] thunderbolt 1-3: new host found, vendor=0x1d6b device=0x4
    [  103.215788] thunderbolt 1-3: Linux invite
    [  104.239972] thunderbolt 0-3: new host found, vendor=0x1d6b device=0x4
    [  104.239997] thunderbolt 0-3: Linux alight
invite:
    [   66.297795] thunderbolt 0-3: Linux enlight
    [   68.345754] thunderbolt 1-3: new host found, vendor=0x1d6b device=0x4
    [   68.345779] thunderbolt 1-3: Linux indite
    [   91.961623] ACPI: bus type thunderbolt unregistered
    [   98.816513] ACPI: bus type thunderbolt registered
    [   98.817030] thunderbolt 0000:16:00.0: device link creation from 0000:15:00.0 failed
    [   98.820435] thunderbolt 0000:5d:00.0: device link creation from 0000:5c:00.0 failed
    [   98.825148] thunderbolt 0000:a4:00.0: device link creation from 0000:a3:00.0 failed
    [  103.100678] thunderbolt 0-3: new host found, vendor=0x1d6b device=0x4
    [  103.100704] thunderbolt 0-3: Linux enlight
    [  104.156555] thunderbolt 1-3: new host found, vendor=0x1d6b device=0x4
    [  104.156580] thunderbolt 1-3: Linux indite
enlight:
    [   66.210368] thunderbolt 1-3: Linux incite
    [   67.234298] thunderbolt 0-3: new host found, vendor=0x1d6b device=0x4
    [   67.234323] thunderbolt 0-3: Linux invite
    [   90.881391] ACPI: bus type thunderbolt unregistered
    [   97.717547] ACPI: bus type thunderbolt registered
    [   97.718249] thunderbolt 0000:16:00.0: device link creation from 0000:15:00.0 failed
    [   97.725002] thunderbolt 0000:5d:00.0: device link creation from 0000:5c:00.0 failed
    [   97.730639] thunderbolt 0000:a4:00.0: device link creation from 0000:a3:00.0 failed
    [  100.994999] thunderbolt 1-3: new host found, vendor=0x1d6b device=0x4
    [  100.995025] thunderbolt 1-3: Linux incite
    [  103.042881] thunderbolt 0-3: new host found, vendor=0x1d6b device=0x4
    [  103.042907] thunderbolt 0-3: Linux invite
