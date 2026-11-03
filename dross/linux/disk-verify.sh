set -e
[ "$(cat /disk/persist/renamed)" = per ]
[ "$(ls -l /disk/persist/renamed | cut -c1-10)" = -rw------- ]
[ "$(cat /disk/persist/link)" = per ]
[ ! -e /disk/persist/deleted ]
[ ! -e /disk/persist/empty ]
[ "$(dd if=/disk/test/random.bin bs=65536 count=16 2>/dev/null | sha256sum | cut -c1-64)" = "$(sha256sum /disk/persist/binary | cut -c1-64)" ]
sync
echo 'disk persistence: ok'
