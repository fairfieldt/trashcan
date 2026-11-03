set -e
mkdir /disk/persist
printf 'persistent-data' > /disk/persist/created
printf '+append' >> /disk/persist/created
chmod 600 /disk/persist/created
[ "$(ls -l /disk/persist/created | cut -c1-10)" = -rw------- ]
touch /disk/persist/created
[ "$(cat /disk/persist/created)" = persistent-data+append ]
dd if=/disk/test/random.bin bs=65536 count=16 2>/dev/null > /disk/persist/binary
[ "$(dd if=/disk/test/random.bin bs=65536 count=16 2>/dev/null | sha256sum | cut -c1-64)" = "$(sha256sum /disk/persist/binary | cut -c1-64)" ]
mv /disk/persist/created /disk/persist/renamed
ln -s renamed /disk/persist/link
[ "$(cat /disk/persist/link)" = persistent-data+append ]
truncate -s 8192 /disk/persist/renamed
[ "$(wc -c < /disk/persist/renamed)" = 8192 ]
truncate -s 3 /disk/persist/renamed
[ "$(cat /disk/persist/renamed)" = per ]
printf delete > /disk/persist/deleted
rm /disk/persist/deleted
[ ! -e /disk/persist/deleted ]
! rmdir /disk/persist 2>/dev/null
mkdir /disk/persist/empty
rmdir /disk/persist/empty
sync
echo 'disk write: ok'
