#!/bin/bash
# Prepare a Mac Pro 6,1 running Debian 13 as a trashcan node, from this Mac. Idempotent: every step can be
# rerun, and running it against an already-provisioned node (incite) changes nothing. See nodes/macpro/README.md.
#
# usage: nodes/macpro/provision.sh HOST [STEP...]
#   default steps: check disk packages tailscale rust system tblink kernel workspace
#   other steps:   reboot (reboots into the new kernel and waits), models, verify, itable (finishes ext4
#                  inode-table zeroing that `disk` paused; run while the node is idle)
#   first run:     provision.sh HOST && provision.sh HOST models reboot verify itable
# env: KVER (7.2.9-p2p), DEBS (~/dev/claude-work/trashcan-kbuild, from kernel/cross-build-7.2.sh),
#      MODELS ("gemma-4-12b-qat-gguf gemma-4-12b-assistant"), MODEL_SRC (incite)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
host=${1:?usage: provision.sh HOST [STEP...]}
shift
steps=${*:-check disk packages tailscale rust system tblink kernel workspace}
KVER=${KVER:-7.2.9-p2p}
DEBS=${DEBS:-$HOME/dev/claude-work/trashcan-kbuild}
MODELS=${MODELS:-gemma-4-12b-qat-gguf gemma-4-12b-assistant}
MODEL_SRC=${MODEL_SRC:-incite}
USER_NAME=fairfieldt
CMDLINE="quiet radeon.si_support=0 amdgpu.si_support=1 panic=10"

say() { printf '\n== %s: %s\n' "$host" "$*"; }
# Run a root shell script on the host; extra args become environment variables.
as_root() {
    local vars=""
    [ $# -eq 0 ] || vars=$(printf '%q ' "$@")
    ssh "$host" "sudo -n env $vars bash -s"
}

step_check() {
    say "check"
    if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" true; then
        echo "cannot ssh to $host; run: ssh-copy-id $USER_NAME@$host" >&2; exit 1
    fi
    if ! ssh "$host" sudo -n true 2>/dev/null; then
        echo "passwordless sudo missing; on $host run:" >&2
        echo "  echo '$USER_NAME ALL=(ALL) NOPASSWD:ALL' | sudo tee /etc/sudoers.d/90-$USER_NAME" >&2
        exit 1
    fi
    ssh "$host" 'echo "host:    $(hostname) ($(hostname -I))"
        echo "model:   $(sudo -n dmidecode -s system-product-name 2>/dev/null)"
        echo "cpu:     $(grep -m1 "model name" /proc/cpuinfo | cut -d: -f2- | xargs) x $(nproc)"
        echo "memory:  $(free -g | awk "/^Mem/{print \$2}") GiB"
        echo "os:      $(. /etc/os-release && echo "$PRETTY_NAME"), kernel $(uname -r)"
        lspci -nn | grep -E "VGA|Display" | sed "s/^/gpu:     /"
        lspci -nn | grep -c "Thunderbolt 2 NHI" | sed "s/^/tb nhis: /"'
}

# ext4 zeroes the inode tables of a new filesystem in the background after every mount (ext4lazyinit). On a
# drive without write-zeroes (the original Apple SSDs) that writes real zeroes, and each write then takes
# 400-700 ms. `disk` pauses it on such drives, now and across reboots (noinit_itable in fstab); `itable`
# finishes it in one go and removes the option.
itable_state='
uninit() { dumpe2fs "$1" 2>/dev/null | awk "/^Group [0-9]/{n++} /ITABLE_ZEROED/{z++} END{print n - z, n}"; }
no_wzeroes() {
    local d; d=$(lsblk -s -n -r -o NAME,TYPE "$1" | awk "\$2 == \"disk\" {print \$1; exit}")
    [ "$(cat /sys/block/$d/queue/write_zeroes_max_bytes 2>/dev/null || echo 0)" = 0 ]
}
fstab_opt() {   # fstab_opt add|del MOUNTPOINT: noinit_itable in that mount'"'"'s fstab options
    awk -v m="$2" -v op="$1" "BEGIN{OFS=\"\t\"} !/^#/ && \$2 == m {
            o = \$4; gsub(/(^|,)noinit_itable/, \"\", o); sub(/^,/, \"\", o); if (o == \"\") o = \"defaults\"
            if (op == \"add\") o = o \",noinit_itable\"; \$4 = o } {print}" /etc/fstab > /etc/fstab.new
    cmp -s /etc/fstab /etc/fstab.new && rm /etc/fstab.new || mv /etc/fstab.new /etc/fstab
}
'

step_disk() {
    say "disk: pause ext4 inode-table zeroing on drives without write-zeroes"
    { echo "$itable_state"; cat <<'EOF'; } | as_root
set -e
findmnt -rn -t ext4 -o TARGET,SOURCE | while read -r mnt dev; do
    read -r left total < <(uninit "$dev")
    if [ "$left" -gt 0 ] && no_wzeroes "$dev"; then
        mount -o remount,noinit_itable "$mnt"
        fstab_opt add "$mnt"
        echo "$mnt: $left of $total inode-table groups not zeroed; zeroing paused (finish with the itable step)"
    else
        echo "$mnt: $left of $total groups not zeroed$( [ "$left" -gt 0 ] && echo ', drive has write-zeroes')"
    fi
done
EOF
}

step_itable() {
    say "itable: finish ext4 inode-table zeroing (the disk is slow until it is done)"
    { echo "$itable_state"; cat <<'EOF'; } | as_root
set -e
findmnt -rn -t ext4 -o TARGET,SOURCE | while read -r mnt dev; do
    read -r left total < <(uninit "$dev")
    if [ "$left" -gt 0 ]; then
        start=$(date +%s)
        mount -o remount,init_itable=0 "$mnt"
        while read -r left total < <(sync -f "$mnt"; uninit "$dev"); [ "$left" -gt 0 ]; do
            echo "$mnt: $left of $total groups left ($(( $(date +%s) - start )) s)"; sleep 60
        done
        echo "$mnt: zeroed in $(( $(date +%s) - start )) s"
    fi
    fstab_opt del "$mnt"
    echo "$mnt: all $total groups zeroed"
done
EOF
}

step_packages() {
    say "packages"
    as_root <<'EOF'
set -e
export DEBIAN_FRONTEND=noninteractive
apt-get update -q
apt-get install -y -q --no-install-recommends -o Dpkg::Options::=--force-unsafe-io \
    build-essential git curl ca-certificates rsync pciutils usbutils lsof dmidecode pkg-config \
    firmware-amd-graphics firmware-misc-nonfree intel-microcode \
    mesa-vulkan-drivers vulkan-tools libvulkan-dev glslc spirv-headers \
    libssl-dev libcurl4-openssl-dev libelf-dev libdw-dev bc bison flex cpio debhelper cmake \
    iperf3 sockperf netcat-openbsd socat python3 smartmontools
EOF
}

step_tailscale() {
    say "tailscale"
    as_root USER_NAME="$USER_NAME" <<'EOF'
set -e
if ! command -v tailscale >/dev/null; then
    codename=$(. /etc/os-release && echo "$VERSION_CODENAME")
    curl -fsSL "https://pkgs.tailscale.com/stable/debian/${codename}.noarmor.gpg" -o /usr/share/keyrings/tailscale-archive-keyring.gpg
    cat > /etc/apt/sources.list.d/tailscale.sources <<SRC
Types: deb
URIs: https://pkgs.tailscale.com/stable/debian
Suites: ${codename}
Components: main
Signed-By: /usr/share/keyrings/tailscale-archive-keyring.gpg
SRC
    apt-get update -q
    DEBIAN_FRONTEND=noninteractive apt-get install -y -q tailscale
fi
systemctl enable --now tailscaled
tailscale set --operator="$USER_NAME"
EOF
    if ! ssh "$host" tailscale status >/dev/null 2>&1; then
        echo "not on the tailnet yet; run 'ssh $host tailscale up' and open the URL it prints"
    else
        ssh "$host" tailscale ip -4
    fi
}

step_rust() {
    say "rust"
    ssh "$host" 'set -e
        if [ ! -x ~/.cargo/bin/rustup ]; then
            curl -fsSL https://sh.rustup.rs | sh -s -- -y -q --profile default
        fi
        ~/.cargo/bin/rustup update stable >/dev/null 2>&1
        ~/.cargo/bin/rustc --version'
}

step_system() {
    say "system: config files, services, GRUB command line"
    local ip mac
    ip=$(ipconfig getifaddr en0)
    mac=$(ifconfig en0 | awk '/ether/{print $2}')
    ssh "$host" 'rm -rf ~/.cache/trashcan-node && mkdir -p ~/.cache/trashcan-node'
    rsync -a "$here/root" "$repo/scripts/fancurve.sh" "$repo/scripts/fancurve.service" \
        "$repo/scripts/system-crash-capture.py" "$repo/scripts/system-crash-capture.service" "$host":.cache/trashcan-node/
    as_root USER_NAME="$USER_NAME" CMDLINE="$CMDLINE" NETCONSOLE_TARGET="$ip/$mac" <<'EOF'
set -e
stage=/home/$USER_NAME/.cache/trashcan-node
rsync -a --chown=root:root "$stage/root/" /
install -m 0755 "$stage/fancurve.sh" /usr/local/sbin/fancurve.sh
install -m 0644 "$stage/fancurve.service" /etc/systemd/system/fancurve.service
install -m 0755 "$stage/system-crash-capture.py" /usr/local/sbin/system-crash-capture.py
install -m 0644 "$stage/system-crash-capture.service" /etc/systemd/system/system-crash-capture.service
echo "NETCONSOLE_TARGET=$NETCONSOLE_TARGET" > /etc/default/netconsole-target
rm -rf "$stage"

usermod -aG render,video "$USER_NAME"
install -d -o "$USER_NAME" -g "$USER_NAME" "/home/$USER_NAME/trashcan-exp"
touch "/home/$USER_NAME/trashcan-exp/.lock" && chown "$USER_NAME:" "/home/$USER_NAME/trashcan-exp/.lock"

sysctl -q --system
systemctl daemon-reload
systemctl enable -q netconsole.service fancurve.service tblink.service system-crash-capture.service
systemctl restart systemd-journald.service
journalctl --flush
systemctl restart netconsole.service fancurve.service tblink.service system-crash-capture.service
systemctl reload ssh
systemctl reload NetworkManager 2>/dev/null || true

# The Mac Pro doesn't resume from suspend.
systemctl mask -q sleep.target suspend.target hibernate.target hybrid-sleep.target suspend-then-hibernate.target
g=/etc/gdm3/greeter.dconf-defaults
if [ -f $g ]; then
    [ -f $g.bak-trashcan ] || cp -a $g $g.bak-trashcan
    if grep -q "^sleep-inactive-ac-type=" $g; then
        sed -i "s/^sleep-inactive-ac-type=.*/sleep-inactive-ac-type='nothing'/" $g
    else
        sed -i "s/^# *sleep-inactive-ac-type=.*/sleep-inactive-ac-type='nothing'/" $g
    fi
    grep -q "^sleep-inactive-ac-type='nothing'" $g
    /usr/share/gdm/generate-config
fi

# amdgpu (not radeon) for the Southern Islands GPUs; reboot 10 s after a panic.
[ -f /etc/default/grub.bak-trashcan ] || cp -a /etc/default/grub /etc/default/grub.bak-trashcan
sed -i "s/^GRUB_CMDLINE_LINUX_DEFAULT=.*/GRUB_CMDLINE_LINUX_DEFAULT=\"$CMDLINE\"/" /etc/default/grub
update-grub 2>/dev/null
echo "netconsole -> $NETCONSOLE_TARGET ($(systemctl is-active netconsole)), fancurve $(systemctl is-active fancurve)"
EOF
}

step_tblink() {
    say "tblink: record this node's Thunderbolt controllers and push the node table everywhere"
    local table="$here/tblink/nodes.tsv" name uuids idx n
    name=$(ssh "$host" hostname)
    uuids=$(ssh "$host" 'for n in 0000:16:00.0 0000:5d:00.0 0000:a4:00.0; do
            d=$(ls -d /sys/bus/pci/devices/$n/domain* 2>/dev/null | head -1)
            cat /sys/bus/thunderbolt/devices/${d##*domain}-0/unique_id 2>/dev/null || echo -
        done' | paste -sd '\t' -)
    case $uuids in *-$'\t'*|*$'\t'-|-) echo "a Thunderbolt controller is missing on $name: $uuids" >&2; exit 1 ;; esac
    idx=$(awk -v h="$name" '!/^#/ && $2 == h { print $1 }' "$table")
    if [ -z "$idx" ]; then
        idx=$(awk '!/^#/ && NF { if ($1 > m) m = $1 } END { print m + 1 }' "$table")
        printf '%s\t%s\t%s\n' "$idx" "$name" "$uuids" >> "$table"
    else
        awk -v h="$name" -v line="$(printf '%s\t%s\t%s' "$idx" "$name" "$uuids")" \
            'BEGIN { OFS = "\t" } !/^#/ && $2 == h { print line; next } { print }' "$table" > "$table.new" && mv "$table.new" "$table"
    fi
    echo "$name is node $idx (commit $table)"
    for n in $(awk '!/^#/ && NF { print $2 }' "$table"); do
        if ssh -o ConnectTimeout=8 "$n" 'sudo -n install -D -m 0644 /dev/stdin /etc/tblink/nodes.tsv' < "$table"; then
            echo "table -> $n"
        else
            echo "table -> $n FAILED (unreachable?)"
        fi
    done
    ssh "$host" 'sudo -n systemctl enable -q --now tblink.service; sleep 8; sudo -n tblink status'
}

step_kernel() {
    say "kernel $KVER"
    local img hdr
    img=$(ls -t "$DEBS"/linux-image-${KVER}_*_amd64.deb 2>/dev/null | head -1)
    hdr=$(ls -t "$DEBS"/linux-headers-${KVER}_*_amd64.deb 2>/dev/null | head -1)
    if [ -z "$img" ] || [ -z "$hdr" ]; then
        echo "no $KVER packages in $DEBS; build them with kernel/cross-build-7.2.sh" >&2; exit 1
    fi
    echo "packages: $(basename "$img") $(basename "$hdr")"
    ssh "$host" 'mkdir -p ~/kernel/7.2/debs'
    rsync -a "$img" "$hdr" "$host":kernel/7.2/debs/
    as_root KVER="$KVER" IMG="$(basename "$img")" HDR="$(basename "$hdr")" USER_NAME="$USER_NAME" <<'EOF'
set -e
cd /home/$USER_NAME/kernel/7.2/debs
want=$(dpkg-deb -f "$IMG" Version)
# Both packages fully installed at this version (an interrupted install leaves them half-installed).
have=$(dpkg-query -W -f '${Version} ${Status},' "linux-image-$KVER" "linux-headers-$KVER" 2>/dev/null || true)
# --force-unsafe-io: without the per-file fsyncs; an original Apple SSD needs up to seconds per cache flush.
[ "$have" = "$want install ok installed,$want install ok installed," ] || dpkg --force-unsafe-io -i "$IMG" "$HDR"
# Pin GRUB's default to this kernel by entry id, so installing other kernels never changes what boots.
update-grub 2>/dev/null
sub=$(grep -o "submenu 'Advanced options[^{]*\$menuentry_id_option '[^']*'" /boot/grub/grub.cfg | sed "s/.*option '//; s/'\$//")
ent=$(grep -o "menuentry '[^']*, with Linux $KVER'[^{]*\$menuentry_id_option '[^']*'" /boot/grub/grub.cfg | sed "s/.*option '//; s/'\$//")
[ -n "$sub" ] && [ -n "$ent" ] || { echo "no GRUB entry for $KVER" >&2; exit 1; }
sed -i "s#^GRUB_DEFAULT=.*#GRUB_DEFAULT=\"$sub>$ent\"#" /etc/default/grub
update-grub 2>/dev/null
grep "^GRUB_DEFAULT=" /etc/default/grub
echo "installed $(dpkg-query -W -f '${Version}' "linux-image-$KVER"); running $(uname -r)"
EOF
}

step_reboot() {
    say "reboot"
    echo "Unplug Thunderbolt cables that loop this Mac's own controllers first, or the firmware hangs before the chime."
    ssh "$host" 'sudo -n systemctl reboot' || true
    local i
    for i in $(seq 30); do ssh -o ConnectTimeout=4 -o BatchMode=yes "$host" true 2>/dev/null || break; sleep 2; done
    for i in $(seq 60); do
        if ssh -o ConnectTimeout=4 -o BatchMode=yes "$host" true 2>/dev/null; then
            ssh "$host" 'echo "up: $(uname -r), $(cut -d. -f1 /proc/uptime) s"'; return 0
        fi
        sleep 5
    done
    echo "$host did not come back within 5 minutes" >&2; exit 1
}

step_workspace() {
    say "workspace: sync and build"
    HOST="$host" "$repo/scripts/sync.sh" push
    ssh "$host" 'set -e; cd ~/trashcan
        ~/.cargo/bin/cargo build --release -q
        make -s -C bench/xgpu
        ls -la target/release/trashcan bench/xgpu/xgpu'
}

step_models() {
    say "models from $MODEL_SRC: $MODELS"
    local m
    ssh "$host" 'mkdir -p ~/models'
    # Streamed through this Mac, so the nodes need no keys for each other. Use LAN addresses for speed.
    for m in $MODELS; do
        if ssh "$host" "[ -d ~/models/$m ] && [ \"\$(du -sb ~/models/$m | cut -f1)\" = \"$(ssh "$MODEL_SRC" "du -sb ~/models/$m | cut -f1")\" ]"; then
            echo "$m: already there"; continue
        fi
        ssh "$MODEL_SRC" "tar -C ~/models -cf - $m" | ssh "$host" "tar -C ~/models -xf - && du -sh ~/models/$m"
    done
}

step_verify() {
    say "verify"
    local token
    token="$host-$(date +%s)"
    ssh "$host" "set -e
        mkdir -p ~/.cache
        echo \"kernel:  \$(uname -r) (want $KVER)\"; [ \"\$(uname -r)\" = $KVER ]
        grep -q 'amdgpu.si_support=1' /proc/cmdline
        for d in /sys/bus/pci/drivers/amdgpu/0000:*; do
            echo \"gpu:     \$(basename \$d) vbios \$(cat \$d/vbios_version) vram \$((\$(cat \$d/mem_info_vram_total) >> 20)) MiB\"
        done
        vulkaninfo --summary 2>/dev/null | grep -E 'deviceName' | sed 's/^ */vulkan:  /'
        echo \"render:  \$(ls /dev/dri/renderD* | xargs)\"
        echo '<6>trashcan provision: netconsole check $token' | sudo -n tee /dev/kmsg >/dev/null
        echo \"services: netconsole \$(systemctl is-active netconsole), fancurve \$(systemctl is-active fancurve)\""
    sleep 2
    if grep -q "trashcan provision: netconsole check $token" ~/dev/claude-work/trashcan-kbuild/netconsole-incite.log 2>/dev/null; then
        echo "netconsole: received on this Mac"
    else
        echo "netconsole: nothing received (is scripts/netconsole-listen.py running here?)"
    fi
    ssh "$host" 'set -e; cd ~/trashcan; exec 9>~/trashcan-exp/.lock; flock 9
        scripts/clockmon.sh ~/.cache/provision-clocks.txt & mon=$!
        (cd bench/xgpu && ./xgpu p2p-verify | tail -2)
        scripts/test-engine.sh > ~/.cache/provision-test-engine.log 2>&1 && te=PASS || te=FAIL
        kill $mon
        echo "test-engine: $te ($(grep -c "^PASS" ~/.cache/provision-test-engine.log) validations passed, log ~/.cache/provision-test-engine.log)"
        echo "max clocks (sclk0 mclk0 sclk1 mclk1): $(awk "NR>1{for(i=2;i<=6;i++) if(\$i>m[i]) m[i]=\$i} END{print m[2], m[3], m[5], m[6]}" ~/.cache/provision-clocks.txt)"'
}

for s in $steps; do
    case $s in
        check|disk|packages|tailscale|rust|system|tblink|kernel|reboot|workspace|models|verify|itable) "step_$s" ;;
        *) echo "unknown step: $s" >&2; exit 2 ;;
    esac
done
