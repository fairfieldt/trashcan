#!/bin/bash
# Prepare a Mac Pro 6,1 running Debian 13 as a trashcan node, over SSH from your workstation. Idempotent: every
# step can be rerun, and running it against a provisioned node changes nothing. See nodes/macpro/README.md.
#
# usage: nodes/macpro/provision.sh HOST [STEP...]
#   default steps: check disk packages rust system tblink kernel workspace
#   other steps:   reboot (reboots into the new kernel and waits), verify, itable (finishes ext4 inode-table
#                  zeroing that `disk` paused; run while the node is idle)
#   first run:     DEBS=DIR provision.sh HOST && provision.sh HOST reboot verify itable
# env: DEBS (kernel/build-7.3.sh's WORK directory: packages, SHA256SUMS, kernel-build-KVER.txt; needed by the
#      kernel step), KVER (7.3.0-rc7-p2p), INSTALL_ARGS (nodes/macpro/install.sh arguments, default --apply)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
host=${1:?usage: provision.sh HOST [STEP...]}
shift
steps=${*:-check disk packages rust system tblink kernel workspace}
KVER=${KVER:-7.3.0-rc7-p2p}
DEBS=${DEBS:-}
# The node's login user and home directory (the account SSH logs in as), read on first use.
USER_NAME= USER_HOME=

say() { printf '\n== %s: %s\n' "$host" "$*"; }
# Run a root shell script on the host; extra args become environment variables.
as_root() {
    local vars=""
    [ $# -eq 0 ] || vars=$(printf '%q ' "$@")
    ssh "$host" "sudo -n env $vars bash -s"
}
remote_user() {
    [ -n "$USER_NAME" ] || read -r USER_NAME USER_HOME < <(ssh "$host" 'echo "$(id -un) $HOME"')
}

step_check() {
    say "check"
    if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" true; then
        echo "cannot ssh to $host; run: ssh-copy-id $host" >&2; exit 1
    fi
    remote_user
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
    say "system: node services and configuration (nodes/macpro/install.sh)"
    local dest=.cache/trashcan-node-install
    ssh "$host" "rm -rf ~/$dest && mkdir -p ~/$dest"
    (cd "$repo" && rsync -aR nodes/macpro/install.sh nodes/macpro/root scripts/fancurve.sh scripts/fancurve.service \
        $( [ -d lights ] && echo lights ) "$host:$dest/")
    ssh "$host" "sudo -n bash $dest/nodes/macpro/install.sh ${INSTALL_ARGS:---apply}"
}

step_tblink() {
    say "tblink: the node registry comes from deploy/trashcan-cluster"
    echo "List every node in your inventory (deploy/cluster.example.toml), then run from this checkout:"
    echo "  deploy/trashcan-cluster -c cluster.toml registry --push --apply"
}

step_kernel() {
    say "kernel $KVER"
    local kdir=.cache/trashcan-kernel
    [ -n "$DEBS" ] || { echo "set DEBS to the directory kernel/build-7.3.sh left the packages in (WORK)" >&2; exit 1; }
    [ -f "$DEBS/SHA256SUMS" ] || { echo "no $DEBS/SHA256SUMS; build with kernel/build-7.3.sh" >&2; exit 1; }
    ssh "$host" "rm -rf ~/$kdir && mkdir -p ~/$kdir"
    rsync -a "$DEBS"/SHA256SUMS "$DEBS"/kernel-build-${KVER}.txt "$DEBS"/linux-image-${KVER}_*.deb \
        "$DEBS"/linux-headers-${KVER}_*.deb "$repo/kernel/install-kernel.sh" "$host:$kdir/"
    ssh "$host" "sudo -n bash $kdir/install-kernel.sh install ~/$kdir --apply"
}

step_reboot() {
    say "reboot"
    echo "Unplug Thunderbolt cables that loop the node's own controllers first, or the firmware hangs before the chime."
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
    say "workspace: copy the repository to ~/trashcan and build"
    # Git-ignored files (models, build outputs) and .git stay behind; the node's own target/ is kept.
    rsync -a --exclude=/.git/ --exclude=/target/ --filter=':- .gitignore' "$repo/" "$host":trashcan/
    ssh "$host" 'set -e; cd ~/trashcan
        ~/.cargo/bin/cargo build --release -q
        make -s -C bench/xgpu
        ls -la target/release/trashcan bench/xgpu/xgpu'
}

step_verify() {
    say "verify (uses the GPUs: make sure nothing else is using them)"
    ssh "$host" "set -e
        echo \"kernel:  \$(uname -r) (want $KVER)\"; [ \"\$(uname -r)\" = $KVER ]
        grep -q 'amdgpu.si_support=1' /proc/cmdline
        for d in /sys/bus/pci/drivers/amdgpu/0000:*; do
            echo \"gpu:     \$(basename \$d) vbios \$(cat \$d/vbios_version) vram \$((\$(cat \$d/mem_info_vram_total) >> 20)) MiB\"
        done
        vulkaninfo --summary 2>/dev/null | grep -E 'deviceName' | sed 's/^ */vulkan:  /'
        echo \"render:  \$(ls /dev/dri/renderD* | xargs)\"
        echo \"services: fancurve \$(systemctl is-active fancurve), tblink \$(systemctl is-active tblink)\""
    ssh "$host" 'set -e; cd ~/trashcan; mkdir -p ~/.cache
        model=~/models/gemma-4-12b-qat-gguf/gemma-4-12b-it-qat-q4_0.gguf
        scripts/clockmon.sh ~/.cache/provision-clocks.txt & mon=$!
        (cd bench/xgpu && ./xgpu p2p-verify | tail -2)
        if [ -f "$model" ]; then
            scripts/test-engine.sh > ~/.cache/provision-test-engine.log 2>&1 && te=PASS || te=FAIL
            echo "test-engine: $te ($(grep -c "^PASS" ~/.cache/provision-test-engine.log) validations passed, log ~/.cache/provision-test-engine.log)"
        else
            echo "test-engine: skipped (no $model)"
        fi
        kill $mon
        echo "max clocks (sclk0 mclk0 sclk1 mclk1): $(awk "NR>1{for(i=2;i<=6;i++) if(\$i>m[i]) m[i]=\$i} END{print m[2], m[3], m[5], m[6]}" ~/.cache/provision-clocks.txt)"'
}

for s in $steps; do
    case $s in
        check|disk|packages|rust|system|tblink|kernel|reboot|workspace|verify|itable) "step_$s" ;;
        *) echo "unknown step: $s" >&2; exit 2 ;;
    esac
done
