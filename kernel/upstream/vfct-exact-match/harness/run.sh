#!/bin/bash
# Replays a VFCT dump through the old and new amdgpu/radeon VFCT lookup code, compiled unchanged in userspace.
# usage: run.sh KERNEL_TREE OLD_REV VFCT.bin
#   KERNEL_TREE  git checkout with the series applied (new code = working tree)
#   OLD_REV      revision without the series, e.g. HEAD~2
#   VFCT.bin     sudo cat /sys/firmware/acpi/tables/VFCT > VFCT.bin (on incite)
# On macOS 27 with Xcode 26.2 selected, link with DEVELOPER_DIR=/Library/Developer/CommandLineTools.
set -euo pipefail
tree=$(cd "$1" && pwd); old=$2; vfct=$(cd "$(dirname "$3")" && pwd)/$(basename "$3")
here=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d "$here/build.XXXXXX"); trap 'rm -rf "$work"' EXIT
cp "$here"/{shim.h,variant.c,main.c} "$work"; cd "$work"

# VFCT typedefs from radeon's atombios.h (same layout as amdgpu's)
awk '/AMD_ACPI_DESCRIPTION_HEADER SHeader/{p=1; print "typedef struct {"} p{print} /}GOP_VBIOS_CONTENT;/{exit}' \
	"$tree/drivers/gpu/drm/radeon/atombios.h" > vfct_types.h
# X_acpi_vfct_match() through the end of X_acpi_vfct_bios()
extract() { awk -v d="$1" '$0 ~ "^static (bool|int) "d"_acpi_vfct_match\\(" {p=1} p{print}
	p && $0 ~ "^static bool "d"_acpi_vfct_bios\\(" {b=1} b && /^}/ {exit}'; }
for d in amdgpu radeon; do
	f=drivers/gpu/drm/radeon/radeon_bios.c; [ $d = amdgpu ] && f=drivers/gpu/drm/amd/amdgpu/amdgpu_bios.c
	git -C "$tree" show "$old:$f" | extract $d > ${d}_old.inc
	extract $d < "$tree/$f" > ${d}_new.inc
	for s in old new; do
		extra=; [ $d = amdgpu ] && extra=-DIS_AMDGPU
		cc -std=gnu11 -Wall -Wno-unused-function -Wno-unused-parameter $extra \
			-DVARIANT_INC="\"${d}_${s}.inc\"" -DENTRY=${d}_${s} -c variant.c -o ${d}_${s}.o
	done
done
cc -std=gnu11 -Wall main.c ./*.o -o replay
./replay "$vfct"
