#!/bin/bash
# Rebuilds and checks one upstream series in the kbuild VM: applies SERIES/series/0*.patch (not the cover letter) on
# the base in SERIES/check.conf, builds the touched objects with W=1 and sparse (C=2) on the base and after each
# patch, runs checkpatch --strict and get_maintainer.pl, and regenerates SERIES/series/ with git format-patch
# (cover letter from SERIES/cover.txt: subject line, blank line, body; notes below a patch's "---" from
# SERIES/notes/N.txt for patch N). Nothing is sent and nothing is signed off.
# usage (in the VM): check-series.sh SERIES_DIR LOG_DIR
#   LINUX  kernel git repository with the remotes named in check.conf (default ~/a18/linux)
#   WT     scratch worktree/build directory (default ~/a18/check/NAME)
#   SPARSE sparse binary (default ~/a18/sparse/bin/sparse: Debian 13's 0.6.4 is too old for the kernel's
#          checker-valid.sh, so sparse is built from git.kernel.org/pub/scm/devel/sparse/sparse.git)
# check.conf sets NAME, BASE (a ref such as hwmon/hwmon-next), BASE_COMMIT (the commit it must resolve to, so a
# moved branch is noticed), CONFIGS (scripts/config arguments), OBJS (objects to build), optionally
# EXTRA_CONFIGS (a second configuration to build OBJS in, e.g. without ACPI) and ALSO_APPLY (more refs the series
# must apply to).
set -euo pipefail
sdir=$(cd "$1" && pwd); logs=$2
# shellcheck source=/dev/null
. "$sdir/check.conf"
linux=${LINUX:-$HOME/a18/linux}
wt=${WT:-$HOME/a18/check/$NAME}
sparse=${SPARSE:-$HOME/a18/sparse/bin/sparse}
kmake=(make -s -j"$(nproc)" ARCH=x86_64 CROSS_COMPILE=x86_64-linux-gnu-)
mkdir -p "$logs"; logs=$(cd "$logs" && pwd)
exec > >(tee "$logs/check.log") 2>&1

resolved=$(git -C "$linux" rev-parse "$BASE^{commit}")
[ "$resolved" = "$(git -C "$linux" rev-parse "$BASE_COMMIT^{commit}")" ] ||
	{ echo "$BASE is now $resolved, not $BASE_COMMIT: fetch, recheck and update check.conf"; exit 1; }
echo "sparse $("$sparse" --version), $(x86_64-linux-gnu-gcc --version | head -1)"
echo "== $NAME on $BASE = $(git -C "$linux" log -1 --format='%h ("%s")' "$resolved")"
patches=("$sdir"/series/0*.patch); patches=("${patches[@]/*0000-cover-letter.patch}")
mapfile -t patches < <(printf '%s\n' "${patches[@]}" | grep . )
n=${#patches[@]}

rm -rf "$wt"; git -C "$linux" worktree prune
git -C "$linux" worktree add -q -B "a18/$NAME" "$wt/src" "$resolved"
cd "$wt/src"
git am -q "${patches[@]}"
echo "applied $n patch(es): $(git log --format=%h "$resolved"..HEAD | tac | tr '\n' ' ')"
for ref in ${ALSO_APPLY:-}; do
	git -C "$linux" worktree add -q --detach "$wt/also" "$ref"
	if git -C "$wt/also" am -q "${patches[@]}" >/dev/null 2>&1; then
		echo "also applies to $ref ($(git -C "$linux" rev-parse --short "$ref"))"
	else
		echo "DOES NOT APPLY to $ref"; git -C "$wt/also" am --abort
	fi
	git -C "$linux" worktree remove --force "$wt/also"
done

configure() { # OUT CONFIG_ARGS...
	local out=$1; shift
	"${kmake[@]}" O="$out" defconfig >/dev/null
	./scripts/config --file "$out/.config" "$@"
	"${kmake[@]}" O="$out" olddefconfig >/dev/null
	echo "config $(basename "$out"): $*"
	local a sym
	for a in "$@"; do
		case $a in --enable|--module|--disable) continue ;; esac
		sym=$a
		grep -E -q "^(CONFIG_$sym=[ym]|# CONFIG_$sym is not set)" "$out/.config" &&
			echo "  $(grep -E "^(CONFIG_$sym=|# CONFIG_$sym )" "$out/.config")" ||
			{ echo "  CONFIG_$sym missing after olddefconfig"; exit 1; }
	done
	"${kmake[@]}" O="$out" prepare >/dev/null
}
build() { # OUT LABEL: W=1 + sparse on OBJS; warnings go to LOG_DIR/build-LABEL.log
	local out=$1 label=$2 o
	for o in $OBJS; do rm -f "$out/$o"; done
	if "${kmake[@]}" O="$out" W=1 C=2 CHECK="$sparse" $OBJS > "$logs/build-$label.log" 2>&1; then
		if grep -q 'is not available or not up to date' "$logs/build-$label.log"; then
			echo "build $label: sparse did not run"; exit 1
		fi
		echo "build $label: ok, $(grep -c -E 'warning|error' "$logs/build-$label.log" || true) warning/error lines"
	else
		echo "build $label: FAILED"; cat "$logs/build-$label.log"; exit 1
	fi
}
cfgs=(main); configure "$wt/out-main" $CONFIGS
if [ -n "${EXTRA_CONFIGS:-}" ]; then cfgs+=(extra); configure "$wt/out-extra" $EXTRA_CONFIGS; fi
for c in "${cfgs[@]}"; do
	git checkout -q "$resolved"; build "$wt/out-$c" "$c-base"
	for i in $(seq 1 "$n"); do
		git checkout -q "a18/$NAME~$((n - i))"; build "$wt/out-$c" "$c-patch$i"
	done
	git checkout -q "a18/$NAME"
	# warning lines the series adds (file:line positions shift, so compare without them)
	strip() { sed -E 's/:[0-9]+:[0-9]+:/:/; s/:[0-9]+:/:/' "$1" | grep -E 'warning|error' | sort -u || true; }
	comm -13 <(strip "$logs/build-$c-base.log") <(strip "$logs/build-$c-patch$n.log") > "$logs/new-warnings-$c.txt"
	echo "new W=1/sparse warnings with the series ($c): $(wc -l < "$logs/new-warnings-$c.txt")"
	cat "$logs/new-warnings-$c.txt"
done

if [ -f "$sdir/cover.txt" ]; then
	git config "branch.a18/$NAME.description" "$(cat "$sdir/cover.txt")"
	cover=(--cover-letter --cover-from-description=subject)
else
	cover=()
fi
notes=()
for i in $(seq 1 "$n"); do
	if [ -f "$sdir/notes/$i.txt" ]; then
		git notes add -f -F "$sdir/notes/$i.txt" "a18/$NAME~$((n - i))"; notes=(--notes)
	fi
done
rm -rf "$wt/series"
git format-patch -q -o "$wt/series" --base="$resolved" "${cover[@]}" "${notes[@]}" "$resolved"..HEAD
echo "format-patch: $(cd "$wt/series" && ls | tr '\n' ' ')"
out=("$wt"/series/0*.patch); out=("${out[@]/*0000-cover-letter.patch}"); mapfile -t out < <(printf '%s\n' "${out[@]}" | grep .)
echo "== checkpatch --strict"
./scripts/checkpatch.pl --strict --show-types "${out[@]}" > "$logs/checkpatch.log" 2>&1 || true
grep -E '^(total|ERROR|WARNING|CHECK)|has no obvious style|has style problems' "$logs/checkpatch.log" || true
echo "== get_maintainer.pl"
./scripts/get_maintainer.pl --no-rolestats "${out[@]}" | grep -v '^$' | tee "$logs/get_maintainer.txt" >/dev/null
./scripts/get_maintainer.pl "${out[@]}" > "$logs/get_maintainer-roles.txt" 2>&1
cat "$logs/get_maintainer-roles.txt"
rm -rf "$sdir/series"; cp -r "$wt/series" "$sdir/series"
echo "base-commit $resolved"
