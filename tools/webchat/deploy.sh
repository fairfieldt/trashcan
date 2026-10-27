#!/bin/bash
# Install the trashcan demo chat on HOST (the machine running `trashcan serve`, e.g. incite) and (re)start it on
# 127.0.0.1:PORT: server.py, cluster.sh and static/ go to ~/DIR, conversations live in ~/DIR/sessions, the log is
# ~/DIR/server-PORT.log. Model switching needs the launcher in ~/trashcan-exp/cluster on HOST (see cluster.sh).
#
#   tools/webchat/deploy.sh HOST [PORT] [DIR]     defaults: 8080, trashcan-exp/webchat
#
# It restarts only the instance it started itself on that port (pid file ~/DIR/server-PORT.pid) and refuses to start
# when something else listens there.
set -euo pipefail
host=${1:?usage: $0 HOST [PORT] [DIR]}
port=${2:-8080}
dir=${3:-trashcan-exp/webchat}
here=$(cd "$(dirname "$0")" && pwd)
python3 -m py_compile "$here/server.py"
ssh "$host" "mkdir -p ~/$dir/.new && rm -rf ~/$dir/.new/*"
COPYFILE_DISABLE=1 tar --no-mac-metadata --no-xattrs -C "$here" -c server.py cluster.sh static | ssh "$host" "tar --warning=no-unknown-keyword -C ~/$dir/.new -x"
ssh "$host" "DIR=\$HOME/$dir PORT=$port bash -s" <<'EOF'
set -euo pipefail
cd "$DIR"
pidfile=server-$PORT.pid
if [ -f "$pidfile" ] && grep -q 'server[.]py' "/proc/$(cat "$pidfile")/cmdline" 2>/dev/null; then
    kill "$(cat "$pidfile")"
    for _ in $(seq 50); do kill -0 "$(cat "$pidfile")" 2>/dev/null || break; sleep 0.1; done
fi
rm -f "$pidfile"
if ss -Hltn "sport = :$PORT" | grep -q .; then
    echo "port $PORT is in use by another program (stop it first):" >&2
    ss -Hltnp "sport = :$PORT" >&2
    exit 1
fi
rm -rf static.old; [ -d static ] && mv static static.old
mv .new/static static; mv .new/server.py .new/cluster.sh .; rm -rf .new static.old
setsid nohup python3 "$DIR/server.py" --port "$PORT" --data "$DIR/sessions" > "server-$PORT.log" 2>&1 < /dev/null &
echo $! > "$pidfile"
sleep 1
curl -fsS "http://127.0.0.1:$PORT/api/state" > /dev/null && echo "webchat running on 127.0.0.1:$PORT (pid $(cat "$pidfile"))" || { cat "server-$PORT.log"; exit 1; }
EOF
