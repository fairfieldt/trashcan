#!/bin/bash
# Install the pipe web UI on a trashcan server on HOST (the machine running `trashcan serve`, e.g. incite) and
# (re)start server.py there on 127.0.0.1:8080: piPE's web/ (from PIPE_WEB, a piPE checkout's web directory, without
# the in-browser model runtime) plus trashcan-transport.js in ~/trashcan-exp/webchat/web, server.py and cluster.sh next
# to it. Model switching needs the launcher in ~/trashcan-exp/cluster on HOST (deploy/ from trashcan main and an
# inventory, cluster.toml) and SSH from HOST to every node.
#
#   tools/webchat/deploy.sh HOST [PIPE_WEB]      default PIPE_WEB: ~/dev/piPE/web
#
# Then, on HOST: http://127.0.0.1:8080/?transport=/trashcan-transport.js&ui=win31&working=open
set -euo pipefail
host=${1:?usage: $0 HOST [PIPE_WEB]}
web=${2:-$HOME/dev/piPE/web}
here=$(cd "$(dirname "$0")" && pwd)
[ -f "$web/host.js" ] || { echo "no piPE web directory at $web" >&2; exit 1; }
grep -q "params.get('transport')" "$web/host.js" || { echo "$web/host.js has no ?transport= hook (piPE too old)" >&2; exit 1; }
ssh "$host" 'mkdir -p ~/trashcan-exp/webchat/web.new && rm -rf ~/trashcan-exp/webchat/web.new/*'
COPYFILE_DISABLE=1 tar --no-mac-metadata -C "$web" --exclude inference -c . | ssh "$host" 'tar -C ~/trashcan-exp/webchat/web.new -x'
scp -q "$here/trashcan-transport.js" "$host:trashcan-exp/webchat/web.new/trashcan-transport.js"
scp -q "$here/server.py" "$here/cluster.sh" "$host:trashcan-exp/webchat/"
ssh "$host" 'set -e; cd ~/trashcan-exp/webchat
  rm -rf web.old; [ -d web ] && mv web web.old; mv web.new web; rm -rf web.old
  pkill -f "^python3 .*server[.]py --port 8080" || true; rm -f index.html app.js markdown.js style.css; sleep 0.5
  setsid nohup python3 ~/trashcan-exp/webchat/server.py --port 8080 > server.log 2>&1 < /dev/null &
  sleep 1; cat server.log'
echo "open on $host: http://127.0.0.1:8080/?transport=/trashcan-transport.js&ui=win31&working=open"
