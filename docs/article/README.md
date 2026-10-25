# Article prototype: "How we deploy 180B-parameter frontier models on Mac Pros"

A parody of Cerebras's
[How we deploy trillion-parameter frontier models on Cerebras](https://www.cerebras.ai/blog/how-we-deploy-trillion-parameter-frontier-models-on-cerebras)
about the six-Mac Flash-Next chain: the same sections, beats and calculations in order, with our hardware and
numbers swapped in. Our own wording and figures; no text or assets copied from the original post.

- `index.html`: the article. `notes.html`: source notes for every number and the supporting charts.
- `charts.js`: draws the bar charts from its `CHARTS` table; each figure also has a data table.
- No new runs were made for it. Numbers come from `docs/qwen4.md`, `docs/hardware.md`, `docs/kernel.md`,
  `docs/server.md` and `results/qwen4/*` up to 2026-10-14.

## Preview (tailnet only)

http://ms.camel-vibes.ts.net:8227/ (also http://ms:8227/). A launchd agent on the workstation serves only this
directory on 127.0.0.1:8227 (`~/Library/LaunchAgents/sh.tdf.trashcan-article.plist`, log in
`~/dev/claude-work/trashcan-article-20261024/server.log`), and `tailscale serve` proxies the tailnet port to it.

```sh
# stop
tailscale serve --http=8227 off
launchctl bootout gui/$(id -u)/sh.tdf.trashcan-article
# start
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/sh.tdf.trashcan-article.plist
tailscale serve --bg --http 8227 http://127.0.0.1:8227
```

## Literal-swap version

`cerebras-parody.sh [OUT.html]` downloads the original post and writes a copy with the trashcan hardware and numbers
swapped in (text pairs and replacement figures in the script; `--check` only reports matches). Run it yourself; the
output isn't kept in the repo or served.
