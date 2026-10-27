# web/apps/conversation/vendor — third-party code, vendored

Nothing here is written by this project. Each file is upstream's, byte
for byte, from the release named below; `tools/vendor-web.py` is the
manifest and re-fetches and verifies every one of them against the
sha512 the npm registry records for that release:

    tools/vendor-web.py --check

Every pin is at least 7 days old at the time it was chosen. The browser
UI is served from the pipe's own zip store and the browser build runs
from the same tree, so there is no npm at build time and no CDN at run
time — a vendored file is the only kind of dependency that works in
both.

The tree sits inside the app that imports it, not at `web/vendor/`,
because `web/htmlboot/boot.html` materializes an app by rewriting the
references that resolve to files *the app ships* into `blob:` URLs —
its own subtree, and nothing else, so a capsule app cannot reach into
the pipe's store. A dependency outside the app root would work under
`--serve` and fail to resolve in a `.pipe.html`.

| package | version | published | license |
|---|---|---|---|
| `markdown-it` | 15.0.2 | 2026-09-11 | MIT |
| `markdown-it-footnote` | 4.0.0 | 2023-12-06 | MIT |
| `markdown-it-deflist` | 4.0.0 | 2026-07-27 | MIT |
| `markdown-it-mark` | 4.0.0 | 2023-12-05 | MIT |
| `markdown-it-task-lists` | 2.1.1 | 2018-03-06 | ISC |
| `@highlightjs/cdn-assets` | 11.12.0 | 2026-08-12 | BSD-3-Clause |
| `highlight.js` | 11.12.0 | 2026-08-12 | BSD-3-Clause |
| `katex` | 0.18.7 | 2026-09-06 | MIT |
| `mermaid` | 11.17.2 | 2026-08-25 | MIT |

Why each is here:

- **markdown-it 15.0.2** — Security releases 15.0.1/15.0.2 fixed quadratic-complexity hangs in fuzzy-link replacement, scheme backscan and smartquotes — reachable from model output, which is exactly what this renderer eats.
- **markdown-it-footnote 4.0.0** — GFM footnotes: [^1] refs and a definitions section.
- **markdown-it-deflist 4.0.0** — Definition lists. 4.0.0 is the release that dropped StateBlock#ddIndent, which markdown-it 15 removed; 3.x silently mis-nests against 15.
- **markdown-it-mark 4.0.0** — ==highlight== spans.
- **markdown-it-task-lists 2.1.1** — - [ ] / - [x] checkboxes. CommonJS only (no ESM entry), so this one loads as a classic script and hands its export to window.markdownitTaskLists; md.js reads it from there.
- **@highlightjs/cdn-assets 11.12.0** — The browser bundle highlight.js publishes for exactly this use: one ESM file, the 36 common grammars (bash, c, cpp, css, diff, go, java, javascript, json, markdown, python, rust, sql, typescript, xml, yaml, …).
- **highlight.js 11.12.0** — Grammars the common bundle leaves out but agents write anyway. Same version as the bundle above so the two cannot drift apart; unminified ESM because only cdn-assets is minified and those files are classic-script shaped.
- **katex 0.18.7** — TeX math. Loaded on demand: katex.min.js + css + the 20 woff2 faces (296 KB) only arrive when a message actually contains math.
- **mermaid 11.17.2** — Diagrams. 11.x, not 12.0.0: 12 raises the browser floor to ES2024 / Safari 17.4+ for the whole page, and its single-file build inlines ELK (~500 KB gzipped) and changes the default layout of every diagram type. 11.17.2 is the mature line and its bundle is 2 MB smaller. Loaded on demand.

## What is *not* vendored, and why

- **A markdown engine of our own.** `md.js` used to be a hand-rolled
  renderer; it is now a thin configuration of markdown-it. Nested lists,
  reference links, setext headings, autolinks and emphasis nesting are
  CommonMark rules with a lot of sharp edges, and the renderer eats
  untrusted model output.
- **An ANSI terminal renderer** (`web/apps/conversation/ansi.js`) — the
  subset worth supporting is SGR colour, bold/dim/italic/underline and
  carriage-return overwrite, which is a hundred lines and has to be
  safe against escape sequences that are not SGR.
- **A diff renderer** (`web/apps/conversation/diff.js`) — an LCS over
  lines, ~60 lines, and it only ever sees the `oldText`/`newText` the
  edit tool was already given.
- **A highlight.js theme.** The stylesheets upstream ships are fixed
  palettes; this UI has two skins and its own tokens, so the mapping
  from the `hljs-*` classes it emits to `--*` variables is written out
  in `app.css` instead (the names are hljs 11's: `attr`, `title.class`,
  `built_in`, …).
