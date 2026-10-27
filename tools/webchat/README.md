# tools/webchat: the trashcan demo chat

A chat page for the model the cluster serves, with web search, page reading and a shell on incite. It is what
https://incite.tdf.sh serves (nginx on incite, tailnet only, proxies to this server on 127.0.0.1:8080).

| file | what |
|---|---|
| `server.py` | the whole backend, Python standard library only: conversations, the agent loop, tools, model switching, the page's files |
| `static/` | the page: `index.html`, `app.js`, `webchat.css`; `pipe/` holds piPE's stylesheets, fonts, markdown renderer and vendored highlighting (below) |
| `cluster.sh` | starts or switches the served model under gpuq leases (`up 6-flashnext` / `up 4-qwen27b`) |
| `deploy.sh` | installs and (re)starts the server on a host and port |
| `test_server.py` | unit tests against a fake `trashcan serve` |

## How it works

```
browser ──HTTP + SSE──▶ server.py ──POST /v1/responses (SSE)──▶ trashcan serve :8000
                          │  conversations: DATA/<id>.json
                          ├─ tools: web_search (DuckDuckGo lite), fetch_page, bash (on this machine)
                          └─ cluster.sh up RECIPE (model switch)
```

- **Turns run on the server.** `POST /api/sessions/ID/messages` starts a turn in a server thread: a streamed Responses
  round, the tools the model called (one round's calls run side by side), the next round with their outputs, until
  the model answers (at most 10 rounds; the last one is sent with `tool_choice: "none"`). Closing the page, locking
  the phone or losing the network does not affect it.
- **Events.** Every change to a conversation is an event with a sequence number: `turn` (a turn started, changed
  shape or ended), `step` (a thinking block, a text block or a tool call appeared or changed), `delta` (text appended
  to a step), `session` (title, context used). The page keeps one `GET /api/events?session=ID&after=N` stream open
  (server-sent events, `id:` = sequence number): it starts with a `snapshot` of the conversation, then the events
  after it. A reconnect (EventSource's own, with `Last-Event-ID`, or the page's watchdog after 40 s of silence or on
  returning to the foreground) continues from the last event. The server keeps the last 4,000 events per
  conversation in memory; older positions get a fresh snapshot. The same stream carries `status` (served model,
  switch progress, which conversations are answering) and a `ping` every 15 s. Any number of pages, on any devices,
  can follow the same turn.
- **Stop** (`POST /api/sessions/ID/stop`) shuts the socket to the engine, which stops at its next write (checked: the
  engine answers `/v1/models` within milliseconds afterwards, so it is idle); a running bash command's process group
  is killed. What was written so far stays, marked "Stopped."
- **What the model sees.** Each turn stores the Responses items it produced (`items` in the JSON, not sent to the
  page): the user message, reasoning summaries, messages, function calls and their outputs, replayed verbatim (ids
  and statuses dropped) in later rounds and turns. Qwen3.8's template keeps earlier reasoning, so replaying it lets
  `trashcan serve` reuse its cached prefix. A turn that failed before the model said anything is left out of later
  requests (the page offers "send again"). A stopped turn keeps its partial answer; calls cut off by Stop get the
  output "Stopped by the user." so the history stays valid.
- **Errors.** No model server: "The model server is not running. Pick a model to start it." Prompt longer than the
  context (HTTP 400 `prompt has N tokens; context is C`) or a response that ends `incomplete` at the context limit:
  "This conversation is too long for MODEL (C tokens). Start a new conversation." Output limit (16,384 tokens, or a
  quarter of the context) with an answer: a warning under it. `response.failed`, or a stream that ends without
  `response.completed`: the reason, with "send again". A turn running when the server stopped is marked "The chat
  server restarted during this reply." on the next start.
- **The model.** Requests carry no `model` field; the engine names what it serves in `response.created`, and the
  server checks `/v1/models` at start and every 30 s while idle (5 s while it finds nothing). The engine handles one
  request at a time, so a check that times out means busy and changes nothing. This server sends one round at a
  time; a second conversation's round waits ("waiting for another conversation to finish").
- **Switching** (`POST /api/switch {"recipe": "6-flashnext" | "4-qwen27b"}`) runs `cluster.sh up RECIPE` and
  reports its phases from the output: stopping the current model, waiting for the Macs (with the blocking gpuq
  ticket), checking the Macs, loading, checking the model; then it confirms the engine serves the expected model id.
  It is refused while a reply runs, and messages are refused while it runs (the page disables Send and shows the
  progress above the composer). Failures show cluster.sh's last lines.

The tools, the system instructions (factual question: 1-3 `web_search`, `fetch_page` when needed, then the direct
answer first, details in sections or bullets, sources last) and the two models (Qwen3.8-Flash-Next, 114,688 tokens, six
Macs; Qwen3.8-27B, 204,800 tokens, four Macs with MTP) are at the top of `server.py`.

### HTTP API

| request | answer |
|---|---|
| `GET /` | the page (no redirect; old `/?transport=…` links still work), static files with `Cache-Control: no-cache` |
| `GET /api/state` | models, thinking levels, status |
| `GET /api/sessions` | conversations, newest first: id, title, turns, running, last status |
| `POST /api/sessions {text?, thinking?}` | new conversation (starting a turn when `text` is given) |
| `GET /api/sessions/ID` | the conversation (what the snapshot event carries) |
| `PATCH /api/sessions/ID {title}` | rename (also `POST /api/sessions/ID/rename`) |
| `DELETE /api/sessions/ID` | stop its turn and delete it; followers get a `deleted` event |
| `POST /api/sessions/ID/messages {text, thinking}` | start a turn; 409 while one runs or a model starts |
| `POST /api/sessions/ID/stop` | stop the running turn |
| `GET /api/events?session=ID&after=N` | the event stream (`session` optional: status only) |
| `POST /api/switch {recipe}` | switch the served model |
| `/v1/*` | passed through to the engine, for API clients |

`thinking` is `off`, `low`, `medium` or `high`: `reasoning.effort` `none` or the level, with `summary: "auto"` so the
thinking is shown.

## The page

piPE's conversation UI in its "pipe 3.1" (win31) skin, rebuilt without piPE's host, RPC layer or features this
backend does not have. From piPE (`~/dev/piPE/web/apps/conversation`, commit 31c21d3), copied unchanged into
`static/pipe/`: `tokens.css`, `app.css`, `win31.css`, `md.js` (markdown-it), `highlight.js`, `ansi.js`, `fonts/` and
`vendor/` (markdown-it and plugins, highlight.js, KaTeX; see `vendor/VENDOR.md`; Mermaid is left out, so a
`mermaid` fence shows its source). `webchat.css` holds this page's changes; `app.js` is new and uses piPE's DOM and
class names, so the skin applies as it is.

Controls: conversations (new, open, delete; the title is the first message), the model picker (both models, what
each needs, switch progress), thinking level (off/low/medium/high, kept per browser), Send/Stop (Esc stops on a
keyboard), the context meter (tokens the last request used, of the served model's context), copy buttons on
messages and code. Each turn shows the question, a "working" section (open by default) with thinking blocks, the
model's notes between tool calls and tool cards (bash: command and output; web_search: the query and the results
list; fetch_page: the URL and the text read), then the answer.

On a phone: the conversations pane is a drawer, the model picker a bottom sheet, controls are finger-sized, text
fields are 16 px (no zoom on focus), the layout follows the visual viewport so the composer stays above the keyboard,
safe areas are respected, the return key writes a new line (the send button sends), and the transcript follows the
answer only while it is scrolled to the bottom ("latest" brings it back).

## Operation

```sh
tools/webchat/deploy.sh incite 8080                      # the live chat: ~/trashcan-exp/webchat, sessions/ in it
tools/webchat/deploy.sh incite 8090 trashcan-exp/webchat2  # a test instance beside it
ssh -N -L 18090:127.0.0.1:8090 incite                     # then http://localhost:18090
python3 tools/webchat/test_server.py -v                   # unit tests, locally
```

`deploy.sh` copies the files, stops the instance it started earlier on that port (pid file `server-PORT.pid`),
refuses to start if anything else listens there, starts `server.py --port PORT --data DIR/sessions` with `setsid
nohup`, and checks `/api/state`. Logs: `~/DIR/server-PORT.log`. Other options: `--upstream URL` (default
`http://127.0.0.1:8000`), `--workdir DIR` (where bash runs, default `~`), `--cluster PATH` (tests use a fake).
Bash runs as the server's user on incite with a 60 s limit; the server binds to loopback, and nginx admits tailnet
addresses only.

## Known limitations

- One engine, one request at a time: a turn in this chat and any other client of `:8000` (the CLI, another chat
  server) queue behind each other.
- Prefix reuse across rounds is partial in practice (one measured tool round reused 585 of 1,232 prompt tokens), so
  the replayed reasoning does not always match the engine's cached tokens exactly; worth a look in the engine's
  rendering of reasoning summaries.
- No editing or regenerating of earlier messages, no attachments, no renaming in the page (the API has it).
- Search is DuckDuckGo's lite page scraped as HTML; if it changes or rate-limits, `web_search` returns an error the
  model sees.
