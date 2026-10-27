# Demo chat rebuild, 2026-10-26

The pipe-UI-on-a-transport chat (piPE's web app in an iframe, `trashcan-transport.js` running the agent loop in the
browser) replaced by `tools/webchat/server.py` (conversations and the agent loop on the server, event log + SSE) and
a single page in the pipe 3.1 look (`tools/webchat/static`). Design and operation: `tools/webchat/README.md`.

The live chat (127.0.0.1:8080 on incite, nginx https://incite.tdf.sh) was not touched: the new server ran beside it
on 127.0.0.1:8090 (`deploy.sh incite 8090 trashcan-exp/webchat2`), reached through `ssh -N -L 18090:127.0.0.1:8090
incite`. The served model stayed Qwen3.8-27B (four Macs, 204,800 context, MTP) throughout; no switch of the real
cluster was made (it would have taken the user's live chat down for minutes, twice).

## Unit tests (local, Python 3.14, `python3 tools/webchat/test_server.py -v`): 21 pass

`unit-tests.txt`. Against a fake `trashcan serve` that streams the engine's event shapes (`engine/src/responses.rs`):
simple answer, persistence and reload by a new server, thinking off -> `effort: none`, history replay (reasoning,
function_call, function_call_output; no ids/statuses), parallel calls and narration, bash exit codes, last round
`tool_choice: none`, Stop mid-stream (the fake engine sees the disconnect), Stop during a tool, prompt overflow 400,
incomplete at the context limit, output-limit warning, server down, `response.failed` and truncated streams, one
turn at a time, a restart marking a running turn interrupted, the event log rebuilding the snapshot, the HTTP API
(create/list/rename/delete, reconnect after N without repeats), two readers of one live turn plus a late reader
(snapshot) and a check that step+delta events never double-count text (a bug found in the browser, below),
static files with no-cache, switch phases/blocker/rejection with a fake `cluster.sh`, and a failed switch.

## Browser tests (Chrome via chrome-devtools; real engine on incite unless noted)

| check | how | result |
|---|---|---|
| first load, new conversation | desktop 1280x820, "Say hello in five words." thinking off | answer in 4.2 s |
| multi-turn with tools, thinking medium | same conversation: "What is the population of Canberra?" | web_search, thinking, structured answer with sources, 19 s (`screenshots/01`) |
| bash tool + code answer, thinking low | iPhone emulation 390x844 touch, uptime/free + Python snippet | 3 bash calls, highlighted code, 90 s (`02`, `03`: label/copy overlap and wrapped bash columns found here, fixed) |
| Stop mid-answer | 500-word essay, Stop at 301 characters | text frozen at 301, "Stopped.", no connection left to :8000, engine answered /v1/models in 12 ms (idle) |
| reload mid-answer | essay, `location.reload()` at 159 characters | after reload 1,077 characters and still streaming; finished normally |
| connection killed mid-answer, two tabs | second tab on the same conversation; `sudo ss -K state established "( sport = :8090 )"` at 19:44:35.8 while streaming | both tabs reconnected 19:44:37, turn ended 19:44:49; both pages' answer text byte-identical to the stored answer (2,815 characters) |
| parallel tool calls, thinking high | new conversation: "Which has more people today, Oslo or Helsinki? Search for both at once." | two web_search calls in one round, answer in 34 s (`07`, `08`) |
| model server down | throwaway instance on 8091 with `--upstream` at a closed port | "The model server is not running. Pick a model to start it." with "send again", banner "No model is running." (`04`: composer width bug found, fixed) |
| model picker on mobile | 390x844 and 390x420 (keyboard-up height), touch, iPhone UA | bottom sheet with both models, current marked, fits the short viewport; confirm dialog before a switch (`06`: stray scrollbar found, fixed) |
| switching progress | local server with a fake `cluster.sh` printing the real script's lines | banner and sheet show Waiting for the Macs -> Waiting for invite (queued: blocked by #312 ...) -> Checking the Macs -> Loading the model -> Checking the model, with a timer; Send disabled; chip shows the new model when ready (`05`) |
| delete / switch conversations | local server: new, open another, delete a non-current one; delete the open one through the API (as another device) | list and selection update; the open page shows "This conversation was deleted." and an empty chat |
| drawer, short viewport, composer | 390x844 and 390x420 touch | drawer with scrim; composer stays at the bottom, 16 px field, thinking chips on one row |
| console | all pages | no errors (one TDZ error at first load during development, fixed) |

Bugs found and fixed during testing: events held references to live step objects, so a step serialized after a
delta repeated that text ("Based Based on"); fixed by freezing events at emit time, with a regression test. Others: a
startup order error, an empty "working" box beside a streaming answer, the code label/copy button over the first
line, wrapped bash output, the composer shrinking on touch, a sideways scrollbar in dialogs, the model status staying
"no model" for 30 s after the engine came back.

Prefix reuse observed: in the Canberra turn the second round reported 585 of 1,232 prompt tokens cached
(`incite-8090/*.json`, `usage`).

`incite-8090/` holds the test conversations and the server log of the last run; `screenshots/` the screenshots named
above (numbered; "before-fix" ones show the bug that was then fixed).
