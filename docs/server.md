# Responses server

Run the same engine as an optional HTTP server:

```sh
target/release/trashcan serve ~/models/gemma-4-12b-qat-gguf/gemma-4-12b-it-qat-q4_0.gguf \
  --host 127.0.0.1 --port 8000 --context 8192 --batch 512
```

`server` is an alias for `serve`. The model stays loaded and requests run serially through one
resident engine. A qwen35 GGUF (Qwen3.8-27B, Qwen3.5-9B) selects the Qwen pipeline, with its own chain,
MTP and prefix-reuse options; see [qwen.md](qwen.md#serving). A qwen4exp GGUF (Qwen3.8-Flash-Next) serves over its
machine chain; see [qwen4.md](qwen4.md#serving). The rest of this page describes Gemma. It reuses an exact token prefix from the last successful request when that
prefix's sliding-window KV is still resident. Before the first ring wrap, all processed prefixes
remain available. After wrap, the furthest KV write must be no more than `--batch` positions
beyond the reused prefix; rejected speculative writes count too. Older or changed prefixes safely
fall back to a full replay. This is a single resident sequence, not durable conversation storage;
clients still send the full history.

The default batch capacity is 512. It gives long prompts larger prefill chunks and retains enough
sliding history for a repeated 256-token reply at the default 8K context. Short prompts use their
actual length. `--batch 128` reduces buffer memory when needed, with a shorter reuse horizon.

Identical prompts also reuse their saved logits when the target candidate count matches and
the drafter is disabled. Other cache hits recompute the changed suffix and at least the final
prompt token. Replaying a suffix with different batch boundaries may produce the same small
floating-point differences as ordinary serial versus chunked prefill. Generation failures and
streaming disconnects invalidate the sequence. Response usage reports the reused positions in
`input_tokens_details.cached_tokens`; `--no-prefix-cache` disables reuse for cold comparisons.

`--model-id` defaults to `gemma-4-12b`. `--thinking`, `--max-tokens`, `--temperature`, `--top-k`,
`--top-p` and `--seed` set request defaults. Requests override them with `reasoning.effort`,
`max_output_tokens`, `temperature`, `top_k`, `top_p` and `seed`. `top_k` is limited to 1–64.
Temperature zero selects greedy sampling. `--draft FILE` enables the same matched assistant as
chat, with `--draft-tokens` (default 3). `--cpu-reduce` selects the CPU relay path.
The target and assistant each rank one candidate for greedy requests and 64 for stochastic requests.
Requests can alternate sampling modes. The target keeps up to four recently used execution graphs;
older graphs are rebuilt as needed without clearing the retained token/KV prefix.

| Endpoint | Behavior |
| --- | --- |
| `GET /health` | Readiness after the model has loaded |
| `GET /v1/models` | Loaded model ID |
| `POST /v1/responses` | JSON response or SSE when `stream: true` |

`/models` and `/responses` also work without the `/v1` prefix. `--cors` enables browser requests.
The server binds to loopback by default and has no built-in authentication or TLS. Use
`--host 0.0.0.0` to expose it on other interfaces.

```sh
curl http://127.0.0.1:8000/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma-4-12b","input":"Explain Redis streams briefly.","temperature":0,"stream":true}'
```

With the OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")
response = client.responses.create(
    model="gemma-4-12b",
    input="Hello there",
    store=False,
)
print(response.output_text)
```

## Tools and continuations

`input` accepts a string or Responses message items. `instructions` supplies the system prompt.
Message content can be a string or an array of `input_text`/`output_text` blocks. System and
developer messages are combined into the template's system turn.

Function tools use the Responses layout: `{"type":"function","name":"...","description":"...",
"parameters":{"type":"object","properties":{...}}}`. Namespace tools are flattened for
Gemma's prompt and returned with the original `name` and `namespace`. Custom tools expose their
free-form input to Gemma as an `input` string parameter and return `custom_tool_call` items.
Tool schemas guide the model; JSON Schema and custom grammar constraints are not enforced.
Only `tool_choice: "auto"` and `"none"` are supported.
`parallel_tool_calls: false` stops generation after the first complete tool call.

Client-executed `tool_search` is also supported. It emits a `tool_search_call` with an argument
object; replay a matching `tool_search_output` containing the discovered `tools` to make those
definitions available on the next request. Server-executed tool search is unsupported.

Execute tools in the client and resend the full conversation, including `response.output` items
and the corresponding `function_call_output` or `custom_tool_call_output` items:

```python
history = [{"role": "user", "content": "What is the weather in Paris?"}]
tools = [{"type": "function", "name": "weather", "description": "Get local weather",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                         "required": ["city"]}}]
response = client.responses.create(model="gemma-4-12b", input=history, tools=tools, store=False)
history.extend(response.output)
for item in response.output:
    if item.type == "function_call":
        # Replace this fixture result with the client's tool execution.
        history.append({"type": "function_call_output", "call_id": item.call_id,
                        "output": '{"temperature_c": 18}'})
response = client.responses.create(model="gemma-4-12b", input=history, tools=tools, store=False)
print(response.output_text)
```

Tool result IDs must match preceding calls. Incomplete/failed output items, orphan results and
unanswered calls are rejected. Native tool argument syntax is converted to JSON, including nested
objects, arrays and Gemma string delimiters. A truncated tool call is never emitted as a completed call.

`reasoning.effort: "none"` disables thinking; other accepted effort values enable it. They do not
select different compute budgets. Reasoning is emitted separately only when `reasoning.summary`
is `"auto"`, `"concise"` or `"detailed"`. These modes expose the model's thought channel;
they do not run a separate summarizer. Returned summary items can be replayed with tool history.

## Codex CLI

Add a provider to the Codex configuration:

```toml
[model_providers.trashcan]
name = "Trashcan"
base_url = "http://127.0.0.1:8000/v1"
wire_api = "responses"
stream_idle_timeout_ms = 1000000
```

```sh
codex --model gemma-4-12b -c model_provider=trashcan
```

## Limits

This is a text Responses endpoint. Images, audio, files, hosted tools, Chat Completions,
WebSockets, automatic truncation, structured output constraints, background generation and
durable storage are unsupported. Non-null `previous_response_id`/`conversation` and `store: true`
return an error; replay the full input instead. HTTP bodies are limited to 8 MiB. Prompt overflow
returns HTTP 400. Exhausting either the output budget or remaining context emits
`response.incomplete` with `max_output_tokens`; generation failures emit `response.failed`.
Streaming disconnects stop generation at the next write.

Run `python3 scripts/test-server.py --url http://127.0.0.1:8000` to check a running server's
JSON/SSE output, token limits, reasoning, request validation and a real tool-result continuation.
Add `--sdk` when the OpenAI Python SDK is installed to validate every response and event against
its schemas as well.

Run `python3 scripts/bench-prefix-cache.py --url http://127.0.0.1:8000 --batch 128 --output prefix.json`
against a server started with `--batch 128` for repeated prompts, cache-retention boundaries,
sampler changes, long histories and tool continuations. Add `--cold` when the server uses
`--no-prefix-cache`, or `--draft` when it uses an assistant. The report includes request bodies,
latency, output, cache usage and correctness assertions for reproduction. Run the cold server
first, then pass `--compare-cold prefix-cold.json` to the cached run to compare every
corresponding response and latency. The report marks any output differences explicitly, including
possible rounding changes when the drafter replays its final hidden row; `passed` records the
protocol and cache-boundary checks, while `repeat_output_exact` and `cold_comparison_exact`
record exact output equivalence separately.

The protocol follows [OpenAI's Responses streaming events](https://developers.openai.com/api/docs/guides/streaming-responses).
