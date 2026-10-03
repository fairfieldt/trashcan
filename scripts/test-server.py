#!/usr/bin/env python3
"""Exercise a running Responses server over HTTP, without external Python packages."""
import argparse
import http.client
import json
from urllib.parse import urlsplit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--skip-tools", action="store_true")
    parser.add_argument("--sdk", action="store_true", help="Also validate every response/event against the installed OpenAI SDK schemas")
    args = parser.parse_args()
    url = urlsplit(args.url)
    connection_type = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
    connection = connection_type(url.hostname, url.port, timeout=300)
    if args.sdk:
        from openai.types.responses import Response as SDKResponse, ResponseStreamEvent
        from pydantic import TypeAdapter
        event_schema = TypeAdapter(ResponseStreamEvent)

    def send(method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        connection.request(method, path, data, {"Content-Type": "application/json"})
        response = connection.getresponse()
        payload = response.read().decode()
        if response.getheader("Content-Type", "").startswith("text/event-stream"):
            events = []
            for block in payload.strip().split("\n\n"):
                event, data = block.split("\ndata: ", 1)
                value = json.loads(data)
                assert event == "event: " + value["type"]
                assert value["sequence_number"] == len(events)
                events.append(value)
            return response.status, events
        return response.status, json.loads(payload)

    status, health = send("GET", "/health")
    assert status == 200 and health["status"] == "ok"
    status, models = send("GET", "/v1/models")
    assert status == 200 and models["object"] == "list"
    model = models["data"][0]["id"]

    def create(**params):
        status, response = send("POST", "/v1/responses", {"model": model, "temperature": 0, "store": False, **params})
        assert status == 200, response
        if params.get("stream"):
            events = response
            if args.sdk:
                for event in events:
                    event_schema.validate_python(event)
            assert events[0]["type"] == "response.created"
            assert events[-1]["type"] in ("response.completed", "response.incomplete")
            response = events[-1]["response"]
            done = [e["item"] for e in events if e["type"] == "response.output_item.done"]
            assert done == response["output"], (done, response)
            deltas = "".join(e["delta"] for e in events if e["type"] == "response.output_text.delta")
            assert deltas == text(response)
        assert response["usage"]["total_tokens"] == response["usage"]["input_tokens"] + response["usage"]["output_tokens"]
        if args.sdk:
            SDKResponse.model_validate(response)
        return response

    def text(response):
        return "".join(part["text"] for item in response["output"] if item["type"] == "message" for part in item["content"])

    prompt = "Reply with exactly these two words: Hello there"
    plain = create(input=prompt, max_output_tokens=64)
    streamed = create(input=prompt, max_output_tokens=64, stream=True)
    assert plain["status"] == streamed["status"] == "completed"
    assert text(plain) and text(plain) == text(streamed)
    # Reuses the connection after chunked SSE, exercising the real HTTP framing.
    assert send("GET", "/v1/models")[0] == 200
    print("PASS: discovery, JSON, SSE, deterministic parity and persistent HTTP", flush=True)

    # Switch candidate counts in a resident engine, then reuse both recorded graphs.
    sampled = create(input="Tell a short story about a lost robot.", temperature=1,
                     seed=7, max_output_tokens=32)
    greedy_again = create(input=prompt, max_output_tokens=64)
    sampled_again = create(input="Tell a short story about a lost robot.", temperature=1,
                           seed=7, max_output_tokens=32, stream=True)
    assert text(greedy_again) == text(plain)
    assert text(sampled) and text(sampled) == text(sampled_again)
    print("PASS: alternating greedy/stochastic graphs and seeded parity", flush=True)

    limited = create(input="Explain how photosynthesis works in detail.", max_output_tokens=1, stream=True)
    assert limited["status"] == "incomplete"
    assert limited["incomplete_details"]["reason"] == "max_output_tokens"
    assert limited["usage"]["output_tokens"] == 1
    thinking = create(input="What is 17 times 19?", max_output_tokens=96,
                      reasoning={"effort": "high", "summary": "auto"}, stream=True)
    assert any(item["type"] == "reasoning" and item["summary"] for item in thinking["output"])
    assert "<|channel>" not in text(thinking) and "<channel|>" not in text(thinking)
    print("PASS: token-limit terminal event and separate reasoning", flush=True)

    for patch in ({"previous_response_id": "resp_missing"}, {"tool_choice": "required"},
                  {"input": [{"type": "function_call_output", "call_id": "missing", "output": "result"}]},
                  {"input": [{"role": "user", "content": [{"type": "input_image", "image_url": "data:..."}]}]},
                  {"input": "word " * 20000}, {"max_output_tokens": 0}, {"temperature": -1}):
        status, response = send("POST", "/v1/responses", {"model": model, "input": "Hi", **patch})
        assert status == 400 and "error" in response, response
    assert send("GET", "/unknown")[0] == 404
    print("PASS: invalid requests and prompt overflow", flush=True)

    if not args.skip_tools:
        history = [{"role": "user", "content": "Call the weather tool with city set to Paris. Do not answer without calling it."}]
        tools = [{"type": "function", "name": "weather", "description": "Get the weather for a city.",
                  "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}]
        first = create(input=history, tools=tools, max_output_tokens=128, stream=True)
        assert first["status"] == "completed", first
        calls = [item for item in first["output"] if item["type"] == "function_call"]
        assert calls and all(item["name"] == "weather" for item in calls), first
        assert all(json.loads(item["arguments"])["city"].lower() == "paris" for item in calls), calls
        history.extend(first["output"])
        history.extend({"type": "function_call_output", "call_id": item["call_id"],
                        "output": '{"city":"Paris","temperature_c":18,"condition":"sunny"}'} for item in calls)
        final = create(input=history, tools=tools, max_output_tokens=128, stream=True)
        assert final["status"] == "completed" and text(final), final
        assert "18" in text(final), final
        print("PASS: native function call and full-history tool continuation", flush=True)

        namespaced = create(input="Call weather.lookup with city Paris.",
                            tools=[{"type": "namespace", "name": "weather", "tools": [
                                {**tools[0], "name": "lookup"}]}], parallel_tool_calls=False,
                            max_output_tokens=128, stream=True)
        calls = [item for item in namespaced["output"] if item["type"] == "function_call"]
        assert namespaced["status"] == "completed" and len(calls) == 1, namespaced
        assert calls[0]["name"] == "lookup" and calls[0]["namespace"] == "weather", calls

        custom = create(input="Call the echo tool with exactly the text Hello world.",
                        tools=[{"type": "custom", "name": "echo", "description": "Echo the supplied text."}],
                        max_output_tokens=128, stream=True)
        calls = [item for item in custom["output"] if item["type"] == "custom_tool_call"]
        assert custom["status"] == "completed" and calls and calls[0]["name"] == "echo", custom
        assert "Hello world" in calls[0]["input"], calls

        search = {"type": "tool_search", "execution": "client", "description": "Find available tools.",
                  "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}
        searched = create(input="Call tool_search with query weather to find weather tools.",
                          tools=[search], max_output_tokens=128, stream=True)
        calls = [item for item in searched["output"] if item["type"] == "tool_search_call"]
        assert searched["status"] == "completed" and calls and calls[0]["execution"] == "client", searched
        assert isinstance(calls[0]["arguments"], dict), calls
        print("PASS: namespaces, single-call control, custom tools and client tool search", flush=True)

    connection.close()
    print("Responses server checks passed", flush=True)


if __name__ == "__main__":
    main()
