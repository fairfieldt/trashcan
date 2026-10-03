#!/usr/bin/env python3
"""Validate and time resident KV reuse against a serial Responses server (stdlib only)."""
import argparse
import http.client
import json
import time
from pathlib import Path
from urllib.parse import urlsplit


def normalize(value):
    """Protocol IDs vary per request, including in replayed tool histories."""
    if isinstance(value, dict):
        return {key: normalize(item) for key, item in value.items()
                if key not in ("id", "call_id")}
    if isinstance(value, list):
        return [normalize(item) for item in value]
    return value


def signature(response):
    return normalize({key: response[key] for key in ("status", "incomplete_details", "output")})


def output_text(response):
    return "".join(part["text"] for item in response["output"] if item["type"] == "message"
                   for part in item["content"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--batch", type=int, default=128, help="Must match the server's --batch")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cold", action="store_true", help="Server uses --no-prefix-cache")
    parser.add_argument("--draft", action="store_true", help="Server uses --draft")
    parser.add_argument("--skip-tools", action="store_true")
    parser.add_argument("--compare-cold", type=Path,
                        help="Compare corresponding outputs and latencies with a --cold report")
    args = parser.parse_args()
    if not 2 <= args.batch <= 512:
        parser.error("--batch must be 2..512 for retention-boundary checks")
    url = urlsplit(args.url)
    connection_type = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
    records = []
    report = {"url": args.url, "batch": args.batch, "cold": args.cold,
              "draft": args.draft, "requests": records, "repeat_checks": [], "passed": False}
    cold = None
    if args.compare_cold:
        cold = json.loads(args.compare_cold.read_text())
        if not cold["cold"] or cold["draft"] != args.draft or cold["batch"] != args.batch:
            parser.error("--compare-cold must be a cold report with matching --draft and --batch")

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def run(name, body, stream=False):
        request = {"temperature": 0, "store": False, **body, "stream": stream}
        connection = connection_type(url.hostname, url.port, timeout=600)
        started = time.perf_counter()
        connection.request("POST", "/v1/responses", json.dumps(request).encode(),
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        first_delta = None
        events = []
        if response.getheader("Content-Type", "").startswith("text/event-stream"):
            while True:
                line = response.readline()
                if not line:
                    break
                if not line.startswith(b"data: "):
                    continue
                event = json.loads(line[6:])
                assert event["sequence_number"] == len(events), event
                events.append(event)
                if first_delta is None and event["type"].endswith(".delta"):
                    first_delta = time.perf_counter() - started
            assert events[-1]["type"] in ("response.completed", "response.incomplete"), events
            value = events[-1]["response"]
            completed = [event["item"] for event in events if event["type"] == "response.output_item.done"]
            assert completed == value["output"]
            delta_text = "".join(event["delta"] for event in events
                                 if event["type"] == "response.output_text.delta")
            assert delta_text == output_text(value)
        else:
            value = json.loads(response.read())
        elapsed = time.perf_counter() - started
        connection.close()
        assert response.status == 200, value
        usage = value["usage"]
        cached = usage["input_tokens_details"]["cached_tokens"]
        assert 0 <= cached <= usage["input_tokens"]
        assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]
        if args.cold:
            assert cached == 0, value
        # Later tool continuations append to input history in place; keep the
        # exact body sent for every earlier request in the reproducibility log.
        records.append({"name": name, "request": json.loads(json.dumps(request)), "seconds": elapsed,
                        "first_delta_seconds": first_delta, "response": value})
        save()
        print(f"{name}: {elapsed:.3f}s, cached {cached}/{usage['input_tokens']}, "
              f"output {usage['output_tokens']}", flush=True)
        return value

    def repeat(name, body, expected_hit=None, draft_hit=False):
        first = run(name + "_first", body)
        again = run(name + "_repeat", body, stream=True)
        equal = signature(first) == signature(again)
        same_count = first["usage"]["output_tokens"] == again["usage"]["output_tokens"]
        report["repeat_checks"].append({"name": name, "exact_output": equal,
                                        "same_output_tokens": same_count})
        if not args.draft or args.cold:
            assert equal and same_count, f"{name}: repeated JSON/SSE output differs"
        elif not equal or not same_count:
            # A cached drafter must replay the final hidden row. Different
            # prefill batch arithmetic can alter near ties; retain both outputs
            # explicitly instead of misreporting exact seeded parity.
            print(f"OUTPUT DIFFERENCE: {name} repeated drafter output differs; see report", flush=True)
        if expected_hit is not None and not args.cold and not args.draft:
            usage = again["usage"]
            cached = usage["input_tokens_details"]["cached_tokens"]
            assert (cached == usage["input_tokens"] if expected_hit else cached == 0), (name, usage)
        if draft_hit and args.draft and not args.cold:
            usage = again["usage"]
            assert usage["input_tokens_details"]["cached_tokens"] == usage["input_tokens"] - 1, (name, usage)
        return again

    try:
        run("warmup", {"input": "Say hello.", "max_output_tokens": 1})
        prompt = ("List all integers from 1 to 1000 in order, separated by spaces. "
                  "Print only the full list, with no introduction, summary, or abbreviations.")
        # Keep the prompt itself beyond the ring's capacity, including at the
        # largest supported batch. The 128-token fixture remains unchanged.
        note_repetitions = max(150, (1024 + args.batch) // 9 + 20)
        long_prompt = ("Reference notes:\n" + "The workshop has red doors and blue windows. " * note_repetitions
                       + "\n" + prompt)
        # The final budget token is emitted without a target forward in ordinary
        # decoding. Therefore N output tokens consume N-1 KV positions. Use a
        # prompt beyond the sliding window so these actually test ring retention.
        for budget in (args.batch, args.batch + 1, args.batch + 2):
            result = repeat(f"boundary_{budget}", {"input": long_prompt, "max_output_tokens": budget},
                            expected_hit=budget <= args.batch + 1)
            assert result["usage"]["input_tokens"] > 1024 + args.batch
            assert result["status"] == "incomplete", "Boundary prompt stopped before exhausting its budget"
            assert result["usage"]["output_tokens"] == budget

        sampled = {"input": prompt, "temperature": 1, "seed": 7, "top_k": 32,
                   "max_output_tokens": min(32, args.batch)}
        repeat("stochastic", sampled, expected_hit=True, draft_hit=True)
        repeat("greedy_again", {"input": prompt, "max_output_tokens": min(32, args.batch)},
               expected_hit=True, draft_hit=True)

        # More than 1024 prompt tokens exercises actual sliding-ring overwrite.
        long_body = {"input": long_prompt, "max_output_tokens": min(32, args.batch - 1)}
        long_response = repeat("ring_wrap", long_body, expected_hit=True, draft_hit=True)
        assert long_response["usage"]["input_tokens"] > 1024
        history = [{"role": "user", "content": long_prompt},
                   {"role": "assistant", "content": output_text(long_response)},
                   {"role": "user", "content": "What color are the workshop doors? Reply in one word."}]
        follow = run("long_history", {"input": history, "max_output_tokens": 32}, stream=True)
        assert "red" in output_text(follow).lower(), follow

        changed = run("changed_system", {**long_body, "instructions": "Answer all questions concisely."})
        assert changed["usage"]["input_tokens_details"]["cached_tokens"] == 0
        shorter = run("shorter_prompt", {"input": "Say hello.", "max_output_tokens": 8}, stream=True)
        assert shorter["usage"]["input_tokens_details"]["cached_tokens"] == 0

        if not args.skip_tools:
            history = [{"role": "user", "content": "Call the weather tool with city set to Paris. Do not answer without calling it."}]
            tools = [{"type": "function", "name": "weather", "description": "Get the weather for a city.",
                      "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                     "required": ["city"]}}]
            tool_response = repeat("tool", {"input": history, "tools": tools, "max_output_tokens": 128})
            calls = [item for item in tool_response["output"] if item["type"] == "function_call"]
            assert tool_response["status"] == "completed" and calls
            assert all(json.loads(item["arguments"])["city"].lower() == "paris" for item in calls)
            history.extend(tool_response["output"])
            history.extend({"type": "function_call_output", "call_id": item["call_id"],
                            "output": '{"city":"Paris","temperature_c":18,"condition":"sunny"}'}
                           for item in calls)
            final = run("tool_continuation", {"input": history, "tools": tools,
                                             "max_output_tokens": 128}, stream=True)
            assert final["status"] == "completed" and "18" in output_text(final), final
        if cold is not None:
            reference = {item["name"]: item for item in cold["requests"]}
            comparisons = []
            for item in records:
                other = reference.get(item["name"])
                assert other is not None, f"cold report missing {item['name']}"
                same_request = normalize(item["request"]) == normalize(other["request"])
                exact = signature(item["response"]) == signature(other["response"])
                same_count = item["response"]["usage"]["output_tokens"] == other["response"]["usage"]["output_tokens"]
                comparisons.append({"name": item["name"], "exact_output": exact,
                                    "same_request": same_request,
                                    "same_output_tokens": same_count,
                                    "cold_seconds": other["seconds"], "seconds": item["seconds"],
                                    "speedup": other["seconds"] / item["seconds"] if same_request else None})
                if not same_request:
                    print(f"REQUEST DIFFERENCE: {item['name']} replays differing upstream output; see both reports", flush=True)
                if not exact or not same_count:
                    print(f"OUTPUT DIFFERENCE: {item['name']} differs from cold reference; see both reports", flush=True)
            report["cold_comparison"] = comparisons
            report["cold_comparison_exact"] = all(item["same_request"] and item["exact_output"] and item["same_output_tokens"]
                                                    for item in comparisons)
        report["repeat_output_exact"] = all(item["exact_output"] and item["same_output_tokens"]
                                             for item in report["repeat_checks"])
        report["passed"] = True
        print("Prefix reuse checks passed", flush=True)
    finally:
        save()


if __name__ == "__main__":
    main()
