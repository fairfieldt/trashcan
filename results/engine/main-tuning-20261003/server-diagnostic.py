#!/usr/bin/env python3
"""Replay the resident greedy/stochastic timing anomaly using Python's standard library."""
import argparse
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

PROMPT = ("List all integers from 1 to 1000 in order, separated by spaces. "
          "Print only the full list, with no introduction, summary, or abbreviations.")


def output_text(response):
    return "".join(part["text"] for item in response["output"] if item["type"] == "message"
                   for part in item["content"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18160")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json-only", action="store_true", help="Use JSON for repeat requests too")
    parser.add_argument("--replay-report", type=Path,
                        help="Replay every exact saved request from a prefix report (all 19 in a complete report)")
    args = parser.parse_args()
    if args.replay_report:
        saved = json.loads(args.replay_report.read_text())
        cases = [(item["name"], item["request"], output_text(item["response"]))
                 for item in saved["requests"]]
    else:
        basic = {"input": PROMPT, "max_output_tokens": 32, "store": False}
        sampled = {**basic, "temperature": 1, "seed": 7, "top_k": 32}
        greedy = {**basic, "temperature": 0}
        cases = [
            ("warmup", {"input": "Say hello.", "temperature": 0, "max_output_tokens": 1,
                        "store": False, "stream": False}, None),
            ("stochastic_first", {**sampled, "stream": False}, None),
            ("stochastic_repeat", {**sampled, "stream": not args.json_only}, None),
            ("greedy_first", {**greedy, "stream": False}, None),
            ("greedy_repeat", {**greedy, "stream": not args.json_only}, None),
        ]
    report = {"url": args.url, "replay_report": str(args.replay_report) if args.replay_report else None,
              "requests": [], "passed": False}

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    try:
        for index, (name, body, expected) in enumerate(cases, 1):
            body = json.loads(json.dumps(body))
            request = urllib.request.Request(args.url.rstrip("/") + "/v1/responses",
                                             data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json"}, method="POST")
            epoch_started = time.time()
            started = time.perf_counter()
            first_delta = None
            event_records = []
            with urllib.request.urlopen(request, timeout=600) as response:
                status = response.status
                headers_seconds = time.perf_counter() - started
                if response.headers.get("Content-Type", "").startswith("text/event-stream"):
                    for line in response:
                        if not line.startswith(b"data: "):
                            continue
                        event = json.loads(line[6:])
                        seconds = time.perf_counter() - started
                        assert event["sequence_number"] == len(event_records), event
                        event_records.append({"seconds": seconds, "event": event})
                        if first_delta is None and event["type"].endswith(".delta"):
                            first_delta = seconds
                    assert event_records[-1]["event"]["type"] in ("response.completed", "response.incomplete")
                    value = event_records[-1]["event"]["response"]
                    delta = "".join(item["event"]["delta"] for item in event_records
                                    if item["event"]["type"] == "response.output_text.delta")
                    assert delta == output_text(value)
                else:
                    value = json.loads(response.read())
            seconds = time.perf_counter() - started
            assert status == 200, value
            item = {"request_index_in_harness": index, "name": name, "request": body,
                    "epoch_started": epoch_started, "seconds": seconds,
                    "headers_seconds": headers_seconds, "first_delta_seconds": first_delta,
                    "response_id": value["id"], "response": value, "events": event_records,
                    "matches_saved_visible_text": output_text(value) == expected if expected is not None else None}
            report["requests"].append(item)
            save()
            usage = value["usage"]
            print(f"{name}: {seconds:.6f}s, first_delta={first_delta}, "
                  f"input={usage['input_tokens']} cached={usage['input_tokens_details']['cached_tokens']} "
                  f"output={usage['output_tokens']} response={value['id']}", flush=True)
        if args.replay_report:
            report["visible_text_equal"] = all(item["matches_saved_visible_text"] for item in report["requests"])
        else:
            measured = report["requests"][1:]
            texts = [output_text(item["response"]) for item in measured]
            report["visible_text_equal"] = bool(texts[0]) and all(text == texts[0] for text in texts)
            assert all(item["response"]["usage"]["input_tokens"] == 47 for item in measured)
            assert all(item["response"]["usage"]["output_tokens"] == 32 for item in measured)
            assert all(item["response"]["usage"]["output_tokens_details"]["reasoning_tokens"] == 0 for item in measured)
        report["passed"] = report["visible_text_equal"]
        assert report["visible_text_equal"], "Visible output differs; all outputs retained for diagnosis"
    except Exception as error:
        report["error"] = repr(error)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
