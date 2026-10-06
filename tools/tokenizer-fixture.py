#!/usr/bin/env python3
"""Write exact llama.cpp tokenizations of INPUTS (a JSON list of strings) as a trashcan tokenizer fixture.

usage: tokenizer-fixture.py LLAMA_TOKENIZE MODEL INPUTS OUT

Each case records the token ids (special tokens parsed, BOS as the model requests) and the text produced
by concatenating llama.cpp's pieces, which `trashcan tokenize MODEL --fixture OUT` compares against.
"""
import json
import subprocess
import sys


def run(binary, model, text, ids):
    args = [binary, "-m", model, "--stdin", "--log-disable"] + (["--ids"] if ids else [])
    out = subprocess.run(args, input=text.encode(), capture_output=True, check=True).stdout
    return out


def main():
    binary, model, inputs, out = sys.argv[1:5]
    cases = []
    for text in json.load(open(inputs)):
        ids = json.loads(run(binary, model, text, True)) if text else []
        # Each piece line is "<id> -> '<piece>'"; pieces may contain newlines, so split on the id markers.
        # Pieces can end inside a UTF-8 sequence, so split and join the raw bytes.
        rest = run(binary, model, text, False) if text else b""
        pieces = []
        for i, token in enumerate(ids):
            marker = f"{token} -> '".encode()
            start = rest.index(marker) + len(marker)
            nxt = rest.find(f"{ids[i + 1]} -> '".encode(), start) if i + 1 < len(ids) else len(rest)
            piece = rest[start:nxt].rstrip(b" ")
            pieces.append(piece[: piece.rfind(b"'")])
            rest = rest[nxt:]
        joined = b"".join(pieces).decode("utf-8", "replace")
        pieces = [p.decode("utf-8", "replace") for p in pieces]
        cases.append({"text": text, "ids": ids, "pieces": pieces, "detokenized": joined})
    json.dump(cases, open(out, "w"), ensure_ascii=False, indent=1)
    print(f"wrote {len(cases)} cases to {out}")


if __name__ == "__main__":
    main()
