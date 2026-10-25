# Independent reference fixtures

`tokenizer-fixture.json` contains 30 exact llama.cpp tokenizations and decoded strings.
`chat-fixture.json` transcribes llama.cpp's Gemma 4 chat template renderings for user, system,
multi-turn and thinking-off prompts.
Its thinking-off token IDs are the fixed 15-token activation prompt in `docs/engine-spec.md`.

Run with the target GGUF, without a llama.cpp build:

```sh
trashcan tokenize MODEL --fixture tests/fixtures/tokenizer-fixture.json
trashcan render MODEL --fixture tests/fixtures/chat-fixture.json
```

`qwen-bench-corpus.jsonl` contains code, Spanish, Chinese, a tool request and result, thinking,
seeded sampling, and a longer tutorial generation. Run it with nothing else on the GPUs, under the clock
wrapper (`scripts/with-clocks.sh`):

```sh
trashcan qwen-bench MODEL --tp-stream /dev/tbstream0,/dev/tbstream1,/dev/tbstream2 \
  --corpus tests/fixtures/qwen-bench-corpus.jsonl --tg 128 --reps 2 [--mtp MTP]
```

Each case can set `thinking`, `tools`, `temperature`, `top_k`, `top_p`, `seed`, and `max_tokens`.
Results are JSONL and include token IDs, exact timed windows, both hosts' CPU settings, and MTP statistics.
The tutorial case requests up to 2048 tokens; other cases use `--tg`. Repetitions keep weights loaded
and restart recurrent state at position zero. Exclude warm-up and flag clock throttling when comparing rates.
