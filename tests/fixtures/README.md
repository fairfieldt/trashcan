# Independent reference fixtures

`tokenizer-fixture.json` contains 30 exact llama.cpp tokenizations and decoded strings.
`chat-fixture.json` transcribes the llama.cpp renderings in
`results/gemma4-chat-template-examples.txt` for user, system, multi-turn and thinking-off prompts.
Its thinking-off token IDs are the fixed 15-token activation prompt in `docs/engine-spec.md`.

Run with the target GGUF, without a llama.cpp build:

```sh
trashcan tokenize MODEL --fixture tests/fixtures/tokenizer-fixture.json
trashcan render MODEL --fixture tests/fixtures/chat-fixture.json
```
