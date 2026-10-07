"""Build the quality-evaluation corpus: fixed slices of local English prose, markdown, code and Italian prose."""
import sys

parts = [
    ('/Users/fairfieldt/dev/claude-work/trashcan-20261006/hf/Qwen3.8-Flash-Next/README.md', 0, 30000),
    ('/Users/fairfieldt/dev/trashcan/llama.cpp/docs/build.md', 0, 25000),
    ('/Users/fairfieldt/dev/trashcan/llama.cpp/tools/server/README.md', 0, 25000),
    ('/Users/fairfieldt/dev/trashcan/llama.cpp/src/llama-sampler.cpp', 20000, 45000),
    ('/Users/fairfieldt/dev/ds4-voyage/speed-bench/promessi_sposi.txt', 100000, 125000),
]
out = []
for path, a, b in parts:
    text = open(path, encoding='utf-8', errors='replace').read()[a:b]
    # Start and end on line boundaries.
    text = text[text.find('\n') + 1:text.rfind('\n')]
    out.append(text)
with open(sys.argv[1], 'w', encoding='utf-8') as f:
    f.write('\n\n'.join(out))
