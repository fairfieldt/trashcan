# trashcan

Fast llama.cpp inference on the 2013 Mac Pro (incite: 2x FirePro D700, Vulkan/RADV), starting with
Vulkan tensor parallelism for Gemma 4 12B QAT Q4_0.

- `docs/hardware.md`: the machine, the D700 clock bug, baseline numbers
- `docs/upstream-survey.md`: upstream TP/meta-backend state, Vulkan PRs and forks
- `llama.cpp/`: upstream clone (own git repo, not tracked here; PR branches fetched as local refs)
