"""Greedy completions from a running llama-server (draft-mtp) for each prompt and draft depth; prints and saves the
generated ids and llama.cpp's draft acceptance.

usage: llama_accept.py URL OUT.json N_PREDICT K[,K...] PROMPT.ids...
"""
import json, os, sys, time, urllib.request

url, out, n_predict = sys.argv[1], sys.argv[2], int(sys.argv[3])
ks = [int(k) for k in sys.argv[4].split(',')]
results = []
for path in sys.argv[5:]:
    ids = [int(x) for x in open(path).read().split()]
    for k in ks:
        req = {"prompt": ids, "n_predict": n_predict, "temperature": 0, "top_k": 1, "cache_prompt": False,
               "return_tokens": True, "samplers": ["top_k"], "speculative.n_max": k, "speculative.n_min": 0,
               "speculative.p_min": 0.0}
        t = time.time()
        r = json.load(urllib.request.urlopen(urllib.request.Request(
            url + '/completion', json.dumps(req).encode(), {'Content-Type': 'application/json'}), timeout=7200))
        tm = r.get("timings", {})
        rec = {"prompt": os.path.basename(path), "prompt_tokens": len(ids), "k": k, "wall_s": time.time() - t,
               "tokens": r.get("tokens"), "text": r.get("content"), "timings": tm}
        results.append(rec)
        dn, da = tm.get("draft_n", 0), tm.get("draft_n_accepted", 0)
        print(f"{rec['prompt']} k={k}: {len(rec['tokens'] or [])} tokens, drafts {da}/{dn} accepted "
              f"({da / max(dn, 1):.3f}), predicted {tm.get('predicted_per_second', 0):.2f} tok/s", flush=True)
        json.dump(results, open(out, "w"), indent=1)
