import json, sys
# One line per prompt: plain and MTP tok/s, identical output, acceptance (the mtp object's scalar fields).
for path in sys.argv[1:]:
    print('==', path.rsplit('/', 1)[-1])
    for l in open(path):
        l = l.strip()
        if not l.startswith('{'):
            continue
        d = json.loads(l)
        m = d['mtp']
        s = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items() if not isinstance(v, (list, dict))}
        same = m.get('ids') == d.get('plain_ids') if 'ids' in m else None
        print(d['prompt'], 'plain', round(d['plain_tok_s'], 2), 'same_ids', same, s)
