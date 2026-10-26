# Prints the named functions of a C file, unchanged, in file order (a definition may start with
# "static ...\nname(" or "static ... name("), so the harness compiles the kernel's own code.
# usage: python3 -I extract.py FILE NAME...
import re, sys
src = open(sys.argv[1]).read().split('\n')
names = sys.argv[2:]
found = {}
for i, line in enumerate(src):
    for n in names:
        if n in found:
            continue
        if re.match(r'^(static\s[^(;=]*\b)?' + re.escape(n) + r'\(', line) and not line.rstrip().endswith(';'):
            start = i - 1 if not line.startswith('static') and src[i - 1].startswith('static') else i
            end = next(j for j in range(i, len(src)) if src[j] == '}')
            found[n] = (start, end)
missing = [n for n in names if n not in found]
if missing:
    sys.exit('not found: ' + ' '.join(missing))
for n, (s, e) in sorted(found.items(), key=lambda kv: kv[1][0]):
    print(f'#line {s + 1} "stream.c"')
    print('\n'.join(src[s:e + 1]))
    print()
