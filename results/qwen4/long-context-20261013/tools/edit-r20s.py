#!/usr/bin/env python3
# r20s (A/B variant, never committed): Local::step waits for a stage's running chunk before submitting the incoming
# one, as main does (one chunk per GPU at a time), instead of queueing the incoming one behind it.
p = '/Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a43eb2477e990f76f/engine/src/qwen4/pipeline.rs'
s = open(p).read()
old = '''            let incoming = input.is_some();
            if let Some((f, x)) = input.take() {
                s.submit(&x, f.ple.as_deref(), f.pos, f.n, false)?;
                tl("sub", i, f.pos, 0., 0.);
                flight.push_back(f);
            }
            let prev = if flight.len() > usize::from(incoming) {'''
new = '''            let cur = input.take();
            let prev = if !flight.is_empty() {'''
assert s.count(old) == 1
s = s.replace(old, new)
old2 = '''            } else {
                None
            };
            match prev {'''
new2 = '''            } else {
                None
            };
            if let Some((f, x)) = cur {
                s.submit(&x, f.ple.as_deref(), f.pos, f.n, false)?;
                tl("sub", i, f.pos, 0., 0.);
                flight.push_back(f);
            }
            match prev {'''
assert s.count(old2) == 1
s = s.replace(old2, new2)
open(p, 'w').write(s)
print('ok')
