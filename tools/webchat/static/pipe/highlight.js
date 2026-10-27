// highlight.js — the vendored syntax highlighter, wired for this UI.
//
// Two halves: the grammars, and the mapping from the strings this UI has
// (a fence's info string, a tool's file path) to one of them. Anything that
// maps to nothing renders as plain code, which is what the renderer did for
// every block before this existed — a wrong grammar is worse than none, so
// nothing is ever guessed from the content.
//
// The bundle (vendor/VENDOR.md) carries the 36 common grammars; the
// extras registered here are the ones agents write that it leaves out. They
// come from the same release as the bundle, so the two cannot drift.

import hljs from './vendor/highlight.esm.min.js';
import dockerfile from './vendor/hljs/dockerfile.js';
import nginx from './vendor/hljs/nginx.js';
import protobuf from './vendor/hljs/protobuf.js';
import powershell from './vendor/hljs/powershell.js';
import elixir from './vendor/hljs/elixir.js';
import haskell from './vendor/hljs/haskell.js';
import clojure from './vendor/hljs/clojure.js';
import erlang from './vendor/hljs/erlang.js';
import scala from './vendor/hljs/scala.js';
import dart from './vendor/hljs/dart.js';
import nim from './vendor/hljs/nim.js';
import ocaml from './vendor/hljs/ocaml.js';

for (const [name, grammar] of Object.entries({
  dockerfile, nginx, protobuf, powershell, elixir, haskell, clojure, erlang,
  scala, dart, nim, ocaml,
})) hljs.registerLanguage(name, grammar);

// A code block past this size is left plain: highlighting is a parse, and a
// model that pastes a megabyte of log should not lock the tab to colour it.
export const MAX_HIGHLIGHT_CHARS = 120000;

// Fence labels that are not the grammar's own name. hljs already answers to
// its aliases ("js", "py", "sh", "ts"), so this is only the rest.
const BY_LABEL = {
  jsx: 'javascript', mjs: 'javascript', cjs: 'javascript', node: 'javascript',
  tsx: 'typescript', python3: 'python', rb: 'ruby', rs: 'rust', golang: 'go',
  sh: 'bash', shell: 'bash', zsh: 'bash', console: 'bash', consolelog: 'bash',
  yml: 'yaml', jsonc: 'json', json5: 'json', htm: 'xml', html: 'xml',
  svg: 'xml', xhtml: 'xml', vue: 'xml', toml: 'ini', cfg: 'ini', conf: 'ini',
  docker: 'dockerfile', containerfile: 'dockerfile', make: 'makefile',
  mk: 'makefile', md: 'markdown', patch: 'diff', udiff: 'diff',
  text: 'plaintext', txt: 'plaintext', plain: 'plaintext', log: 'plaintext',
  cxx: 'cpp', cc: 'cpp', hpp: 'cpp', hxx: 'cpp', 'c++': 'cpp',
  cs: 'csharp', 'c#': 'csharp', ps1: 'powershell', pwsh: 'powershell',
  kts: 'kotlin', exs: 'elixir', lhs: 'haskell', cljs: 'clojure',
  cljc: 'clojure', edn: 'clojure', hrl: 'erlang', sbt: 'scala',
  mli: 'ocaml', mll: 'ocaml', mly: 'ocaml', pl: 'perl', pm: 'perl',
  psql: 'sql', mysql: 'sql', sqlite: 'sql', gql: 'graphql',
  proto: 'protobuf', proto3: 'protobuf', properties: 'ini', env: 'ini',
  gitignore: 'bash', hcl: 'hcl', tf: 'hcl', nginxconf: 'nginx',
  wat: 'wasm',
};

// Extensions, for the case where the path is all we have — a `read` result,
// a file in a grep hit. The name in a fence label is tried against this too,
// so "```py" and "foo.py" agree.
const BY_EXT = {
  c: 'c', h: 'c', cpp: 'cpp', cc: 'cpp', cxx: 'cpp', hpp: 'cpp', hh: 'cpp',
  cs: 'csharp', java: 'java', kt: 'kotlin', kts: 'kotlin', swift: 'swift',
  m: 'objectivec', mm: 'objectivec',
  js: 'javascript', mjs: 'javascript', cjs: 'javascript', jsx: 'javascript',
  ts: 'typescript', tsx: 'typescript',
  py: 'python', pyi: 'python', rb: 'ruby', php: 'php', pl: 'perl', pm: 'perl',
  go: 'go', rs: 'rust', lua: 'lua', r: 'r', jl: 'julia', dart: 'dart',
  nim: 'nim', ml: 'ocaml', mli: 'ocaml', hs: 'haskell', lhs: 'haskell',
  clj: 'clojure', cljs: 'clojure', cljc: 'clojure', erl: 'erlang',
  hrl: 'erlang', ex: 'elixir', exs: 'elixir', scala: 'scala',
  sh: 'bash', bash: 'bash', zsh: 'bash', fish: 'bash', ksh: 'bash',
  ps1: 'powershell', psm1: 'powershell',
  html: 'xml', htm: 'xml', xhtml: 'xml', xml: 'xml', svg: 'xml', vue: 'xml',
  css: 'css', scss: 'scss', less: 'less',
  json: 'json', jsonc: 'json', json5: 'json',
  yaml: 'yaml', yml: 'yaml', toml: 'ini', ini: 'ini', cfg: 'ini', conf: 'ini',
  properties: 'ini', env: 'ini',
  md: 'markdown', markdown: 'markdown',
  sql: 'sql', graphql: 'graphql', gql: 'graphql', proto: 'protobuf',
  diff: 'diff', patch: 'diff', tf: 'hcl', hcl: 'hcl',
  dockerfile: 'dockerfile', containerfile: 'dockerfile',
  mk: 'makefile', makefile: 'makefile', cmake: 'makefile',
  wasm: 'wasm', wat: 'wasm', txt: 'plaintext', log: 'plaintext',
};

function known(lang) {
  return !!lang && !!hljs.getLanguage(lang);
}

function resolve(name) {
  if (!name) return null;
  const raw = String(name).trim().toLowerCase();
  if (!raw) return null;
  if (known(raw)) return raw;
  const mapped = BY_LABEL[raw] || BY_EXT[raw];
  return known(mapped) ? mapped : null;
}

// A fence's info string ("```js", "```bash title=…") to a grammar name.
export function languageForLabel(label) { return resolve(label); }

// A path (or a bare filename) to a grammar name.
export function languageForPath(path) {
  if (!path) return null;
  const name = String(path).trim().split(/[?#]/)[0].split('/').pop().toLowerCase();
  if (!name) return null;
  if (name === 'dockerfile' || name.startsWith('dockerfile.') || name === 'containerfile') return 'dockerfile';
  if (name === 'makefile' || name === 'gnumakefile' || name === 'cmakelists.txt') return 'makefile';
  const dot = name.lastIndexOf('.');
  return resolve(dot > 0 ? name.slice(dot + 1) : name);
}

// The highlighted HTML for a block, or null when it should stay plain. The
// caller escapes the plain form, so a null return is never a hole.
export function highlight(code, lang) {
  if (!lang || !known(lang) || code.length > MAX_HIGHLIGHT_CHARS) return null;
  try {
    return hljs.highlight(code, { language: lang, ignoreIllegals: true }).value;
  } catch (e) {
    return null; // a grammar that throws on odd input is not worth the block
  }
}

export function isKnownLanguage(lang) { return known(lang); }
export { hljs };
