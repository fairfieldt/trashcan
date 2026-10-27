// md.js — markdown rendering for assistant text.
//
// The parser is markdown-it (vendored; see vendor/VENDOR.md), which is
// where CommonMark correctness comes from: nested lists, emphasis that nests,
// reference links, setext headings, autolinks, code spans with any number of
// backticks. What this file owns is everything that is about *this* UI rather
// than about markdown — the DOM app.css already styles, fenced blocks that
// carry a filename, line numbers and highlighted ranges, the `diff`/`ansi`
// shapes a tool result arrives in, and the math/diagram placeholders that
// hydrate() fills in once a message has stopped streaming.
//
// The security property is unchanged and is the reason for `html: false`
// below: a model's text is never markup. Everything the parser emits is
// either a tag this file wrote or a string it escaped. Inline HTML is
// escaped, not passed through, deliberately — do not "fix" that.
//
// renderMarkdown stays synchronous: it is called per streaming delta, and an
// await in that path would reorder the DOM. The two things that genuinely
// need to be async (KaTeX, Mermaid) are separate: renderMarkdown leaves a
// placeholder, and hydrate() upgrades it afterwards.

import MarkdownIt from './vendor/markdown-it.esm.min.mjs';
import footnotePlugin from './vendor/markdown-it-footnote.mjs';
import deflistPlugin from './vendor/markdown-it-deflist.mjs';
import markPlugin from './vendor/markdown-it-mark.mjs';
import { highlight, languageForLabel } from './highlight.js';
import { ansiToHtml, hasAnsi, stripAnsi } from './ansi.js';

const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
const escAttr = (s) => esc(s).replace(/'/g, '&#39;');

// The two big optional libraries load on demand, so their paths are written
// out in full rather than assembled from a prefix: web/htmlboot/boot.html
// materializes an app by rewriting a reference into a blob: URL only when the
// literal in the source resolves to a file the app ships, and './vendor/' +
// rel is not a file. The names below are, and `vendor()` turns one into an
// absolute URL (which is a no-op on the blob URL the rewriter already left).
const VENDOR_FILES = {
  katexCss: './vendor/katex/katex.min.css',
  katexJs: './vendor/katex/katex.min.js',
  mermaidJs: './vendor/mermaid.min.js',
};
const vendor = (rel) => new URL(rel, import.meta.url).href;

// A block longer than this starts folded; the reader opens it in place. A
// model that pastes 400 lines should not push the rest of the answer off the
// screen, and the text is still in the DOM (and still what "copy" copies).
const FOLD_AFTER_LINES = 40;

// ------------------------------------------------------------ code spans
//
// A `|` inside a code span is content, not a table cell separator. Models
// write `a|b` in a table far more often than they escape it, and the strict
// rule turns one such cell into two broken ones. markdown-it's table rule
// splits on every unescaped pipe, so pipes inside a *closed* backtick span
// are swapped for a placeholder that cannot otherwise occur in the input, and
// swapped back on the way out. The swap is per line and the restore is
// global, so a line that is not a table round-trips byte for byte.
const PIPE = '\u0001';

function protectCodePipes(line) {
  if (line.indexOf('`') < 0 || line.indexOf('|') < 0) return line;
  const runs = [];
  for (let j = 0; j < line.length; j++) {
    if (line[j] !== '`') continue;
    let n = 1;
    while (line[j + n] === '`') n++;
    runs.push([j, n]);
    j += n - 1;
  }
  // Only spans that actually close on this line: an unclosed backtick is
  // literal text, and its pipes really are separators.
  const spans = [];
  for (let k = 0; k < runs.length; k++) {
    for (let l = k + 1; l < runs.length; l++) {
      if (runs[l][1] === runs[k][1]) { spans.push([runs[k][0], runs[l][0] + runs[l][1]]); k = l; break; }
    }
  }
  if (!spans.length) return line;
  let out = '', prev = 0;
  for (const [s, e] of spans) {
    out += line.slice(prev, s) + line.slice(s, e).split('|').join(PIPE);
    prev = e;
  }
  return out + line.slice(prev);
}

// ------------------------------------------------------------- math ($…$)
//
// A small plugin rather than a preprocess: it has to sit in the token stream
// so that `$` inside a code span, a fence or a link label stays literal. The
// rules are the conventional ones — the opening `$` may not be followed by
// space and the closing one may not be preceded by it, so "$5 and $10" and
// "$HOME" stay text.

function mathPlugin(md) {
  const isSpace = (ch) => ch === ' ' || ch === '\t' || ch === '\n';

  md.inline.ruler.after('escape', 'math_inline', function mathInline(state, silent) {
    const src = state.src;
    const start = state.pos;
    if (src.charCodeAt(start) !== 0x24 /* $ */) return false;
    if (src[start + 1] === '$') return false;           // that is display math
    let pos = start + 1;
    if (pos >= src.length || isSpace(src[pos])) return false;
    while (pos < src.length) {
      const ch = src[pos];
      if (ch === '\\') { pos += 2; continue; }
      if (ch === '$') break;
      if (ch === '\n') return false;                    // inline math is one line
      pos++;
    }
    if (pos >= src.length || pos === start + 1 || isSpace(src[pos - 1])) return false;
    if (!silent) {
      const token = state.push('math_inline', 'math', 0);
      token.content = src.slice(start + 1, pos);
    }
    state.pos = pos + 1;
    return true;
  });

  md.block.ruler.before('fence', 'math_block', function mathBlock(state, startLine, endLine, silent) {
    const pos = state.bMarks[startLine] + state.tShift[startLine];
    const max = state.eMarks[startLine];
    const src = state.src;
    if (src.slice(pos, pos + 2) !== '$$') return false;
    if (silent) return true;
    const rest = src.slice(pos + 2, max);
    let content = null, nextLine = startLine + 1;
    const sameLine = rest.indexOf('$$');
    if (sameLine >= 0) {
      content = rest.slice(0, sameLine);
    } else {
      const buf = [rest];
      for (let l = startLine + 1; l < endLine; l++) {
        const text = src.slice(state.bMarks[l] + state.tShift[l], state.eMarks[l]);
        const at = text.indexOf('$$');
        nextLine = l + 1;
        if (at >= 0) { buf.push(text.slice(0, at)); content = buf.join('\n'); break; }
        buf.push(text);
      }
    }
    if (content === null) return false;
    const token = state.push('math_block', 'math', 0);
    token.block = true;
    token.content = content.trim();
    token.map = [startLine, nextLine];
    state.line = nextLine;
    return true;
  }, { alt: ['paragraph', 'reference', 'blockquote', 'list'] });
}

function mathPlaceholder(kind, tex, env) {
  if (env && env.math === false) {
    return kind === 'block' ? `<p>${esc('$$' + tex + '$$')}</p>` : esc('$' + tex + '$');
  }
  const cls = kind === 'block' ? 'math math-block' : 'math math-inline';
  return `<${kind === 'block' ? 'div' : 'span'} class="${cls}" data-tex="${escAttr(tex)}">${esc(tex)}</${kind === 'block' ? 'div' : 'span'}>`;
}

// ------------------------------------------------------------ the parser

const md = new MarkdownIt({
  html: false,        // never trust a model's markup (see the header)
  linkify: true,      // a bare URL in prose is a link, as it always was
  breaks: false,      // CommonMark: a soft break joins, `  \n` is a hard one
  typographer: false, // no smart quotes: they rewrite text the user may mean
});

md.use(footnotePlugin).use(deflistPlugin).use(markPlugin).use(mathPlugin);

// -[ ] / -[x]. The plugin is CommonJS-only, so index.html loads it as a
// classic script; if that tag ever goes missing the renderer degrades to a
// plain list rather than throwing on every message.
const taskLists = globalThis.markdownitTaskLists;
if (taskLists) md.use(taskLists, { label: false });   // disabled inputs: a message is not a form

// Headings shift down two levels: a message is already inside a page that
// uses h1/h2, and the old renderer did the same.
const shiftHeading = (tag) => 'h' + Math.min(Number(tag.slice(1)) + 2, 6);
md.renderer.rules.heading_open = (tokens, idx, options, env, slf) => {
  tokens[idx].tag = shiftHeading(tokens[idx].tag);
  return slf.renderToken(tokens, idx, options);
};
md.renderer.rules.heading_close = (tokens, idx, options, env, slf) => {
  tokens[idx].tag = shiftHeading(tokens[idx].tag);
  return slf.renderToken(tokens, idx, options);
};

// Links: a URL opens in a new tab; a relative path is a reference to a file
// the agent can actually show, so it is marked for the app to intercept
// rather than being a broken navigation.
const HAS_SCHEME = /^[a-zA-Z][a-zA-Z0-9+.-]*:/;
md.renderer.rules.link_open = (tokens, idx, options, env, slf) => {
  const token = tokens[idx];
  const href = token.attrGet('href') || '';
  if (HAS_SCHEME.test(href)) {
    token.attrSet('target', '_blank');
    token.attrSet('rel', 'noopener noreferrer');
  } else if (href && href[0] !== '#' && /[./]/.test(href)) {
    token.attrSet('class', 'file-link');
    token.attrSet('data-file', href);
    token.attrSet('href', '#');
  }
  return slf.renderToken(tokens, idx, options);
};

const defaultImage = md.renderer.rules.image;
md.renderer.rules.image = (tokens, idx, options, env, slf) => {
  const token = tokens[idx];
  token.attrSet('loading', 'lazy');
  token.attrSet('decoding', 'async');
  token.attrSet('class', 'md-img');
  return defaultImage(tokens, idx, options, env, slf);
};

md.renderer.rules.math_inline = (tokens, idx, options, env) => mathPlaceholder('inline', tokens[idx].content, env);
md.renderer.rules.math_block = (tokens, idx, options, env) => mathPlaceholder('block', tokens[idx].content, env);

// Tables. markdown-it's own renderer is close, but this app already styles
// `.table-wrap`/`.md-table`/`.ta-*` and a test pins the exact DOM, so the
// rules are overridden to emit that shape rather than a second one.
const cellOpen = (tag) => (tokens, idx) => {
  let cls = '';
  for (const [name, value] of tokens[idx].attrs || []) {
    if (name !== 'style') continue;
    const m = /text-align:\s*(left|center|right)/.exec(value);
    if (m) cls = ` class="ta-${m[1]}"`;
  }
  return `<${tag}${cls}>`;
};
Object.assign(md.renderer.rules, {
  table_open: () => '<div class="table-wrap"><table class="md-table">',
  table_close: () => '</table></div>',
  thead_open: () => '<thead>', thead_close: () => '</thead>',
  tbody_open: () => '<tbody>', tbody_close: () => '</tbody>',
  tr_open: () => '<tr>', tr_close: () => '</tr>',
  th_open: cellOpen('th'), th_close: () => '</th>',
  td_open: cellOpen('td'), td_close: () => '</td>',
});

// ------------------------------------------------------------ fenced code

function parseInfo(info) {
  const out = { lang: '', title: '', numbers: false, ranges: null };
  let rest = String(info || '').trim();
  const m = /^[^\s{]*/.exec(rest);
  out.lang = m[0];
  rest = rest.slice(out.lang.length);

  const brace = /\{([^{}]*)\}/.exec(rest);
  if (brace) {
    out.ranges = new Set();
    for (const part of brace[1].split(/[,\s]+/)) {
      const range = /^(\d+)-(\d+)$/.exec(part);
      if (range) { for (let n = +range[1]; n <= +range[2]; n++) out.ranges.add(n); }
      else if (/^\d+$/.test(part)) out.ranges.add(+part);
    }
    if (!out.ranges.size) out.ranges = null;
    rest = rest.replace(brace[0], ' ');
  }

  const titled = /\btitle\s*=\s*(?:"([^"]*)"|'([^']*)'|(\S+))/i.exec(rest);
  if (titled) {
    out.title = titled[1] ?? titled[2] ?? titled[3] ?? '';
    rest = rest.replace(titled[0], ' ');
  } else {
    const quoted = /^\s*"([^"]*)"/.exec(rest);
    if (quoted) out.title = quoted[1];
  }

  for (const word of rest.toLowerCase().split(/[\s,]+/)) {
    if (word === 'ln' || word === 'line-numbers' || word === 'linenos') out.numbers = true;
  }
  return out;
}

// hljs emits spans that can straddle a newline (a multi-line string, a
// comment block). To number lines the HTML has to be split per line, which
// means closing every open span at the break and reopening it after — the
// only way to keep the markup balanced without re-parsing it.
function splitHighlightedLines(html) {
  const lines = [];
  const open = [];
  let cur = '';
  const closeAll = () => '</span>'.repeat(open.length);
  const addText = (text) => {
    const parts = text.split('\n');
    for (let i = 0; i < parts.length; i++) {
      if (i) { lines.push(cur + closeAll()); cur = open.join(''); }
      cur += parts[i];
    }
  };
  const tag = /<\/?span\b[^>]*>/g;
  let last = 0, m;
  while ((m = tag.exec(html))) {
    addText(html.slice(last, m.index));
    if (m[0][1] === '/') { open.pop(); cur += '</span>'; }
    else { open.push(m[0]); cur += m[0]; }
    last = tag.lastIndex;
  }
  addText(html.slice(last));
  lines.push(cur + closeAll());
  return lines;
}

function lineSpans(htmlLines, info) {
  const first = info.firstLine > 0 ? info.firstLine : 1;
  return htmlLines.map((line, i) => {
    const hl = info.ranges && info.ranges.has(i + first) ? ' hl' : '';
    return `<span class="code-line${hl}">${line}</span>`;
  }).join('');
}

// Wrap a block in the fold a long one starts in: the content stays in the
// DOM (and is still what "copy" copies), it just does not own the screen.
function fold(html, lineCount) {
  return `<div class="code-fold" data-lines="${lineCount}">${html}` +
    `<button class="code-more" type="button">show all ${lineCount} lines</button></div>`;
}

// A code block, the way this app draws one. Exported because the tool cards
// draw the same thing for a command, a file, a readout — one shape, one set
// of rules about escaping, highlighting and folding.
//
// opts: {lang, title, numbers, ranges, firstLine, fold, ansi}
//   firstLine  the number the first line carries (a `read` window at 40)
//   ranges     Set of 1-based line numbers to mark as changed
//   ansi       render escape sequences as colour instead of stripping them
//   fold       false to keep a long block open (the caller knows better)
export function codeBlock(code, opts = {}) {
  const raw = String(code == null ? '' : code);
  const lang = opts.lang || '';
  // Raw escapes never reach the DOM. Either the block is rendered as a
  // terminal capture — colours, cursor-overwrite and all — or they are
  // dropped before anything else looks at the text: a raw ESC is invisible
  // in the browser and the copy button would faithfully hand it back.
  const terminal = !!opts.ansi || (hasAnsi(raw) && isTerminalLabel(lang));
  const text = terminal ? raw : (hasAnsi(raw) ? stripAnsi(raw) : raw);
  const lines = text.split('\n');
  const numbers = !!opts.numbers || !!opts.ranges;
  const grammar = languageForLabel(lang);
  // A diff is read by the line, so it gets line spans even without numbers:
  // that is what lets a removal or an addition be washed edge to edge.
  const byLine = numbers || grammar === 'diff' || grammar === 'patch';

  let body = null;
  if (terminal) {
    body = ansiToHtml(text);
  } else if (opts.highlight !== false) {
    const marked = highlight(text, grammar);
    if (marked) {
      const split = byLine ? splitHighlightedLines(marked) : null;
      body = split && split.length === lines.length ? lineSpans(split, opts) : marked;
    }
  }
  if (body === null) body = numbers ? lineSpans(lines.map(esc), opts) : esc(text);

  const classes = ['code'];
  if (numbers) classes.push('code-lines');
  const reset = numbers && opts.firstLine > 1 ? ` style="counter-reset:ln ${opts.firstLine - 1}"` : '';
  const pre = `<pre class="${classes.join(' ')}" data-lang="${escAttr(lang)}">` +
    `<code${reset}>${body}</code>` +
    `<button class="copy" type="button" title="Copy" aria-label="Copy code">copy</button></pre>`;

  let out = opts.fold !== false && lines.length > FOLD_AFTER_LINES ? fold(pre, lines.length) : pre;
  if (!opts.title) return out;
  return `<div class="code-block"><div class="code-head">` +
    `<span class="code-title"${/[./]/.test(opts.title) ? ` data-file="${escAttr(opts.title)}"` : ''}>${esc(opts.title)}</span>` +
    `</div>${out}</div>`;
}

const isTerminalLabel = (label) =>
  /^(ansi|terminal|term|console|consolelog|shell-session|log|text|txt|plaintext|session)$/.test(String(label).toLowerCase());

function renderFence(token, env) {
  const info = parseInfo(token.info);
  const code = token.content.replace(/\n$/, '');
  const label = info.lang.toLowerCase();

  if (label === 'mermaid' || label === 'mmd') return renderDiagram(code, info);
  if (label === 'math' || label === 'latex' || label === 'katex') {
    return mathPlaceholder('block', code.trim(), env);
  }
  return codeBlock(code, {
    lang: info.lang,
    title: info.title,
    numbers: info.numbers,
    ranges: info.ranges,
    highlight: !env || env.highlight !== false,
  });
}

md.renderer.rules.fence = (tokens, idx, options, env) => renderFence(tokens[idx], env);
md.renderer.rules.code_block = (tokens, idx, options, env) => renderFence(tokens[idx], env);

// A ```mermaid block is a placeholder until hydrate() runs: the diagram
// library is 2 MB and most messages have no diagram in them.
function renderDiagram(code, info) {
  const title = info.title ? `<span class="code-title">${esc(info.title)}</span>` : '';
  return `<figure class="diagram" data-src="${escAttr(code)}">` +
    `<div class="diagram-head"><span class="eyebrow">mermaid</span>${title}` +
    `<button class="copy" type="button" title="Copy source" aria-label="Copy diagram source">copy</button></div>` +
    `<div class="diagram-body"><pre class="code" data-lang=""><code>${esc(code)}</code></pre></div></figure>`;
}

// ------------------------------------------------------------ public API

// `opts.highlight:false` skips the syntax pass, which is what a streaming
// delta wants: the block is re-rendered on every token, and only the final
// render is worth colouring. `opts.math:false` leaves math as literal text.
export function renderMarkdown(src, opts) {
  if (src == null || src === '') return '';
  const text = String(src).split(PIPE).join('').replace(/\r\n?/g, '\n');
  const env = { highlight: !opts || opts.highlight !== false, math: !opts || opts.math !== false };
  const html = md.render(text.split('\n').map(protectCodePipes).join('\n'), env);
  return html.indexOf(PIPE) < 0 ? html : html.split(PIPE).join('|');
}

// ------------------------------------------------------------- hydration

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const el = document.createElement('script');
    el.src = src;
    el.onload = resolve;
    el.onerror = () => reject(new Error('could not load ' + src));
    document.head.appendChild(el);
  });
}

function loadStyle(href) {
  return new Promise((resolve) => {
    const el = document.createElement('link');
    el.rel = 'stylesheet';
    el.href = href;
    el.onload = resolve;
    el.onerror = resolve;   // a missing stylesheet must not block the math
    document.head.appendChild(el);
  });
}

let katexPromise = null;
function loadKatex() {
  if (!katexPromise) {
    katexPromise = loadStyle(vendor(VENDOR_FILES.katexCss))
      .then(() => loadScript(vendor(VENDOR_FILES.katexJs)))
      .then(() => globalThis.katex)
      // A failed load is not cached: a later hydrate() (after the network comes
      // back, say) gets to try again.
      .catch((e) => { katexPromise = null; throw e; });
  }
  return katexPromise;
}

// A custom property's computed value is its *token text*, and this stylesheet
// declares its surfaces as light-dark(a, b) — a colour function mermaid's own
// parser rejects. Asking for the used value of a real colour property makes the
// browser resolve it to rgb() first, which every consumer understands.
function cssColor(name, fallback) {
  const probe = document.createElement('span');
  probe.setAttribute('aria-hidden', 'true');
  probe.style.cssText = 'position:absolute;visibility:hidden;pointer-events:none';
  probe.style.color = `var(${name}, ${fallback})`;
  document.body.appendChild(probe);
  const out = getComputedStyle(probe).color;
  probe.remove();
  return out || fallback;
}

let mermaidPromise = null;
function loadMermaid() {
  if (!mermaidPromise) {
    mermaidPromise = loadScript(vendor(VENDOR_FILES.mermaidJs)).then(() => {
      const mermaid = globalThis.mermaid;
      const token = (name, fallback) =>
        (getComputedStyle(document.documentElement).getPropertyValue(name) || '').trim() || fallback;
      mermaid.initialize({
        startOnLoad: false,
        securityLevel: 'strict',
        theme: 'base',
        fontFamily: token('--font-mono', 'monospace'),
        themeVariables: {
          background: cssColor('--bg-1', '#111'),
          primaryColor: cssColor('--bg-2', '#222'),
          primaryTextColor: cssColor('--text', '#eee'),
          primaryBorderColor: cssColor('--line', '#444'),
          lineColor: cssColor('--text-3', '#888'),
          secondaryColor: cssColor('--bg-2', '#222'),
          tertiaryColor: cssColor('--bg-1', '#111'),
        },
      });
      return mermaid;
    }).catch((e) => { mermaidPromise = null; throw e; });
  }
  return mermaidPromise;
}

function mathNodes(root) {
  return root.querySelectorAll ? root.querySelectorAll('.math[data-tex]:not([data-hydrated])') : [];
}

async function hydrateMath(root) {
  const nodes = mathNodes(root);
  if (!nodes.length) return;
  let katex;
  try { katex = await loadKatex(); } catch (e) { return; }   // offline: the TeX stays readable
  if (!katex) return;
  for (const node of nodes) {
    const tex = node.getAttribute('data-tex') || '';
    node.setAttribute('data-hydrated', '');
    try {
      katex.render(tex, node, {
        displayMode: node.classList.contains('math-block'),
        throwOnError: false,
        errorColor: 'var(--red-ink, #c33)',
      });
    } catch (e) {
      node.textContent = tex;   // never leave a blank hole where math was
    }
  }
}

let diagramSeq = 0;

async function hydrateDiagrams(root) {
  const nodes = root.querySelectorAll ? root.querySelectorAll('.diagram[data-src]:not([data-hydrated])') : [];
  if (!nodes.length) return;
  let mermaid;
  try { mermaid = await loadMermaid(); } catch (e) { return; }
  if (!mermaid) return;
  for (const node of nodes) {
    const src = node.getAttribute('data-src') || '';
    const body = node.querySelector('.diagram-body');
    node.setAttribute('data-hydrated', '');
    const id = 'pipe-diagram-' + (++diagramSeq);
    try {
      const { svg } = await mermaid.render(id, src);
      if (body) body.innerHTML = svg;
      node.classList.add('diagram-done');
    } catch (e) {
      // A diagram that will not parse is still the model's source, so it
      // stays on screen with the reason next to it rather than vanishing.
      node.classList.add('diagram-error');
      const note = document.createElement('div');
      note.className = 'diagram-note';
      note.textContent = (e && e.message ? e.message : 'could not render diagram');
      if (body) body.prepend(note);
    } finally {
      // mermaid measures in a temporary element it usually removes itself; when
      // it throws, it can leave one behind. Only that: the svg it returned has
      // the same id and lives in this node, so it is not a stray.
      const stray = document.getElementById('d' + id) || document.getElementById(id);
      if (stray && stray.parentNode && !node.contains(stray)) stray.parentNode.removeChild(stray);
    }
  }
}

// Upgrade the placeholders renderMarkdown left behind. Call it once a message
// has stopped streaming — not per delta, which would re-render the same
// diagram dozens of times. Both halves are no-ops when there is nothing to
// upgrade, so the common message never loads either library.
export async function hydrate(root) {
  if (!root) return;
  await Promise.all([hydrateMath(root), hydrateDiagrams(root)]);
}

export { esc as escapeHtml };
