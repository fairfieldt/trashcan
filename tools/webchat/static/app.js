// The trashcan demo chat page. server.py runs the conversations and their tools; this page shows them. It keeps one
// event stream open (GET /api/events?session=ID): a snapshot of the open conversation, then every change as it
// happens, with sequence numbers, so a dropped connection (a phone locking, a network change) resumes where it
// stopped. The look is piPE's "pipe 3.1" skin: its stylesheets and markdown renderer are in pipe/.
import { renderMarkdown, hydrate, codeBlock, escapeHtml as esc } from './pipe/md.js';
import { ansiToHtml, hasAnsi } from './pipe/ansi.js';

const $ = (id) => document.getElementById(id);
const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html !== undefined) e.innerHTML = html; return e; };
const fmtTok = (n) => (n >= 1000 ? `${(n / 1000).toFixed(n >= 10000 ? 0 : 1)}k` : String(n | 0));
const fmtTime = (ms) => new Date(ms).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
const ago = (ms) => { const s = (Date.now() - ms) / 1000; if (s < 60) return 'now'; if (s < 3600) return `${(s / 60) | 0}m`; if (s < 86400) return `${(s / 3600) | 0}h`; return `${(s / 86400) | 0}d`; };
const oneLine = (s, max = 140) => { const t = String(s == null ? '' : s).replace(/\s+/g, ' ').trim(); return t.length > max ? t.slice(0, max - 1) + '…' : t; };
const touch = matchMedia('(hover: none) and (pointer: coarse)').matches;
const narrow = () => matchMedia('(max-width: 1100px)').matches;
const store = {
  get: (k, d) => { try { const v = localStorage.getItem('webchat.' + k); return v === null ? d : v; } catch (e) { return d; } },
  set: (k, v) => { try { localStorage.setItem('webchat.' + k, v); } catch (e) { /* private mode */ } },
};

const state = {
  models: [], levels: ['off', 'low', 'medium', 'high'],
  status: null,
  sessions: [], listVersion: -1,
  current: null,            // the open conversation (the server's public view), or null for a new one
  thinking: store.get('thinking', 'medium'),
  open: new Map(),          // which cards are open, by key, kept across re-renders
  pinned: true,
};

// ------------------------------------------------------------------------------------------------- viewport
// iOS keeps the layout viewport when the keyboard opens and scrolls it instead; the app follows the visual viewport
// so the composer stays right above the keyboard and nothing scrolls away.
const root = document.documentElement;
function fitViewport() {
  const vv = window.visualViewport;
  if (!vv) return;
  root.style.setProperty('--vvh', `${vv.height}px`);
  root.style.setProperty('--vvt', `${vv.offsetTop}px`);
  root.toggleAttribute('data-kb', touch && innerHeight - vv.height > 120);
  if (state.pinned) scrollIfPinned();
}
if (window.visualViewport) {
  visualViewport.addEventListener('resize', fitViewport);
  visualViewport.addEventListener('scroll', fitViewport);
}
if (touch) root.dataset.touch = '1';

// ---------------------------------------------------------------------------------------------------- toasts
function toast(message, level = 'info', ms = 4500) {
  const t = el('div', `toast ${level}`);
  t.textContent = message;
  $('toasts').appendChild(t);
  setTimeout(() => { t.style.opacity = '0'; setTimeout(() => t.remove(), 250); }, ms);
}

// ------------------------------------------------------------------------------------------- scroll pinning
// Follows the reply while the reader is at the bottom; scrolling up (wheel, finger, fling) lets go, and the
// "latest" pill or scrolling back down takes it up again. (Adapted from piPE's app.js.)
const stream = $('stream');
const hero = $('hero');
let scrollFrame = 0, scrollGesture = false, userScrollUntil = 0, touching = false, flingUntil = 0, lastTop = 0;
function updatePinned() {
  const gap = stream.scrollHeight - stream.scrollTop - stream.clientHeight;
  state.pinned = gap < 80;
  $('scroll-down').classList.toggle('hidden', state.pinned);
}
function unpin() { state.pinned = false; $('scroll-down').classList.remove('hidden'); }
stream.addEventListener('scroll', () => {
  const top = stream.scrollTop, up = top < lastTop;
  lastTop = top;
  const fingers = touching || performance.now() < flingUntil;
  if (!scrollGesture && !fingers && performance.now() > userScrollUntil) return;
  if (up && fingers) { unpin(); return; }
  updatePinned();
});
function releaseAutoScroll() { if (scrollFrame) cancelAnimationFrame(scrollFrame); scrollFrame = 0; }
stream.addEventListener('wheel', (e) => { releaseAutoScroll(); userScrollUntil = performance.now() + 250; if (e.deltaY < 0) unpin(); }, { passive: true });
stream.addEventListener('pointerdown', () => { scrollGesture = true; releaseAutoScroll(); });
addEventListener('pointerup', () => { if (!scrollGesture) return; updatePinned(); scrollGesture = false; if (state.pinned) scrollIfPinned(); });
addEventListener('pointercancel', () => { scrollGesture = false; });
stream.addEventListener('touchstart', () => { touching = true; flingUntil = 0; lastTop = stream.scrollTop; releaseAutoScroll(); }, { passive: true });
function touchDone() { if (!touching) return; touching = false; flingUntil = performance.now() + 1500; updatePinned(); if (state.pinned) scrollIfPinned(); }
addEventListener('touchend', touchDone, { passive: true });
addEventListener('touchcancel', touchDone, { passive: true });
addEventListener('keydown', (e) => { if (['PageUp', 'ArrowUp', 'Home'].includes(e.key) && document.activeElement === stream) unpin(); });
let holdScroll = false;
function scrollIfPinned() {
  if (!state.pinned || scrollGesture || touching || holdScroll) return;
  stream.scrollTop = stream.scrollHeight;
  if (scrollFrame) cancelAnimationFrame(scrollFrame);
  scrollFrame = requestAnimationFrame(() => { scrollFrame = 0; if (state.pinned && !holdScroll) stream.scrollTop = stream.scrollHeight; });
}
function pinToBottom() { state.pinned = true; $('scroll-down').classList.add('hidden'); scrollIfPinned(); }
$('scroll-down').addEventListener('click', pinToBottom);
new ResizeObserver(scrollIfPinned).observe(stream);
new MutationObserver(scrollIfPinned).observe(stream, { childList: true, subtree: true, characterData: true });
stream.addEventListener('load', scrollIfPinned, true);
// Opening or closing a card keeps its summary line where it was (no jump to the bottom while following).
let toggling = null;
function armToggle(sum) {
  toggling = { sum, top: sum.getBoundingClientRect().top };
  releaseAutoScroll();
  holdScroll = true;
  setTimeout(() => { if (toggling && toggling.sum === sum) { toggling = null; holdScroll = false; } }, 800);
}
stream.addEventListener('pointerdown', (e) => { const sum = e.target.closest && e.target.closest('summary'); if (sum) armToggle(sum); });
stream.addEventListener('keydown', (e) => { if (e.key !== 'Enter' && e.key !== ' ') return; const sum = e.target.closest && e.target.closest('summary'); if (sum) armToggle(sum); });
stream.addEventListener('toggle', (e) => {
  const d = e.target;
  if (d.dataset && d.dataset.key) state.open.set(d.dataset.key, d.open);
  flingUntil = 0;
  if (!toggling || toggling.sum.parentElement !== d) return;
  const { sum, top } = toggling;
  toggling = null;
  const delta = sum.getBoundingClientRect().top - top;
  if (delta) stream.scrollTop += delta;
  requestAnimationFrame(() => { holdScroll = false; updatePinned(); });
}, true);

// ---------------------------------------------------------------------------------------------- the transcript
const modelOf = (id) => state.models.find((m) => m.id === id);
const turnEl = (id) => stream.querySelector(`.turn[data-turn="${CSS.escape(id)}"]`);

function details(cls, key, defaultOpen, summary) {
  const d = el('details', cls);
  d.dataset.key = key;
  d.open = state.open.has(key) ? state.open.get(key) : defaultOpen;
  d.innerHTML = summary;
  return d;
}

// A text step in a round that also called a tool is narration of the work; otherwise it is (part of) the answer.
function stepKinds(turn) {
  const toolRounds = new Set(turn.steps.filter((s) => s && s.kind === 'tool').map((s) => s.round));
  return turn.steps.map((s) => (s && s.kind === 'text' && toolRounds.has(s.round) ? 'note' : s && s.kind));
}

function proseHtml(text, live) {
  return renderMarkdown(text || '', live ? { highlight: false } : undefined);
}

function thinkEl(turn, i, step) {
  const d = details('think', `${turn.id}:${i}`, true, `<summary><span class="eyebrow">thinking</span><span class="muted small think-meta"></span></summary><div class="think-body"><div class="prose"></div></div>`);
  d.dataset.step = i;
  d.querySelector('.prose').innerHTML = proseHtml(step.text, !step.done);
  if (!step.done) d.querySelector('.prose').classList.add('caret');
  return d;
}

function noteEl(i, step) {
  const d = el('div', 'act-note', `<span class="eyebrow pi">▸</span><div class="prose"></div>`);
  d.dataset.step = i;
  d.querySelector('.prose').innerHTML = proseHtml(step.text, !step.done);
  return d;
}

function toolSummary(step) {
  const a = step.arguments || {};
  if (step.name === 'bash') return oneLine(a.command);
  if (step.name === 'web_search') return oneLine(a.query);
  if (step.name === 'fetch_page') return oneLine(String(a.url || '').replace(/^https?:\/\//, ''));
  return oneLine(JSON.stringify(a));
}

function argsHtml(step) {
  const a = step.arguments || {};
  if (step.name === 'bash') return ['command', `<div class="cmd"><span class="cmd-prompt" aria-hidden="true">$</span>${codeBlock(String(a.command ?? ''), { lang: 'bash', fold: false })}</div>`];
  if (step.name === 'web_search') return ['query', `<div class="query">${esc(a.query ?? '')}</div>`];
  if (step.name === 'fetch_page') {
    const url = String(a.url ?? '');
    const link = /^https?:\/\//.test(url) ? `<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${esc(url)}</a>` : esc(url);
    return ['page', `<div class="query">${link}</div>`];
  }
  return ['arguments', codeBlock(JSON.stringify(a, null, 2), { lang: 'json', fold: false })];
}

const pre = (text, term) => (hasAnsi(text) ? `<pre class="out-pre term">${ansiToHtml(text)}</pre>` : `<pre class="out-pre${term ? ' term' : ''}">${esc(text)}</pre>`);

function resultHtml(step) {
  const r = step.result;
  if (step.state === 'stopped') return ['', `<div class="out-foot muted small">Stopped.</div>`];
  if (!r || r.error) return ['error', pre(r && r.error ? r.error : step.output || '(no output)')];
  if (step.name === 'bash') {
    const code = r.exitCode;
    return ['output', pre(r.output || '(no output)', true) + (code ? `<div class="out-foot"><span class="chip bad">exit code ${code}</span></div>` : '')];
  }
  if (step.name === 'web_search') {
    const rs = r.results || [];
    if (!rs.length) return ['results', `<div class="out-foot muted small">No results.</div>`];
    return ['results', `<ol class="results">${rs.map((x) => `<li><a href="${esc(x.url)}" target="_blank" rel="noopener noreferrer">${esc(x.title || x.url)}</a><span class="url">${esc(x.url)}</span>${x.snippet ? `<span class="snip">${esc(x.snippet)}</span>` : ''}</li>`).join('')}</ol>`];
  }
  if (step.name === 'fetch_page') return ['page text', pre(r.text || '(empty page)') + (r.truncated ? `<div class="out-foot muted small">First 8,000 characters.</div>` : '')];
  return ['output', pre(step.output || '')];
}

const STATE_TEXT = { pending: '', running: 'running…', done: '', error: 'failed', stopped: 'stopped' };
function toolEl(turn, i, step) {
  const open = step.state === 'error' || step.name === 'bash' || step.name === 'web_search';
  const d = details('tool', `${turn.id}:${i}`, open,
    `<summary class="tool-head"><svg class="i chev"><use href="#i-chev"/></svg><span class="tool-name"></span><span class="tool-summary"></span><span class="tool-state"></span></summary><div class="tool-body"></div>`);
  d.dataset.step = i;
  d.dataset.state = step.state === 'pending' ? 'running' : step.state;
  d.querySelector('.tool-name').textContent = step.name;
  d.querySelector('.tool-summary').textContent = toolSummary(step);
  const code = step.name === 'bash' && step.result && step.result.exitCode;
  d.querySelector('.tool-state').textContent = code && step.state === 'error' ? `exit ${code}` : STATE_TEXT[step.state] || '';
  const body = d.querySelector('.tool-body');
  const [argLabel, args] = argsHtml(step);
  body.appendChild(el('div', '', `<span class="eyebrow">${argLabel}</span><div class="args">${args}</div>`));
  if (step.state !== 'pending' && step.state !== 'running') {
    const [label, out] = resultHtml(step);
    body.appendChild(el('div', 'result', `${label ? `<span class="eyebrow">${label}</span>` : ''}<div class="out">${out}</div>`));
  }
  return d;
}

function stepSummary(step, kind) {
  if (kind === 'tool') return `${step.name} · ${toolSummary(step)}`;
  return oneLine(step.text);
}

// Speed and tokens for a reply: decode rate, prompt tokens processed over all its rounds, tokens generated, and how
// many prompt tokens the engine reused from its cache. While it streams, an estimate from the streamed tokens.
function statsText(turn) {
  const n = (v) => Number(v || 0).toLocaleString();
  if (turn.status === 'running' && turn.live) {
    return `${turn.live.tps.toFixed(1)} tok/s · ${n(turn.live.output)} out so far`;
  }
  const st = turn.stats;
  if (!st) return '';
  const parts = [];
  if (st.tps) parts.push(`${st.tps.toFixed(1)} tok/s`);
  parts.push(`${n(st.input)} in`, `${n(st.output)} out`);
  if (st.input) parts.push(`${Math.round((100 * st.cached) / st.input)}% cached`);
  return parts.join(' · ');
}

function statsEl(turn) {
  const text = statsText(turn);
  const d = el('div', 'turn-stats mono');
  d.textContent = text;
  if (turn.status === 'running') d.classList.add('live');
  d.hidden = !text;
  return d;
}

function renderTurn(turn) {
  let node = turnEl(turn.id);
  if (!node) {
    node = el('div', 'turn');
    node.dataset.turn = turn.id;
    stream.appendChild(node);
  }
  const live = turn.status === 'running';
  const parts = [];

  const user = el('div', 'msg msg-user', `<div class="msg-head"><span class="eyebrow you">you</span><time>${fmtTime(turn.started)}</time><button class="msg-copy" type="button" title="Copy" aria-label="Copy message"><svg class="i"><use href="#i-copy"/></svg></button></div>`);
  user.__md = turn.user;
  const bubble = el('div', 'bubble');
  bubble.textContent = turn.user;
  user.appendChild(bubble);
  parts.push(user);

  const kinds = stepKinds(turn);
  const work = [], answer = [];
  turn.steps.forEach((s, i) => { if (s) (kinds[i] === 'text' ? answer : work).push(i); });
  if (work.length || (live && !answer.length)) {
    const act = details('activity', `${turn.id}:act`, true,
      `<summary class="activity-head"><svg class="i chev"><use href="#i-chev"/></svg><span class="eyebrow">working</span><span class="activity-summary"></span><span class="activity-count"></span></summary><div class="activity-body"></div>`);
    if (live) act.dataset.live = '1';
    const body = act.querySelector('.activity-body');
    let summary = '', note = '';
    for (const i of work) {
      const s = turn.steps[i];
      body.appendChild(kinds[i] === 'thinking' ? thinkEl(turn, i, s) : kinds[i] === 'note' ? noteEl(i, s) : toolEl(turn, i, s));
      if (kinds[i] === 'note' && s.text.trim()) note = oneLine(s.text);
      summary = stepSummary(s, kinds[i]) || summary;
    }
    // the collapsed line: the model's latest words about its work, else the latest item
    summary = note || summary;
    if (turn.waiting) summary = 'waiting for another conversation to finish';
    else if (live && !turn.steps.length) summary = 'waiting for the model';
    act.querySelector('.activity-summary').textContent = summary;
    act.querySelector('.activity-count').textContent = work.length ? String(work.length) : '';
    parts.push(act);
  }

  if (answer.length) {
    const m = el('div', 'msg msg-assistant');
    m.appendChild(el('div', 'msg-head', `<span class="eyebrow pi">answer</span><span class="muted small mono">${esc(turn.model || '')}</span><time>${fmtTime(turn.ended || turn.started)}</time><button class="msg-copy" type="button" title="Copy" aria-label="Copy answer"><svg class="i"><use href="#i-copy"/></svg></button>`));
    m.__md = answer.map((i) => turn.steps[i].text).join('\n\n');
    for (const i of answer) {
      const s = turn.steps[i];
      const p = el('div', 'prose', proseHtml(s.text, !s.done));
      p.dataset.step = i;
      if (!s.done) p.classList.add('caret');
      m.appendChild(p);
    }
    parts.push(m);
  }

  if (turn.stats || turn.live) parts.push(statsEl(turn));
  if (turn.status === 'stopped') parts.push(el('div', 'msg-system', 'Stopped.'));
  if (turn.error) {
    const warn = turn.error.kind === 'length';
    const row = el('div', `msg-system ${warn ? 'warn' : 'err'}`);
    row.appendChild(document.createTextNode(turn.error.message));
    if (['down', 'upstream', 'interrupted', 'internal'].includes(turn.error.kind)) {
      const b = el('button', 'btn btn-sm retry', 'send again');
      b.type = 'button';
      b.addEventListener('click', () => send(turn.user));
      row.appendChild(b);
    } else if (turn.error.kind === 'context') {
      const b = el('button', 'btn btn-sm retry', 'new conversation');
      b.type = 'button';
      b.addEventListener('click', () => openSession(null));
      row.appendChild(b);
    }
    parts.push(row);
  }
  node.replaceChildren(...parts);
  if (!live) hydrate(node);
  hero.classList.add('hidden');
  tickElapsed();
}

// Streaming text: only the step's own element is redrawn, once per frame at most.
const dirty = new Map();
let deltaFrame = 0;
function markDirty(turn, i) {
  dirty.set(`${turn.id}:${i}`, [turn, i]);
  if (!deltaFrame) deltaFrame = requestAnimationFrame(flushDeltas);
}
function flushDeltas() {
  deltaFrame = 0;
  for (const [turn, i] of dirty.values()) {
    const node = turnEl(turn.id);
    const step = turn.steps[i];
    const box = node && node.querySelector(`[data-step="${i}"]`);
    const prose = box && (box.classList.contains('prose') ? box : box.querySelector('.prose'));
    if (!prose || !step) { renderTurn(turn); continue; }
    prose.innerHTML = proseHtml(step.text, true);
    const msg = prose.closest('.msg');
    if (msg) msg.__md = turn.steps.filter((s, j) => stepKinds(turn)[j] === 'text').map((s) => s.text).join('\n\n');
    const sum = node.querySelector('.activity-summary');
    if (sum && box.closest('.activity') && step.text.trim() && stepKinds(turn)[i] === 'note') sum.textContent = oneLine(step.text);
  }
  dirty.clear();
  scrollIfPinned();
}

// The live working section counts the seconds while the model has not said anything yet (a long prompt can take
// minutes to read).
let elapsedTimer = 0;
function tickElapsed() {
  const t = state.current && state.current.turns[state.current.turns.length - 1];
  const waiting = t && t.status === 'running' && !t.steps.length;
  if (waiting && !elapsedTimer) elapsedTimer = setInterval(tickElapsed, 1000);
  if (!waiting) { clearInterval(elapsedTimer); elapsedTimer = 0; return; }
  const c = turnEl(t.id) && turnEl(t.id).querySelector('.activity-count');
  if (c) c.textContent = `${Math.max(0, Math.round((Date.now() - t.started) / 1000))} s`;
}

function renderAll() {
  stream.querySelectorAll('.turn').forEach((n) => n.remove());
  const turns = state.current ? state.current.turns : [];
  hero.classList.toggle('hidden', turns.length > 0);
  for (const t of turns) renderTurn(t);
  pinToBottom();
}

stream.addEventListener('click', (e) => {
  const btn = e.target.closest('.copy, .msg-copy');
  if (btn) {
    e.preventDefault();
    let text = '';
    if (btn.classList.contains('msg-copy')) text = btn.closest('.msg').__md || '';
    else {
      const code = btn.closest('pre.code') && btn.closest('pre.code').querySelector('code');
      const lines = code ? code.querySelectorAll(':scope > .code-line') : [];
      text = lines.length ? [...lines].map((l) => l.textContent).join('\n') : (code ? code.textContent : '');
    }
    if (!text) return;
    copyText(text).then(() => { btn.classList.add('copied'); setTimeout(() => btn.classList.remove('copied'), 1200); }, () => toast('Could not copy.', 'error'));
    return;
  }
  const more = e.target.closest('.code-more');
  if (more) {
    const box = more.closest('.code-fold');
    box.classList.toggle('open');
    more.textContent = box.classList.contains('open') ? `fold ${box.dataset.lines} lines` : `show all ${box.dataset.lines} lines`;
    return;
  }
  if (e.target.closest('[data-file]')) e.preventDefault();   // a relative link in an answer: nothing to open here
});

function copyText(text) {
  if (navigator.clipboard && window.isSecureContext) return navigator.clipboard.writeText(text);
  return new Promise((resolve, reject) => {
    const t = el('textarea');
    t.value = text;
    t.style.cssText = 'position:fixed;opacity:0';
    document.body.appendChild(t);
    t.select();
    const ok = document.execCommand('copy');
    t.remove();
    (ok ? resolve : reject)();
  });
}

// -------------------------------------------------------------------------------------------------- events
const conn = { es: null, sid: null, last: 0, lastMsg: 0, retry: 0, timer: 0, downSince: 0 };

function connect(sid) {
  if (conn.es) conn.es.close();
  clearTimeout(conn.timer);
  conn.sid = sid;
  const after = sid && state.current && state.current.id === sid ? state.current.seq : 0;
  const es = new EventSource(`/api/events?${sid ? `session=${sid}&` : ''}after=${after}`);
  conn.es = es;
  conn.lastMsg = Date.now();
  const on = (name, fn) => es.addEventListener(name, (e) => {
    if (conn.es !== es) return;
    conn.lastMsg = Date.now();
    conn.retry = 0;
    if (conn.downSince) { conn.downSince = 0; renderBanner(); }
    let data;
    try { data = JSON.parse(e.data); } catch (err) { return; }
    fn(data, e);
  });
  on('snapshot', (s) => {
    const same = state.current && state.current.id === s.id;
    state.current = s;
    renderAll();
    if (!same) renderSessions();
    renderHeader();
  });
  on('session', (d) => {
    if (!state.current) return;
    Object.assign(state.current, ...['title', 'context_tokens', 'context_model', 'updated'].filter((k) => k in d).map((k) => ({ [k]: d[k] })));
    state.current.seq = d.seq;
    renderHeader();
  });
  on('turn', (d) => {
    const s = state.current;
    if (!s) return;
    s.seq = d.seq;
    const i = s.turns.findIndex((t) => t.id === d.turn.id);
    if (i >= 0) s.turns[i] = d.turn; else s.turns.push(d.turn);
    s.running = d.turn.status === 'running' || s.turns.some((t) => t.status === 'running');
    renderTurn(d.turn);
    renderHeader();
  });
  on('step', (d) => {
    const t = state.current && state.current.turns.find((x) => x.id === d.turn);
    if (!t) return;
    state.current.seq = d.seq;
    t.steps[d.index] = d.step;
    renderTurn(t);
  });
  on('delta', (d) => {
    const t = state.current && state.current.turns.find((x) => x.id === d.turn);
    if (!t || !t.steps[d.index]) return;
    state.current.seq = d.seq;
    t.steps[d.index].text += d.text;
    markDirty(t, d.index);
  });
  on('stats', (d) => {
    const t = state.current && state.current.turns.find((x) => x.id === d.turn);
    if (!t) return;
    state.current.seq = d.seq;
    t.live = d.live;
    if (d.stats) t.stats = d.stats;
    const node = turnEl(t.id);
    if (!node) return;
    let line = node.querySelector('.turn-stats');
    if (!line) { line = statsEl(t); node.appendChild(line); return; }
    line.textContent = statsText(t);
    line.hidden = !line.textContent;
    line.classList.toggle('live', t.status === 'running');
  });
  on('status', (st) => applyStatus(st));
  on('ping', () => {});
  on('deleted', () => {
    toast('This conversation was deleted.');
    openSession(null);
  });
  es.onopen = () => { conn.lastMsg = Date.now(); };
  es.onerror = () => {
    if (conn.es !== es) return;
    if (!conn.downSince) { conn.downSince = Date.now(); setTimeout(renderBanner, 3000); }
    // EventSource retries by itself (with Last-Event-ID) unless the server answered with an error.
    if (es.readyState === EventSource.CLOSED) reconnectSoon();
  };
}

function reconnectSoon() {
  clearTimeout(conn.timer);
  const delay = Math.min(10000, 1000 * 2 ** conn.retry++);
  conn.timer = setTimeout(async () => {
    if (conn.sid) {
      // A conversation deleted elsewhere answers 404: show a new one instead of retrying forever.
      try {
        const r = await fetch(`/api/sessions/${conn.sid}`, { cache: 'no-store' });
        if (r.status === 404) { toast('This conversation was deleted.'); openSession(null); return; }
      } catch (e) { /* server unreachable: try again */ }
    }
    connect(conn.sid);
  }, delay);
}

// A phone coming back from the background may hold a connection that silently died: anything quiet for longer
// than the server's 15 s keepalive is reopened, from the last event seen.
setInterval(() => { if (conn.es && Date.now() - conn.lastMsg > 40000) connect(conn.sid); }, 5000);
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState !== 'visible') return;
  if (!conn.es || conn.es.readyState === EventSource.CLOSED || Date.now() - conn.lastMsg > 20000) connect(conn.sid);
  refreshSessions();
});
addEventListener('online', () => connect(conn.sid));
addEventListener('pageshow', (e) => { if (e.persisted) connect(conn.sid); });

// ------------------------------------------------------------------------------------------ server status
let lastSwitch = null;
function applyStatus(st) {
  const prev = state.status;
  state.status = st;
  if (st.list_version !== state.listVersion) refreshSessions();
  else markRunning();
  const sw = st.switch;
  if (prev && prev.switch.state === 'switching' && sw.state === 'idle' && st.model.state === 'ready') toast(`${st.model.name} is ready.`, 'ok');
  lastSwitch = sw;
  renderHeader();
  renderModels();
  renderBanner();
}

function renderHeader() {
  const st = state.status;
  const name = $('model-name');
  if (!st) name.textContent = '…';
  else if (st.switch.state === 'switching') name.textContent = `starting ${modelOf(st.model.id) ? modelOf(st.model.id).name : ''}`;
  else if (st.model.state === 'down' || !st.model.id) name.textContent = 'no model';
  else name.textContent = st.model.name || st.model.id;
  const s = state.current;
  $('title').textContent = s && s.title ? s.title : '';
  document.title = s && s.title ? `${s.title} · trashcan chat` : 'trashcan chat';
  // context meter: what the last request used, against the model it went to
  const ctx = $('ctx');
  const m = s && s.context_tokens ? (modelOf(st && st.model.id) || modelOf(s.context_model)) : null;
  if (m) {
    const used = s.context_tokens, pct = Math.min(100, (used / m.context) * 100);
    $('ctx-fill').style.width = `${pct}%`;
    $('ctx-fill').dataset.level = pct > 90 ? 'hot' : pct > 70 ? 'warn' : '';
    $('ctx-text').textContent = `${fmtTok(used)} / ${fmtTok(m.context)}`;
    ctx.title = `Context used: ${used.toLocaleString()} of ${m.context.toLocaleString()} tokens`;
  }
  ctx.classList.toggle('hidden', !m);
  const running = !!(s && s.turns.some((t) => t.status === 'running'));
  $('send').classList.toggle('hidden', running);
  $('abort').classList.toggle('hidden', !running);
  $('send').disabled = !!(st && st.switch.state === 'switching');
  $('composer').toggleAttribute('data-streaming', running);
}

function renderBanner() {
  const b = $('banner'), st = state.status;
  let html = '', cls = '';
  if (conn.downSince && Date.now() - conn.downSince >= 2500) {
    html = 'Reconnecting to the chat server…';
    cls = 'warn';
  } else if (st && st.switch.state === 'switching') {
    const m = modelOf(st.model.id);
    const secs = Math.round(Date.now() / 1000 - st.switch.since);
    html = `Starting ${esc(m ? m.name : 'the model')}: ${esc(st.switch.phase || 'starting')} <span class="muted">${Math.floor(secs / 60)}:${String(secs % 60).padStart(2, '0')}</span>`;
    cls = 'warn';
  } else if (st && st.switch.state === 'error' && !bannerDismissed(st.switch.error)) {
    html = `${esc(st.switch.error)} <button class="btn btn-sm" type="button" data-act="dismiss">ok</button>`;
    cls = 'err';
  } else if (st && st.model.state === 'down') {
    html = 'No model is running. <button class="btn btn-sm" type="button" data-act="models">choose a model</button>';
    cls = 'err';
  }
  b.className = `banner msg-system ${cls}${html ? '' : ' hidden'}`;
  if (b.__html !== html) { b.innerHTML = html; b.__html = html; }
}
let dismissed = '';
const bannerDismissed = (msg) => msg === dismissed;
$('banner').addEventListener('click', (e) => {
  const act = e.target.closest('[data-act]');
  if (!act) return;
  if (act.dataset.act === 'dismiss') { dismissed = state.status.switch.error; renderBanner(); }
  if (act.dataset.act === 'models') openModels();
});
setInterval(() => { if (state.status && state.status.switch.state === 'switching') renderBanner(); }, 1000);

// ---------------------------------------------------------------------------------------------- the model picker
function renderModels() {
  const ul = $('model-list'), st = state.status;
  ul.replaceChildren();
  for (const m of state.models) {
    const current = st && st.model.id === m.id;
    const starting = current && st.switch.state === 'switching';
    const li = el('li', '', `<span class="m-id"></span>${starting ? '<span class="m-badge">starting</span>' : current && st.model.state === 'ready' ? '<span class="m-badge">running</span>' : '<span></span>'}<span class="m-meta"></span>`);
    li.querySelector('.m-id').textContent = m.name;
    li.querySelector('.m-meta').textContent = `${m.macs} Macs · ${m.context.toLocaleString()} tokens of context`;
    li.dataset.current = String(!!current);
    li.tabIndex = 0;
    li.setAttribute('role', 'button');
    li.addEventListener('click', () => pickModel(m));
    li.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pickModel(m); } });
    ul.appendChild(li);
  }
  const p = $('model-status');
  let text = '', cls = '';
  if (st && st.switch.state === 'switching') text = `${st.switch.phase || 'Starting'}. This takes 1 to 3 minutes, longer when the Macs are busy.`;
  else if (st && st.switch.state === 'error') { text = st.switch.error; cls = 'err'; }
  else if (st && st.model.state === 'down') text = 'No model is running. Choose one to start it.';
  else if (st && st.running.length) text = 'Switching is possible once the running reply ends.';
  p.textContent = text;
  p.className = `pop-status small ${cls}${text ? '' : ' hidden'}`;
}

async function pickModel(m) {
  const st = state.status;
  if (!st) return;
  if (st.switch.state === 'switching') { toast('A model is already starting.'); return; }
  if (st.model.id === m.id && st.model.state === 'ready') { closeModels(); return; }
  const ok = await confirmDialog(`Start ${m.name}?`, `It runs on ${m.macs} Macs and takes 1 to 3 minutes to start. Chat is unavailable until it is ready.`, 'start');
  if (!ok) return;
  const r = await fetch('/api/switch', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ recipe: m.recipe }) }).catch(() => null);
  if (!r) { toast('Could not reach the chat server.', 'error'); return; }
  const j = await r.json().catch(() => ({}));
  if (!r.ok) { toast(j.error || `Could not switch (HTTP ${r.status}).`, 'error'); return; }
  applyStatus(j);
}

const pop = $('model-pop');
function openModels() {
  renderModels();
  pop.classList.add('open');
  showScrim('models');
  $('model-chip').setAttribute('aria-expanded', 'true');
}
function closeModels() {
  pop.classList.remove('open');
  $('model-chip').setAttribute('aria-expanded', 'false');
  hideScrim('models');
}
$('model-chip').addEventListener('click', () => (pop.classList.contains('open') ? closeModels() : openModels()));
$('model-close').addEventListener('click', closeModels);

// ------------------------------------------------------------------------------------ drawer, scrim, dialog
const app = $('app');
const scrimFor = new Set();
function showScrim(who) { scrimFor.add(who); $('scrim').classList.remove('hidden'); $('scrim').dataset.for = [...scrimFor].join(' '); }
function hideScrim(who) { scrimFor.delete(who); if (!scrimFor.size) $('scrim').classList.add('hidden'); }
$('scrim').addEventListener('click', () => { closeModels(); setDrawer(false); });

function setDrawer(open) {
  app.dataset.left = open ? 'open' : 'closed';
  $('toggle-left').setAttribute('aria-expanded', String(open));
  if (narrow()) { if (open) showScrim('drawer'); else hideScrim('drawer'); }
  else { hideScrim('drawer'); store.set('left', open ? 'open' : 'closed'); }
}
setDrawer(narrow() ? false : store.get('left', 'open') === 'open');
matchMedia('(max-width: 1100px)').addEventListener('change', () => setDrawer(narrow() ? false : store.get('left', 'open') === 'open'));
$('toggle-left').addEventListener('click', () => setDrawer(app.dataset.left !== 'open'));

function confirmDialog(title, body, okLabel) {
  const d = $('confirm-dialog');
  $('confirm-title').textContent = title;
  $('confirm-body').textContent = body;
  $('confirm-ok').textContent = okLabel;
  d.returnValue = '';
  d.showModal();
  return new Promise((resolve) => d.addEventListener('close', () => resolve(d.returnValue === 'ok'), { once: true }));
}

addEventListener('keydown', (e) => {
  if (e.key !== 'Escape') return;
  if (pop.classList.contains('open')) { closeModels(); return; }
  if (narrow() && app.dataset.left === 'open') { setDrawer(false); return; }
  if (isRunning() && !$('confirm-dialog').open) { e.preventDefault(); stop(); }
});

// ------------------------------------------------------------------------------------------- conversations
let listBusy = null;
async function refreshSessions() {
  if (listBusy) return listBusy;
  listBusy = (async () => {
    try {
      const r = await fetch('/api/sessions', { cache: 'no-store' });
      const j = await r.json();
      state.sessions = j.sessions;
      state.listVersion = j.list_version;
      renderSessions();
    } catch (e) { /* the event stream reports the server being away */ }
    listBusy = null;
  })();
  return listBusy;
}

function renderSessions() {
  const ul = $('sessions');
  ul.replaceChildren();
  if (!state.sessions.length) { ul.appendChild(el('li', 'empty', 'No conversations yet.')); return; }
  for (const s of state.sessions) {
    const li = el('li', '', `<span class="s-title"></span><span class="s-meta"></span>`);
    li.dataset.id = s.id;
    li.querySelector('.s-title').textContent = s.title;
    li.setAttribute('aria-selected', String(!!(state.current && state.current.id === s.id)));
    li.tabIndex = 0;
    const del = el('button', 'iconbtn s-del', '<svg class="i" aria-hidden="true"><use href="#i-x"/></svg>');
    del.type = 'button';
    del.title = 'Delete';
    del.setAttribute('aria-label', `Delete ${s.title}`);
    del.addEventListener('click', (e) => { e.stopPropagation(); deleteSession(s); });
    li.appendChild(del);
    li.addEventListener('click', () => { openSession(s.id); if (narrow()) setDrawer(false); });
    li.addEventListener('keydown', (e) => { if (e.key === 'Enter') { openSession(s.id); if (narrow()) setDrawer(false); } });
    ul.appendChild(li);
  }
  markRunning();
}

function markRunning() {
  const running = new Set(state.status ? state.status.running : []);
  for (const li of $('sessions').querySelectorAll('li[data-id]')) {
    const s = state.sessions.find((x) => x.id === li.dataset.id);
    if (!s) continue;
    const live = running.has(s.id);
    li.dataset.live = live ? 'running' : '';
    li.querySelector('.s-meta').textContent = live ? 'answering' : `${s.turns} message${s.turns === 1 ? '' : 's'} · ${ago(s.updated)}${s.status === 'error' ? ' · error' : ''}`;
  }
}
setInterval(markRunning, 60000);

function openSession(id) {
  if (state.current && id === state.current.id) return;
  state.open.clear();
  state.current = null;
  store.set('current', id || 'new');
  renderAll();
  hero.classList.toggle('hidden', !!id);
  renderHeader();
  renderSessions();
  connect(id);
  if (!id && !touch) $('composer-input').focus();
}

async function deleteSession(s) {
  const ok = await confirmDialog('Delete this conversation?', s.title, 'delete');
  if (!ok) return;
  const r = await fetch(`/api/sessions/${s.id}`, { method: 'DELETE' }).catch(() => null);
  if (!r || (!r.ok && r.status !== 404)) { toast('Could not delete it.', 'error'); return; }
  if (state.current && state.current.id === s.id) openSession(null);
  refreshSessions();
}

$('new-session').addEventListener('click', () => { openSession(null); if (narrow()) setDrawer(false); });
$('new-thread').addEventListener('click', () => openSession(null));

// ------------------------------------------------------------------------------------------------ composer
const input = $('composer-input');
function autosize() {
  if (CSS.supports('field-sizing: content')) return;
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, innerHeight * 0.35) + 'px';
}
input.addEventListener('input', autosize);
input.addEventListener('keydown', (e) => {
  // A phone's return key writes a new line; the send button sends. A keyboard's Enter sends, Shift+Enter breaks.
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing && !touch) { e.preventDefault(); $('composer').requestSubmit(); }
});
$('composer').addEventListener('submit', (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  send(text);
});
$('abort').addEventListener('click', stop);
const isRunning = () => !!(state.current && state.current.turns.some((t) => t.status === 'running'));

let sending = false;
async function send(text) {
  if (sending) return;
  const st = state.status;
  if (st && st.switch.state === 'switching') { toast('A model is starting. Send again when it is ready.'); return; }
  if (isRunning()) { toast('A reply is still running. Stop it first.'); return; }
  sending = true;
  const before = input.value;
  input.value = '';
  autosize();
  pinToBottom();
  try {
    const s = state.current;
    const body = JSON.stringify({ text, thinking: state.thinking });
    const r = await fetch(s ? `/api/sessions/${s.id}/messages` : '/api/sessions', { method: 'POST', headers: { 'content-type': 'application/json' }, body });
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j.error || `HTTP ${r.status}`);
    if (j.error) throw new Error(j.error);
    if (!s) {
      state.current = j.session;
      store.set('current', j.session.id);
      renderAll();
      renderHeader();
      connect(j.session.id);
      refreshSessions();
    }
  } catch (e) {
    if (!input.value) { input.value = before === '' ? text : before; autosize(); }
    toast(e.message === 'Failed to fetch' || e.message === 'Load failed' ? 'Could not reach the chat server. Try again.' : e.message, 'error');
  } finally {
    sending = false;
  }
}

async function stop() {
  const s = state.current;
  if (!s) return;
  const r = await fetch(`/api/sessions/${s.id}/stop`, { method: 'POST' }).catch(() => null);
  if (!r) toast('Could not reach the chat server.', 'error');
}

function renderThinking() {
  const box = $('thinking-chips');
  box.replaceChildren();
  for (const level of state.levels) {
    const b = el('button', 'tchip', level);
    b.type = 'button';
    b.setAttribute('role', 'radio');
    b.setAttribute('aria-checked', String(level === state.thinking));
    b.addEventListener('click', () => { state.thinking = level; store.set('thinking', level); renderThinking(); });
    box.appendChild(b);
  }
}

// The pill and toasts sit above the composer, whatever its height.
new ResizeObserver(() => root.style.setProperty('--composer-h', `${$('composer').offsetHeight}px`)).observe($('composer'));

// ---------------------------------------------------------------------------------------------------- start
async function boot() {
  fitViewport();
  renderThinking();
  for (let i = 0; ; i++) {
    try {
      const r = await fetch('/api/state', { cache: 'no-store' });
      const j = await r.json();
      state.models = j.models;
      state.levels = j.levels;
      if (!state.levels.includes(state.thinking)) state.thinking = 'medium';
      renderThinking();
      applyStatus(j.status);
      break;
    } catch (e) {
      conn.downSince = conn.downSince || Date.now() - 3000;
      renderBanner();
      await new Promise((res) => setTimeout(res, Math.min(10000, 1000 * 2 ** i)));
    }
  }
  conn.downSince = 0;
  await refreshSessions();
  const saved = store.get('current', '');
  const exists = (id) => state.sessions.some((s) => s.id === id);
  let id = null;
  if (saved && saved !== 'new' && exists(saved)) id = saved;
  else if (!saved) id = (state.sessions.find((s) => s.running) || state.sessions[0] || {}).id || null;
  openSession(id);
}
boot();
