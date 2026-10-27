// ansi.js — render terminal output as HTML instead of as escape codes.
//
// A bash tool result is whatever the command wrote, escape sequences and all:
// `ls --color`, `git diff --color`, npm, cargo, pytest all emit SGR runs, and
// without this the card shows the literal "[0m" junk between the words. What
// matters here is that the *unrecognized* sequences are dropped rather than
// passed through — a result is model- and command-controlled text going into
// innerHTML, and the terminal protocol has more in it than colour.
//
// Handled: SGR colour (30-37/90-97, 40-47/100-107, 38;5;n, 48;5;n, 38;2;r;g;b),
// bold, dim, italic, underline, strikethrough, reverse; carriage returns as
// terminal overwrite (progress bars); backspace; and the erase/OSC/other CSI
// sequences, which are consumed and discarded.

const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

// The 8 base colours as CSS variables, so a skin can restyle the terminal
// without this file knowing about skins. Bright variants get the same name
// with a -bright suffix.
const FG = ['--term-black', '--term-red', '--term-green', '--term-yellow',
            '--term-blue', '--term-magenta', '--term-cyan', '--term-white'];

// 256-colour cube + greys, as literal colours: a skin cannot name 240 values,
// and the cube is defined by the standard, not by us.
function xterm256(n) {
  if (n < 8) return `var(${FG[n]})`;
  if (n < 16) return `var(${FG[n - 8]}-bright)`;
  if (n < 232) {
    const i = n - 16;
    const steps = [0, 95, 135, 175, 215, 255];
    const r = steps[Math.floor(i / 36) % 6];
    const g = steps[Math.floor(i / 6) % 6];
    const b = steps[i % 6];
    return `rgb(${r},${g},${b})`;
  }
  const v = 8 + (n - 232) * 10;
  return `rgb(${v},${v},${v})`;
}

// One SGR parameter list ("1;31") applied to a style record.
function applySgr(style, params) {
  const codes = params.length ? params : [0];
  for (let i = 0; i < codes.length; i++) {
    const n = codes[i];
    if (n === 0) { style.fg = null; style.bg = null; style.bold = false; style.dim = false;
                   style.italic = false; style.underline = false; style.strike = false; style.reverse = false; }
    else if (n === 1) style.bold = true;
    else if (n === 2) style.dim = true;
    else if (n === 3) style.italic = true;
    else if (n === 4) style.underline = true;
    else if (n === 9) style.strike = true;
    else if (n === 7) style.reverse = true;
    else if (n === 22) { style.bold = false; style.dim = false; }
    else if (n === 23) style.italic = false;
    else if (n === 24) style.underline = false;
    else if (n === 29) style.strike = false;
    else if (n === 27) style.reverse = false;
    else if (n >= 30 && n <= 37) style.fg = `var(${FG[n - 30]})`;
    else if (n === 39) style.fg = null;
    else if (n >= 40 && n <= 47) style.bg = `var(${FG[n - 40]})`;
    else if (n === 49) style.bg = null;
    else if (n >= 90 && n <= 97) style.fg = `var(${FG[n - 90]}-bright)`;
    else if (n >= 100 && n <= 107) style.bg = `var(${FG[n - 100]}-bright)`;
    else if (n === 38 || n === 48) {
      const key = n === 38 ? 'fg' : 'bg';
      const mode = codes[i + 1];
      if (mode === 5) { style[key] = xterm256(codes[i + 2] || 0); i += 2; }
      else if (mode === 2) {
        const [r, g, b] = [codes[i + 2] || 0, codes[i + 3] || 0, codes[i + 4] || 0];
        style[key] = `rgb(${r},${g},${b})`;
        i += 4;
      } else i += 1; // malformed: skip the introducer, keep going
    }
  }
}

function styleAttrs(s) {
  const out = [];
  if (s.fg) out.push(`color:${s.fg}`);
  if (s.bg) out.push(`background:${s.bg}`);
  if (s.bold) out.push('font-weight:600');
  if (s.dim) out.push('opacity:.6');
  if (s.italic) out.push('font-style:italic');
  if (s.underline) out.push('text-decoration:underline');
  if (s.strike) out.push('text-decoration:line-through');
  if (s.reverse) out.push('filter:invert(1)');
  return out.join(';');
}

function freshStyle() {
  return { fg: null, bg: null, bold: false, dim: false, italic: false,
           underline: false, strike: false, reverse: false };
}

// Terminals overwrite the current line on \r. Keeping the last non-empty
// segment is what a user sees when a progress bar finishes.
function collapseCarriageReturns(text) {
  if (text.indexOf('\r') < 0) return text;
  return text.split('\n').map((line) => {
    if (line.indexOf('\r') < 0) return line;
    const parts = line.split('\r');
    for (let i = parts.length - 1; i >= 0; i--) if (parts[i].length) return parts[i];
    return '';
  }).join('\n');
}

// Backspace is how a terminal rubs out the character before it.
function applyBackspace(text) {
  if (text.indexOf('\b') < 0) return text;
  const out = [];
  for (const ch of text) {
    if (ch === '\b') out.pop();
    else out.push(ch);
  }
  return out.join('');
}

export function stripAnsi(text) {
  return collapseCarriageReturns(applyBackspace(String(text || '')))
    .replace(/\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)/g, '')  // OSC … BEL / ST
    .replace(/\x1b\[[0-9;?]*[ -\/]*[@-~]/g, '')          // CSI
    .replace(/\x1b[@-Z\\-_]/g, '')                       // two-byte escapes
    .replace(/[\x00-\x08\x0b-\x1f\x7f]/g, '');           // other controls
}

export function hasAnsi(text) {
  return /[\x1b\x07\x08\r]/.test(String(text || ''));
}

// Terminal text to HTML: styled spans, no escapes, nothing unescaped.
export function ansiToHtml(text) {
  const src = applyBackspace(collapseCarriageReturns(String(text || '')));
  let out = '';
  let plain = '';
  const style = freshStyle();
  const flush = () => {
    if (!plain) return;
    const attrs = styleAttrs(style);
    out += attrs ? `<span style="${attrs}">${esc(plain)}</span>` : esc(plain);
    plain = '';
  };
  let i = 0;
  while (i < src.length) {
    const ch = src[i];
    if (ch !== '\x1b') {
      // Other control characters carry no meaning here; dropping them keeps
      // them out of the DOM rather than relying on the browser to hide them.
      if (ch === '\n' || ch === '\t' || ch >= ' ') plain += ch;
      i++;
      continue;
    }
    const next = src[i + 1];
    if (next === '[') {
      const m = /^\x1b\[([0-9;?]*)[ -\/]*([@-~])/.exec(src.slice(i));
      if (!m) { i += 2; continue; }
      if (m[2] === 'm') { flush(); applySgr(style, m[1].split(';').filter((x) => x !== '').map(Number)); }
      // every other CSI (erase, cursor motion, …) is dropped, not printed
      i += m[0].length;
      continue;
    }
    if (next === ']') {
      const m = /^\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)/.exec(src.slice(i));
      i += m ? m[0].length : 2;
      continue;
    }
    i += /^[\x1b][@-Z\\-_]/.test(src.slice(i, i + 2)) ? 2 : 1;
  }
  flush();
  return out;
}
