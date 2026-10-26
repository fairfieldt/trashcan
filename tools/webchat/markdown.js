// A small Markdown renderer for model output: fenced code, headings, lists, quotes, tables, paragraphs, and inline
// code, bold, italic and links. Input is escaped first, so model text never becomes HTML.
(function () {
  const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

  function inline(s) {
    const codes = [];
    s = s.replace(/`([^`]+)`/g, (_, c) => { codes.push(c); return '\u0000' + (codes.length - 1) + '\u0000'; });
    s = esc(s);
    s = s.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
    s = s.replace(/(^|[\s(])(https?:\/\/[^\s<)]+)/g, '$1<a href="$2" target="_blank" rel="noopener">$2</a>');
    s = s.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    s = s.replace(/(^|[^*])\*([^*\s][^*]*)\*/g, '$1<em>$2</em>');
    return s.replace(/\u0000(\d+)\u0000/g, (_, i) => '<code>' + esc(codes[i]) + '</code>');
  }

  function table(rows) {
    const cells = (r) => r.trim().replace(/^\||\|$/g, '').split('|').map((c) => c.trim());
    const head = cells(rows[0]);
    let h = '<table><thead><tr>' + head.map((c) => '<th>' + inline(c) + '</th>').join('') + '</tr></thead><tbody>';
    for (const r of rows.slice(2)) h += '<tr>' + cells(r).map((c) => '<td>' + inline(c) + '</td>').join('') + '</tr>';
    return h + '</tbody></table>';
  }

  window.renderMarkdown = function (src) {
    const lines = String(src || '').replace(/\r/g, '').split('\n');
    let out = '', i = 0;
    while (i < lines.length) {
      const l = lines[i];
      if (/^```/.test(l)) {
        const body = [];
        for (i++; i < lines.length && !/^```/.test(lines[i]); i++) body.push(lines[i]);
        i++;
        out += '<pre><code>' + esc(body.join('\n')) + '</code></pre>';
      } else if (/^#{1,6}\s/.test(l)) {
        const n = l.match(/^#+/)[0].length;
        out += `<h${n}>` + inline(l.replace(/^#+\s*/, '')) + `</h${n}>`;
        i++;
      } else if (/^\s*\|.*\|\s*$/.test(l) && i + 1 < lines.length && /^\s*\|?\s*:?-{2,}/.test(lines[i + 1])) {
        const rows = [];
        while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) rows.push(lines[i++]);
        out += table(rows);
      } else if (/^\s*([-*+]|\d+[.)])\s+/.test(l)) {
        const ordered = /^\s*\d/.test(l);
        out += ordered ? '<ol>' : '<ul>';
        while (i < lines.length && /^\s*([-*+]|\d+[.)])\s+/.test(lines[i])) {
          let item = lines[i++].replace(/^\s*([-*+]|\d+[.)])\s+/, '');
          while (i < lines.length && /^\s{2,}\S/.test(lines[i]) && !/^\s*([-*+]|\d+[.)])\s+/.test(lines[i])) item += ' ' + lines[i++].trim();
          out += '<li>' + inline(item) + '</li>';
        }
        out += ordered ? '</ol>' : '</ul>';
      } else if (/^>\s?/.test(l)) {
        const q = [];
        while (i < lines.length && /^>\s?/.test(lines[i])) q.push(lines[i++].replace(/^>\s?/, ''));
        out += '<blockquote>' + window.renderMarkdown(q.join('\n')) + '</blockquote>';
      } else if (/^\s*(---|\*\*\*)\s*$/.test(l)) {
        out += '<hr>';
        i++;
      } else if (!l.trim()) {
        i++;
      } else {
        const p = [];
        while (i < lines.length && lines[i].trim() && !/^(```|#{1,6}\s|>|\s*([-*+]|\d+[.)])\s+|\s*\|.*\|\s*$)/.test(lines[i])) p.push(lines[i++]);
        if (!p.length) p.push(lines[i++]);
        out += '<p>' + inline(p.join(' ')) + '</p>';
      }
    }
    return out;
  };
})();
