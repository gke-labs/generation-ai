// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

function init(rec) {
  const isDynamic = Boolean(new URLSearchParams(window.location.search).get('src'));

  if (rec.notebook && (isDynamic || document.title === '__TITLE__')) {
    document.title = rec.notebook;
    const h1 = document.querySelector('header h1');
    if (h1 && (isDynamic || h1.textContent === '__TITLE__')) {
      h1.textContent = rec.notebook;
    }
  }

  const metaEl = document.getElementById('recorded-meta');
  if (metaEl && (isDynamic || metaEl.textContent.includes('__RECORDED__'))) {
    const mins = Math.floor((rec.duration || 0) / 60);
    const secs = String(Math.floor((rec.duration || 0) % 60)).padStart(2, '0');
    metaEl.textContent = `recorded ${rec.recorded || 'unknown'} · real duration ${mins}m${secs}s`;
    if (rec.aborted) {
      const abortedSpan = document.createElement('span');
      abortedSpan.className = 'meta';
      abortedSpan.style.color = '#b3261e';
      abortedSpan.textContent = ` · aborted: ${rec.aborted}`;
      metaEl.parentNode.insertBefore(abortedSpan, metaEl.nextSibling);
    }
  }

  const main = document.getElementById('main');
  main.innerHTML = '';
  const cells = (rec.cells || []).map((c, i) => {
    const el = document.createElement('div');
    el.className = 'cell ' + c.type;
    el.id = 'cell-' + i;
    if (c.type === 'code') {
      el.innerHTML = '<div class="prompt">[ ]:</div><div class="input"><pre class="highlight">' + c.html + '</pre></div><div class="outputs"></div>';
    } else {
      el.innerHTML = '<div class="prompt"></div><div class="body">' + c.html + '</div>';
    }
    main.appendChild(el);
    return el;
  });

  // Effective timeline: real gaps between events, each capped.
  let eff = [];
  function buildTimeline() {
    const cap = parseFloat(document.getElementById('gap').value);
    eff = [];
    let acc = 0, prev = 0;
    for (const e of (rec.events || [])) {
      acc += Math.min(Math.max(e.t - prev, 0), cap);
      prev = e.t;
      eff.push(acc);
    }
  }

  // Apply a stream chunk the way a notebook output area does: \r returns
  // to the start of the current line, \b deletes one character.
  function appendStream(pre, chunk) {
    let text = pre.textContent;
    for (const ch of chunk) {
      if (ch === '\r') {
        const nl = text.lastIndexOf('\n');
        text = text.slice(0, nl + 1);
      } else if (ch === '\b') {
        text = text.slice(0, -1);
      } else {
        text += ch;
      }
    }
    pre.textContent = text;
  }

  function fmt(s) {
    s = Math.max(0, Math.round(s));
    return String(Math.floor(s / 60)).padStart(2, '0') + ':' + String(s % 60).padStart(2, '0');
  }

  const counts = {};
  let pendingClear = {};
  function reset() {
    pendingClear = {};
    for (const el of cells) {
      el.classList.remove('running');
      const o = el.querySelector('.outputs');
      if (o) o.innerHTML = '';
      const p = el.querySelector('.prompt');
      if (p && el.classList.contains('code')) p.textContent = '[ ]:';
    }
  }

  function beforeOutput(cellIndex, out) {
    if (pendingClear[cellIndex]) {
      out.innerHTML = '';
      pendingClear[cellIndex] = false;
    }
  }

  function apply(e) {
    const el = cells[e.cell];
    if (!el) return;
    const out = el.querySelector('.outputs');
    const prompt = el.querySelector('.prompt');
    if (e.type === 'input') {
      el.classList.add('running');
      prompt.textContent = '[*]:';
      counts[e.cell] = e.count;
    } else if (e.type === 'status' && e.state === 'idle') {
      el.classList.remove('running');
      if (counts[e.cell] != null) prompt.textContent = '[' + counts[e.cell] + ']:';
    } else if (e.type === 'stream') {
      beforeOutput(e.cell, out);
      let pre = out.lastElementChild;
      if (!pre || !pre.classList.contains('stream') || !pre.classList.contains(e.name)) {
        pre = document.createElement('pre');
        pre.className = 'stream ' + e.name;
        out.appendChild(pre);
      }
      appendStream(pre, e.text);
    } else if (e.type === 'display' || e.type === 'update') {
      const existing = e.display_id ? main.querySelector('.display[data-display-id="' + CSS.escape(e.display_id) + '"]') : null;
      if (e.type === 'update') {
        if (existing) existing.innerHTML = e.html;
      } else {
        beforeOutput(e.cell, out);
        const d = document.createElement('div');
        d.className = 'display';
        if (e.display_id) d.dataset.displayId = e.display_id;
        d.innerHTML = e.html;
        out.appendChild(d);
      }
    } else if (e.type === 'error') {
      beforeOutput(e.cell, out);
      const pre = document.createElement('pre');
      pre.className = 'error';
      pre.textContent = e.text;
      out.appendChild(pre);
    } else if (e.type === 'clear') {
      if (e.wait) pendingClear[e.cell] = true;
      else {
        out.innerHTML = '';
        pendingClear[e.cell] = false;
      }
    }
    if (document.getElementById('follow').checked && e.type !== 'status') {
      el.scrollIntoView({ block: 'center', behavior: 'smooth' });
    }
  }

  let pos = 0, clock = 0, playing = false, last = null, raf = null;
  const seek = document.getElementById('seek'),
        timeEl = document.getElementById('time'),
        playBtn = document.getElementById('play');

  function total() {
    return eff.length ? eff[eff.length - 1] : 0;
  }

  function render() {
    while (pos < (rec.events || []).length && eff[pos] <= clock) {
      apply(rec.events[pos++]);
    }
    seek.value = total() ? Math.round(1000 * clock / total()) : 0;
    const real = pos ? rec.events[pos - 1].t : 0;
    timeEl.textContent = fmt(clock) + ' / ' + fmt(total()) + ' (real ' + fmt(real) + ')';
    if (pos >= (rec.events || []).length && playing) {
      playing = false;
      playBtn.textContent = 'Replay';
    }
  }

  function tick(now) {
    if (!playing) return;
    if (last != null) {
      clock += (now - last) / 1000 * parseFloat(document.getElementById('speed').value);
    }
    last = now;
    render();
    raf = requestAnimationFrame(tick);
  }

  function seekTo(t) {
    reset();
    pos = 0;
    clock = t;
    render();
  }

  playBtn.onclick = () => {
    if (pos >= (rec.events || []).length) {
      seekTo(0);
    }
    playing = !playing;
    playBtn.textContent = playing ? 'Pause' : 'Play';
    last = null;
    if (playing) raf = requestAnimationFrame(tick);
  };

  seek.oninput = () => {
    seekTo(total() * seek.value / 1000);
  };

  document.getElementById('gap').onchange = () => {
    const frac = total() ? clock / total() : 0;
    buildTimeline();
    seekTo(total() * frac);
  };

  buildTimeline();
  render();
}

function loadRecording() {
  const params = new URLSearchParams(window.location.search);
  const src = params.get('src');
  if (src) {
    fetch(src)
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status} ${r.statusText}`);
        return r.json();
      })
      .then(init)
      .catch(err => {
        console.error(err);
        const main = document.getElementById('main');
        if (main) {
          main.innerHTML = `<div class="cell"><div class="prompt"></div><div class="outputs"><pre class="error">Failed to load ${src}: ${err.message}</pre></div></div>`;
        }
      });
    return;
  }

  const el = document.getElementById('recording');
  if (el) {
    const text = el.textContent.trim();
    if (text && text !== '__RECORDING__') {
      try {
        init(JSON.parse(text));
        return;
      } catch (err) {
        console.error(err);
        const main = document.getElementById('main');
        if (main) {
          main.innerHTML = `<div class="cell"><div class="prompt"></div><div class="outputs"><pre class="error">Failed to parse embedded recording: ${err.message}</pre></div></div>`;
        }
        return;
      }
    }
  }

  const main = document.getElementById('main');
  if (main) {
    main.innerHTML = '<div class="cell"><div class="prompt"></div><div class="outputs"><pre class="display">No recording loaded. Specify ?src=&lt;recording.json&gt; in the URL.</pre></div></div>';
  }
}

document.addEventListener('keydown', (ev) => {
  if (ev.code === 'Space' && ev.target.tagName !== 'SELECT' && ev.target.tagName !== 'INPUT') {
    ev.preventDefault();
    const playBtn = document.getElementById('play');
    if (playBtn) playBtn.click();
  }
});

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', loadRecording);
} else {
  loadRecording();
}
