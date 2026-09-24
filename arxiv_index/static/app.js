/* The search page (index.html): searching, the listings, the settings panel,
   and the import, export and update runs it starts on the server. */

const $ = s => document.querySelector(s);

/* ---- LaTeX -------------------------------------------------------------
   arXiv metadata is raw LaTeX in two distinct flavours, and they need
   different treatment:

     1. Real maths between $…$ or \(…\) — handed to KaTeX.
     2. Accents and special letters in ordinary prose, above all in author
        names and titles: M\"obius, Erd\H{o}s, \c{c}, \ss. These sit OUTSIDE
        maths mode, so KaTeX never sees them and they would otherwise show up
        as literal backslashes.

   So: render the maths first, then rewrite accents only in the text nodes
   KaTeX did not claim. Doing it in that order means a stray \v or \k inside
   an equation is left alone. */

const DELIMS = [
  {left: "$$", right: "$$", display: true},
  {left: "\\[", right: "\\]", display: true},
  {left: "$", right: "$", display: false},
  {left: "\\(", right: "\\)", display: false},
];

/* TeX accent command -> Unicode combining mark. Applying the mark after the
   base letter and normalising to NFC yields the precomposed character, which
   covers far more of the corpus than any hand-written lookup table would. */
const COMBINING = {
  '"': "̈", "'": "́", "`": "̀", "^": "̂", "~": "̃",
  "=": "̄", ".": "̇", "u": "̆", "v": "̌", "H": "̋",
  "c": "̧", "k": "̨", "r": "̊", "d": "̣", "b": "̱",
};
const LETTERS = {
  "ss": "ß", "ae": "æ", "AE": "Æ", "oe": "œ", "OE": "Œ", "aa": "å", "AA": "Å",
  "o": "ø", "O": "Ø", "l": "ł", "L": "Ł", "i": "ı", "j": "ȷ",
};

function deTeX(s) {
  if (!s || s.indexOf("\\") < 0 && s.indexOf("--") < 0) return s;

  // Special letters first, so that TeX's \'\i ("accent over a dotless i", the
  // standard way to write í) has a real letter to accent by the time the
  // accent pass runs. Matches \cmd{} or \cmd at a word boundary.
  // The trailing separator is consumed, not kept: in TeX a control word
  // swallows the whitespace that terminates it, so "\i msson" is one word.
  s = s.replace(/\\(ss|ae|AE|oe|OE|aa|AA|[oOlLij])(\{\}|[ \t]+|\b)/g,
                (m, cmd) => LETTERS[cmd] || m);

  // \"o  \"{o}  \c{c}  \H{o}  \'ı
  s = s.replace(
    /\\([\"'`^~=.]|[uvHckrdb])\s*\{([A-Za-zıȷ])\}|\\([\"'`^~=.])\s*([A-Za-zıȷ])/g,
    (m, c1, l1, c2, l2) => {
      const acc = COMBINING[c1 !== undefined ? c1 : c2];
      let base = l1 !== undefined ? l1 : l2;
      if (!acc) return m;
      // An accented dotless i/j is just the accented i/j; the dotless form
      // exists only so the accent does not collide with the tittle.
      if (base === "ı") base = "i";
      else if (base === "ȷ") base = "j";
      return (base + acc).normalize("NFC");
    });

  // Markup that carries no meaning once the text is HTML. Only these three
  // shapes are safe to touch: anything else beginning with a backslash out
  // here is an author's own maths written without $…$ delimiters, and
  // guessing where such a formula starts does more harm than leaving it.
  s = s.replace(/\\cite[tp]?\s*(\[[^\]]*\])?\s*\{[^}]*\}/g, "");
  s = s.replace(/\\(?:emph|textit|textbf|textrm|texttt|text|mbox)\s*\{([^{}]*)\}/g, "$1");
  s = s.replace(/\{\\(?:it|bf|rm|sl|sc|tt|em)\s+([^{}]*)\}/g, "$1");

  s = s.replace(/\\([&%_#])/g, "$1");   // escaped punctuation
  s = s.replace(/\\ /g, " ");           // forced space
  s = s.replace(/---/g, "—").replace(/--/g, "–");
  return s.replace(/[ \t]{2,}/g, " ");
}

/* Rewrite accents in every text node KaTeX has not already rendered. */
function deTeXTree(root) {
  const walk = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  const nodes = [];
  while (walk.nextNode()) {
    // Never touch a biblatex entry: its backslashes are meant to stay LaTeX.
    if (!walk.currentNode.parentElement.closest(".katex, .katex-display, pre"))
      nodes.push(walk.currentNode);
  }
  for (const n of nodes) {
    const out = deTeX(n.nodeValue);
    if (out !== n.nodeValue) n.nodeValue = out;
  }
}

function typeset(el) {
  try {
    renderMathInElement(el, {
      delimiters: DELIMS,
      throwOnError: false,      // arXiv LaTeX is frequently not self-contained
      ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
    });
  } catch (e) { /* fall through to the accent pass regardless */ }
  deTeXTree(el);
}
const esc = s => (s||"").replace(/[&<>"]/g, c =>
  ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const tidy = s => (s||"").replace(/\s+/g, " ").trim();

let stats = null, lastNote = null;
// Whether a fetch or an embedding run is going, and whether it embeds; set by
// updateState() below, read by note() to word the pending count.
let running = false, embedding = false;

function refreshStats() {
  return fetch("/api/stats").then(r => r.json()).then(s => {
    stats = s;
    $("#scope").textContent = "· " + s.categories.join(" · ");
    categoryBoxes(s.categories);
    note();
  });
}
refreshStats();

/* One checkbox per category, built from the server's list. Rebuilt only if
   that list changes, so a refresh of the counts keeps what is ticked. */
function categoryBoxes(categories) {
  const box = $("#catboxes"), key = categories.join(" ");
  if (box.dataset.for === key) return;
  box.dataset.for = key;
  box.replaceChildren(...categories.map(c => {
    const label = document.createElement("label"),
          input = document.createElement("input");
    input.type = "checkbox";
    input.className = "cat";
    input.value = c;
    label.append(input, " " + c);
    return label;
  }));
}

/* `extra` is remembered rather than passed through, so re-reading the counts
   after a fetch finishes does not wipe the line describing what is on screen. */
function note(extra) {
  if (extra !== undefined) lastNote = extra;
  let bits = [];
  if (stats) {
    bits.push(stats.embedded.toLocaleString() + " papers searchable");
    if (stats.missing.length)
      bits.push(stats.missing.join(", ") + " not in the index yet — "
                + (stats.local ? "import the arXiv snapshot under ⚙"
                               : "run build to add"));
    if (stats.pending > 0)
      bits.push(stats.pending.toLocaleString() + (embedding
        ? " still embedding — results improve as it goes"
        : " not embedded yet, so not searchable"));
  }
  if (lastNote) bits.unshift(lastNote);
  $("#status-text").textContent = bits.join("  ·  ");
  // Offered only when nothing is running: a fetch embeds what is pending too.
  $("#embed").hidden = !(stats && stats.pending > 0) || running;
  renderData();
}

function scoreBadges(p) {
  if (p.score == null) return "";
  return '<span class="score" title="Cosine similarity of the embeddings, '
       + '-1 to 1">' + p.score.toFixed(3) + "</span>";
}

function card(p) {
  const a = document.createElement("article");
  const cats = esc(p.categories);
  a.innerHTML = `
    <div class="top">
      ${scoreBadges(p)}
      <div style="flex:1">
        <p class="title"><a href="https://arxiv.org/abs/${esc(p.id)}"
           target="_blank" rel="noopener">${esc(tidy(p.title))}</a></p>
        <p class="authors">${esc(tidy(p.authors) || "")}</p>
        <p class="meta"><span class="cat">${cats}</span> ·
           ${esc(p.update_date||"")} · ${esc(p.id)}</p>
      </div>
    </div>
    <p class="abs">${esc(tidy(p.abstract))}</p>
    <div class="acts">
      <button class="link toggle">Abstract</button>
      <button class="link cite">BibLaTeX</button>
      <button class="link sim">Similar papers</button>
      <a href="https://arxiv.org/abs/${esc(p.id)}" target="_blank"
         rel="noopener">arXiv ↗</a>
      <a href="https://arxiv.org/pdf/${esc(p.id)}" target="_blank"
         rel="noopener">PDF ↗</a>
    </div>
    <div class="bib"><pre></pre>
      <div class="bibbar"><button class="link copy">Copy</button>
        <span class="copied"></span></div>
    </div>`;
  a.querySelector(".toggle").onclick = () => a.classList.toggle("open");
  a.querySelector(".sim").onclick = () => similar(p.id, tidy(p.title));
  a.querySelector(".cite").onclick = () => showCite(a, p.id);
  a.querySelector(".copy").onclick = () => copyCite(a);
  return a;
}

function render(data, label) {
  const box = $("#results");
  box.textContent = "";
  if (!data.results || !data.results.length) {
    box.innerHTML = '<p class="empty">' + esc(data.hint || "No matches.") + "</p>";
    note(data.hint ? "No ranked matches" : "No matches");
    return;
  }
  data.results.forEach(p => box.appendChild(card(p)));
  // One pass over the whole list. Abstracts are still display:none at this
  // point, which is fine -- KaTeX builds DOM and needs no layout.
  typeset(box);
  const bits = [data.total && data.total > data.results.length
    ? `${data.results.length} of ${data.total} results in ${data.ms} ms`
    : `${data.results.length} results in ${data.ms} ms`];
  if (data.warning) bits.push(data.warning);
  if (label) bits.push(label);
  note(bits.join("  ·  "));
  window.scrollTo({top: 0, behavior: "smooth"});
}

async function run(url, label) {
  $("#go").disabled = true;
  note("Searching…");
  try {
    const r = await fetch(url);
    const data = await r.json();
    if (data.error) { note("Error: " + data.error); return; }
    render(data, label);
  } catch (e) {
    note("Request failed: " + e.message);
  } finally {
    $("#go").disabled = false;
  }
}

// Off by default; remembered once set, since it is a display preference rather
// than part of the query.
const scoresOn = localStorage.getItem("arxiv-index-scores") === "1";
$("#showscores").checked = scoresOn;
document.body.classList.toggle("with-scores", scoresOn);
$("#showscores").onchange = e => {
  document.body.classList.toggle("with-scores", e.target.checked);
  localStorage.setItem("arxiv-index-scores", e.target.checked ? "1" : "0");
};

/* ---- The profile, and the two listings built on it ---------------------
   Both fields live in the index, not in this page: they describe the reader,
   not the tab, and the interests embedding has to be computed server-side
   anyway. So the editor is a view of server state -- opening it re-reads,
   Cancel discards by re-reading, and nothing is kept in localStorage. */

const prof = $("#settings"), pnote = $("#p-note");
let profile = {authors: [], interests: [], blend: 0.35, embedded: 0};

function pnotice(text, bad) {
  pnote.textContent = text || "";
  pnote.classList.toggle("bad", !!bad);
}

/* One editable row per interest. Built rather than written as markup because
   the text is the reader's and must never be interpolated into HTML. */
function addInterest(entry) {
  const row = document.createElement("div");
  row.className = "interest";

  const weight = document.createElement("input");
  weight.type = "number";
  weight.min = "0"; weight.max = "2"; weight.step = "0.1";
  weight.value = entry.weight;
  weight.title = "How much this interest counts. 0 switches it off.";

  // The field, and the two arrows stacked at its right edge. The buttons
  // carry the step, so it is the same 0.1 whether clicked or keyed.
  const box = document.createElement("div");
  box.className = "weight";
  const step = (delta) => {
    const at = Math.round((Number(weight.value || 0) + delta) * 10) / 10;
    weight.value = Math.min(2, Math.max(0, at)).toFixed(1);
    bound();
  };
  const arrow = (glyph, delta, label) => {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = glyph;
    b.title = label;
    b.tabIndex = -1;  // The field itself is the tab stop; arrows step it.
    b.onclick = () => step(delta);
    return b;
  };
  const less = arrow("\u2212", -0.1, "Count this interest less");
  const more = arrow("+", 0.1, "Count this interest more");
  // Nothing past the ends, and the button says so rather than going dead.
  const bound = () => {
    const at = Number(weight.value || 0);
    less.disabled = at <= 0;
    more.disabled = at >= 2;
  };
  weight.oninput = bound;
  bound();
  box.append(weight, more, less);

  const text = document.createElement("textarea");
  text.value = entry.text;
  text.spellcheck = false;
  text.placeholder = "Combinatorial K-theory of matroids";
  // An entry saved without a vector cannot be ranked by, which is worth
  // seeing on the row itself and not only in the notice.
  if (entry.text && entry.embedded === false) {
    text.title = "Saved, but not embedded yet — this one cannot be ranked by.";
    text.style.borderColor = "var(--warn)";
  }

  const drop = document.createElement("button");
  drop.type = "button";
  drop.className = "drop";
  drop.textContent = "×";
  drop.title = "Remove this interest";
  drop.onclick = () => {
    row.remove();
    // Never leave the list with nothing to type into.
    if (!$("#p-interests").children.length) addInterest({text: "", weight: 1});
  };

  row.append(box, text, drop);
  $("#p-interests").append(row);
  return row;
}

function showBlend() {
  const v = Number($("#p-blend").value);
  $("#p-blendout").textContent =
    v <= 0 ? "best match only" : v >= 1 ? "all equally" : v.toFixed(2);
}

function fillProfile() {
  $("#p-authors").value = profile.authors.join("\n");
  $("#p-interests").textContent = "";
  const rows = profile.interests.length
    ? profile.interests : [{text: "", weight: 1}];
  rows.forEach(addInterest);
  $("#p-blend").value = profile.blend;
  showBlend();
  const waiting = profile.interests.filter(i => !i.embedded).length;
  pnotice(waiting
    ? `${waiting} interest(s) saved but not embedded — those cannot be `
      + `ranked by.` : "");
}

async function loadProfile() {
  try {
    profile = await (await fetch("/api/profile")).json();
  } catch (e) { /* leave the defaults; saving will report the real error */ }
  fillProfile();
}

/* One control opens and shuts the panel. Opening re-reads both halves from
   the server, so what is on screen is what is stored -- the same rule the
   profile editor already followed, now covering the schedule too. */
function showSettings(open) {
  prof.hidden = !open;
  document.body.classList.toggle("settings-open", open);
  $("#cog").setAttribute("aria-expanded", open ? "true" : "false");
  // The header stops being sticky as it opens, so anywhere down the results
  // it would otherwise open off-screen.
  if (open) { window.scrollTo({top: 0}); loadProfile(); loadSchedule(); }
}

$("#cog").onclick = () => {
  const opening = prof.hidden;
  showSettings(opening);
  if (opening) $("#p-authors").focus();
};
$("#p-cancel").onclick = () => { showSettings(false); fillProfile(); };
$("#p-add").onclick = () => addInterest({text: "", weight: 1})
                              .querySelector("textarea").focus();
$("#p-blend").oninput = showBlend;

$("#p-save").onclick = async () => {
  const btn = $("#p-save");
  btn.disabled = true;
  pnotice("Saving…");
  try {
    const r = await fetch("/api/profile", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        // One author per line. The server trims, drops blanks and
        // de-duplicates, so this does not have to.
        authors: $("#p-authors").value.split("\n"),
        // Likewise for blank rows: posted as typed, cleaned server-side, and
        // re-rendered below from whatever came back.
        interests: [...$("#p-interests").children].map(row => ({
          text: row.querySelector("textarea").value,
          weight: row.querySelector("input").value,
        })),
        blend: $("#p-blend").value,
      }),
    });
    const data = await r.json();
    if (data.error) { pnotice("Error: " + data.error, true); return; }
    profile = data;
    // Re-render from what came back, so the list shown is the list stored.
    fillProfile();
    if (data.warning) pnotice(data.warning, true);
    else pnotice(`Saved · ${data.authors.length} author(s) · `
                 + `${data.embedded}/${data.interests.length} interest(s) `
                 + `embedded`);
  } catch (e) {
    pnotice("Request failed: " + e.message, true);
  } finally {
    btn.disabled = false;
  }
};

loadProfile();

/* --- Automatic updates -----------------------------------------------------

   Three controls for one setting, so they are saved on change rather than
   behind the profile's Save button: a switch that needs a separate confirming
   click is a switch people believe they have already set. The server is the
   one that decides what a setting means, so every save re-renders from the
   response rather than from what was typed. */

let schedule = {mode: "off", hours: 6, at: "07:00", next_run: null};

function whenText(t) {
  if (t === null || t === undefined) return "";
  const d = new Date(t * 1000), now = new Date();
  const hhmm = d.toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"});
  if (t * 1000 <= Date.now()) return "· due now";
  const sameDay = d.toDateString() === now.toDateString();
  const tomorrow = new Date(now.getTime() + 86400000).toDateString()
                     === d.toDateString();
  return "· next " + (sameDay ? hhmm
    : tomorrow ? "tomorrow " + hhmm
    : d.toLocaleDateString([], {month: "short", day: "numeric"}) + " " + hhmm);
}

function fillSchedule() {
  $("#s-mode").value = schedule.mode;
  $("#s-hours").value = schedule.hours;
  $("#s-at").value = schedule.at;
  $("#s-every").hidden = schedule.mode !== "interval";
  $("#s-at").hidden = schedule.mode !== "daily";
  $("#s-next").textContent =
    schedule.mode === "off" ? "" : whenText(schedule.next_run);
}

async function loadSchedule() {
  try {
    schedule = await (await fetch("/api/schedule")).json();
  } catch (e) { /* leave the defaults; saving will report the real error */ }
  fillSchedule();
}

async function saveSchedule() {
  // Render the new mode at once, so the hours/time control appears under the
  // pointer rather than after a round trip.
  schedule = {mode: $("#s-mode").value, hours: $("#s-hours").value,
              at: $("#s-at").value, next_run: schedule.next_run};
  fillSchedule();
  try {
    const r = await fetch("/api/schedule", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({mode: $("#s-mode").value,
                            hours: $("#s-hours").value,
                            at: $("#s-at").value}),
    });
    schedule = await r.json();
    fillSchedule();
  } catch (e) {
    $("#s-next").textContent = "· could not save: " + e.message;
  }
}

$("#s-mode").onchange = saveSchedule;
$("#s-hours").onchange = saveSchedule;
$("#s-at").onchange = saveSchedule;

loadSchedule();

/* The window both buttons act on. Both bounds are optional and neither is
   ever filled in on the reader's behalf: these buttons take the dates exactly
   as the form has them, the same way Search does, so clicking one cannot move
   a boundary that was deliberately set or left blank. Two empty fields mean
   the whole index, which is what an empty date field plainly says. */
function windowParams() {
  const p = new URLSearchParams();
  if ($("#since").value) p.set("since", $("#since").value);
  if ($("#until").value) p.set("until", $("#until").value);
  document.querySelectorAll(".cat:checked").forEach(c => p.append("cat", c.value));
  return p;
}

function rangeLabel() {
  const a = $("#since").value, b = $("#until").value;
  if (a && b) return `${a} to ${b}`;
  if (a) return `since ${a}`;
  if (b) return `up to ${b}`;
  return "all dates";
}

$("#followed").onclick = () => {
  const p = windowParams();
  run("/api/followed?" + p, "followed authors, " + rangeLabel());
};

$("#byinterest").onclick = () => {
  const p = windowParams();
  p.set("k", $("#k").value);
  run("/api/interests?" + p, "by your interests, " + rangeLabel());
};

$("#f").onsubmit = e => {
  e.preventDefault();
  const q = $("#q").value.trim(), author = $("#author").value.trim();
  const cats = [...document.querySelectorAll(".cat:checked")].map(c => c.value);
  const since = $("#since").value, until = $("#until").value;
  // Any single criterion is a valid search; only nothing at all is a no-op.
  if (!q && !author && !since && !until && !cats.length) return;
  const p = new URLSearchParams({q, k: $("#k").value});
  if (author) p.set("author", author);
  document.querySelectorAll(".cat:checked").forEach(c => p.append("cat", c.value));
  if ($("#since").value) p.set("since", $("#since").value);
  if ($("#until").value) p.set("until", $("#until").value);
  run("/api/search?" + p, author && !q ? "by " + author + ", newest first" : null);
};

// Typing in the author box should search too, not just the main field.
$("#author").addEventListener("keydown", e => {
  if (e.key === "Enter") { e.preventDefault(); $("#f").requestSubmit(); }
});

/* The entry is fetched once per card and then cached in the DOM, so toggling
   it shut and open again costs nothing. Note the <pre> is filled with
   textContent, never innerHTML: a biblatex entry is full of braces and
   backslashes and must not be typeset or parsed as markup. */
async function showCite(card, id) {
  const box = card.querySelector(".bib"), pre = box.querySelector("pre");
  if (pre.textContent) { card.classList.toggle("cited"); return; }
  pre.textContent = "Generating…";
  card.classList.add("cited");
  try {
    const r = await fetch("/api/bibtex?id=" + encodeURIComponent(id));
    const data = await r.json();
    pre.textContent = data.error ? "Error: " + data.error : data.entry;
  } catch (e) {
    pre.textContent = "Request failed: " + e.message;
  }
}

async function copyCite(card) {
  const text = card.querySelector(".bib pre").textContent;
  const flash = card.querySelector(".copied");
  try {
    await navigator.clipboard.writeText(text);
    flash.textContent = "copied";
  } catch (e) {
    // Clipboard access can be refused; select the text so Ctrl-C still works.
    const range = document.createRange();
    range.selectNodeContents(card.querySelector(".bib pre"));
    const sel = window.getSelection();
    sel.removeAllRanges(); sel.addRange(range);
    flash.textContent = "selected — press Ctrl-C";
  }
  setTimeout(() => { flash.textContent = ""; }, 2500);
}

/* ---- Fetching new papers ----------------------------------------------
   The button starts a top-up on the server and then polls it. It cannot wait
   on the response: a week's catch-up is about a minute, mostly embedding, and
   coming back from a long absence is many minutes of paging.

   The server owns the answer to "is one running?", which is also how a page
   reloaded mid-run picks the run back up instead of offering to start a
   second one. */

const fetchBtn = $("#fetch"), embedBtn = $("#embed"), updBox = $("#update");
let updTimer = null, statsTimer = null;
// Whether this page has seen the current run go by. A finished run stays on
// the server until the next one, and a reload an hour later should not
// announce it as though it had just happened.
let watched = false;

function showUpdate(text, bad) {
  updBox.textContent = text;
  updBox.classList.toggle("bad", !!bad);
  updBox.hidden = false;
}

function updateState(s) {
  // While an upload is on its way, the server may not have started the run
  // yet; an idle answer then is stale, not the end of it.
  if (uploading && s.state !== "running") return;
  running = s.state === "running";
  embedding = running && (s.kind === "embed" || !!s.progress);
  fetchBtn.disabled = running;
  fetchBtn.textContent = running && s.kind === "update"
    ? "Fetching…" : "Fetch new papers";
  note();
  if (running) watched = true;

  if (running) {
    // Embedding is the long half and reports a count; the walk before it only
    // has its own narration to offer, so show whichever exists.
    showUpdate(s.read
      ? `Reading ${bytes(s.read.done)} of ${bytes(s.read.total)}`
        + (s.kind === "snapshot"
           ? ` · ${s.read.matched.toLocaleString()} papers in scope` : "") + "…"
      : s.progress
      ? `Embedding ${s.progress.done.toLocaleString()} of `
        + `${s.progress.total.toLocaleString()}…`
      : (s.lines.length ? s.lines[s.lines.length - 1].trim() : "Starting…"));
  } else if (!watched) {
    updBox.hidden = true;
  } else if (s.state === "failed") {
    showUpdate(({embed: "Embedding", snapshot: "Import", index: "Import"}
                [s.kind] || "Update")
               + " failed: " + (s.error || "unknown error"), true);
  } else if (s.state === "done") {
    showUpdate((s.kind === "snapshot"
      ? `Imported ${s.imported.toLocaleString()} paper(s)` + (s.embedded
        ? `, embedded ${s.embedded.toLocaleString()}` : ", not embedded yet")
      : s.kind === "index"
      ? [s.lines.find(l => /^(Merged|Imported)/.test(l)) || "Index imported",
         s.lines.find(l => /settings/.test(l))].filter(Boolean).join(" ")
      : s.kind === "embed"
      ? `Embedded ${s.embedded.toLocaleString()} paper(s)`
      : s.embedded
      ? `Fetched and embedded ${s.embedded.toLocaleString()} paper(s)`
      : "Already up to date") + ` · ${s.elapsed}s`);
  } else {
    updBox.hidden = true;
  }

  if (running && !updTimer) {
    updTimer = setInterval(pollUpdate, 1500);
    // A long embedding run makes papers searchable as it goes; keep the
    // header's count moving with it rather than frozen at the start.
    statsTimer = setInterval(refreshStats, 30000);
  } else if (!running && updTimer) {
    clearInterval(updTimer);
    clearInterval(statsTimer);
    updTimer = statsTimer = null;
    // The header counts were read once at load; a finished run has moved them.
    refreshStats();
    // And a run that just finished is the one the next one is timed from.
    loadSchedule();
  }
}

async function pollUpdate() {
  try {
    updateState(await (await fetch("/api/update")).json());
  } catch (e) { /* transient — the next tick asks again */ }
}

fetchBtn.onclick = async () => {
  fetchBtn.disabled = true;
  watched = true;
  showUpdate("Starting…");
  try {
    // A 409 means someone else got there first; its body is the live state,
    // so handing it to updateState() shows that run rather than an error.
    updateState(await (await fetch("/api/update", {method: "POST"})).json());
  } catch (e) {
    showUpdate("Could not start the update: " + e.message, true);
    fetchBtn.disabled = false;
  }
};

embedBtn.onclick = async () => {
  embedBtn.hidden = true;
  watched = true;
  showUpdate("Starting…");
  try {
    updateState(await (await fetch("/api/embed", {method: "POST"})).json());
  } catch (e) {
    showUpdate("Could not start embedding: " + e.message, true);
    embedBtn.hidden = false;
  }
};

/* ---- Import and export ------------------------------------------------
   Offered only to a page opened on the server's own machine. An import posts
   the chosen file as the request body -- the browser streams it from disk --
   and the server reads it as it arrives, so the request lasts as long as the
   upload does. Progress comes from polling /api/update meanwhile, as for a
   fetch; the embedding or merge that follows the upload carries on without
   the tab. */

const snapIn = $("#d-snap"), snapGo = $("#d-snap-go"),
      idxIn = $("#d-index"), idxGo = $("#d-index-go"), expGo = $("#d-export");
let uploading = false;

const bytes = n => n >= 1e9 ? (n / 1e9).toFixed(2) + " GB"
                            : Math.round(n / 1e6).toLocaleString() + " MB";

function renderData() {
  const box = $("#data");
  box.hidden = !(stats && stats.local);
  if (box.hidden) return;
  const busy = running || uploading, held = !stats.missing.length;
  snapIn.disabled = $("#d-embed").disabled = held || busy;
  snapGo.disabled = held || busy || !snapIn.files.length;
  $("#d-snap-note").innerHTML = held
    ? "The index holds all your categories. To add one, list it in your "
      + "settings file and restart the server."
    : `Imports ${esc(stats.missing.join(", "))} from Kaggle's `
      + '<a href="https://www.kaggle.com/datasets/Cornell-University/arxiv" '
      + 'target="_blank" rel="noopener">arXiv snapshot</a>, unzipped: '
      + "arxiv-metadata-oai-snapshot.json.";
  idxIn.disabled = $("#d-mode").disabled = $("#d-take").disabled = busy;
  idxGo.disabled = busy || !idxIn.files.length;
  expGo.disabled = busy || !stats.papers;
}
snapIn.onchange = idxIn.onchange = renderData;

async function upload(kind, file, params) {
  uploading = watched = true;
  renderData();
  showUpdate(`Sending ${file.name}…`);
  if (!updTimer) {
    updTimer = setInterval(pollUpdate, 1500);
    statsTimer = setInterval(refreshStats, 30000);
  }
  const p = new URLSearchParams({name: file.name, ...params});
  let answer = null;
  try {
    const r = await fetch(`/api/import/${kind}?${p}`,
                          {method: "POST", body: file});
    answer = await r.json();
  } catch (e) {
    answer = {error: e.message};
  }
  uploading = false;
  if (answer.state) {
    updateState(answer);
    return;
  }
  // Refused before it started, or cut off: the server's own state says
  // which, and a run that failed explains itself better than the socket.
  try {
    const s = await (await fetch("/api/update")).json();
    if (s.kind === kind && s.state !== "idle") { updateState(s); return; }
  } catch (e) { /* fall through */ }
  updateState({state: "idle", lines: []});
  showUpdate("Import not started: " + answer.error, true);
}

snapGo.onclick = () => upload("snapshot", snapIn.files[0],
                              {embed: $("#d-embed").checked ? "1" : "0"});

idxGo.onclick = () => {
  const mode = $("#d-mode").value;
  if (mode === "replace" && !confirm(
      "Replace this index with the export? Papers and categories only this "
      + "index holds will be gone."))
    return;
  upload("index", idxIn.files[0],
         {mode, settings: $("#d-take").checked ? "1" : "0"});
};

expGo.onclick = () => {
  // A plain download: the server names the file and states its size.
  const a = document.createElement("a");
  a.href = "/api/export" + ($("#d-with").checked ? "?settings=1" : "");
  a.download = "";
  a.click();
};

pollUpdate();

function similar(id, title) {
  run(`/api/similar?id=${encodeURIComponent(id)}&k=${$("#k").value}`,
      "similar to " + deTeX(title).slice(0, 60));
}
