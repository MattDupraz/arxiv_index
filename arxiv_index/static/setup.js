/* The first-time setup page (setup.html), served instead of the search page
   while the index holds no papers. Its imports are the settings panel's,
   driven the same way (see the "Import and export" section of app.js). */

const $ = s => document.querySelector(s);
const bytes = n => n >= 1e9 ? (n / 1e9).toFixed(2) + " GB"
                            : Math.round(n / 1e6).toLocaleString() + " MB";
const count = n => n.toLocaleString();
// `busy` locks the forms from the moment an import is launched until it has
// failed; one that succeeded leaves them locked, since the index is set up.
let busy = false, uploading = false, timer = null;
let current = null;     // the model chosen so far, if any

function refresh() {
  for (const el of ["#cats", "#model", "#snap", "#tar", "#embed", "#take",
                    ".path[data-path=build]", ".path[data-path=import]"])
    $(el).disabled = busy;
  // Only building needs a model chosen here; an export brings its own.
  $("#snap-go").disabled = busy || !$("#model").value || !$("#snap").files.length;
  $("#tar-go").disabled = busy || !$("#tar").files.length;
}
$("#snap").onchange = $("#tar").onchange = refresh;

/* The two ways to fill the index. Choosing one opens its form and closes the
   other's, whose choices are then not used. */
function choose(path) {
  for (const b of document.querySelectorAll(".path"))
    b.setAttribute("aria-pressed", String(b.dataset.path === path));
  $("#build").hidden = path !== "build";
  $("#import").hidden = path !== "import";
}
for (const b of document.querySelectorAll(".path"))
  b.onclick = () => choose(b.dataset.path);

async function init() {
  const s = await (await fetch("/api/setup")).json();
  $("#cats").value = s.categories.join(" ");
  if (!s.local) {
    $("#remote").hidden = false;
    $("#choose").hidden = true;
    return;
  }
  current = s.model;
  for (const m of s.models) {
    const o = new Option(`${m.name}  (${m.dim.toLocaleString()} dimensions)`,
                         m.name, false, m.name === s.model);
    $("#model").append(o);
  }
  if (!s.models.length)
    $("#model-err").textContent = s.ollama_error || "No embedding model is "
      + "installed. Run: ollama pull qwen3-embedding:4b, then reload this page.";
  else if (!s.models.some(m => m.name === s.model))
    // Nothing chosen yet: offer the default if it is installed.
    $("#model").value = s.models.some(m => m.name === s.default)
      ? s.default : s.models[0].name;
  refresh();
  // A reload in the middle of an import picks the run up, not a second one.
  const u = await (await fetch("/api/update")).json();
  if (u.state === "running" && (u.kind === "snapshot" || u.kind === "index")) {
    choose(u.kind === "snapshot" ? "build" : "import");
    watch(u.kind);
  }
}

async function saveCategories() {
  $("#cats-err").textContent = "";
  const r = await fetch("/api/categories", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({categories: $("#cats").value})});
  const s = await r.json();
  if (!r.ok) {
    $("#cats-err").textContent = s.error;
    $("#cats").focus();
    return false;
  }
  $("#cats").value = s.categories.join(" ");
  return true;
}

function setBar(fraction) {
  $("#p-bar").style.width = (100 * Math.max(0, Math.min(1, fraction))) + "%";
}

function watch(kind) {
  busy = true;
  refresh();
  $("#s3").hidden = false;
  $("#p-title").textContent = kind === "snapshot"
    ? "Importing from the snapshot" : "Importing the export";
  $("#p-err").textContent = "";
  $("#p-done").hidden = true;
  if (!timer) timer = setInterval(poll, 1000);
}

function stop() {
  clearInterval(timer);
  timer = null;
}

/* Unlock the forms after a failure, so it can be put right and tried again. */
function unlock() {
  busy = false;
  refresh();
}

async function poll() {
  try { render(await (await fetch("/api/update")).json()); }
  catch (e) { /* transient; the next tick asks again */ }
}

function render(s) {
  // Until the upload's request has been taken up, an idle answer is stale.
  if (uploading && s.state !== "running") return;
  if (s.state === "running") {
    if (s.read) {
      setBar(s.read.done / s.read.total);
      $("#p-text").textContent = `Reading ${bytes(s.read.done)} of `
        + `${bytes(s.read.total)}`
        + (s.kind === "snapshot"
           ? ` · ${count(s.read.matched)} papers in your categories so far` : "")
        + ". Keep this tab open until the file has been read.";
    } else if (s.progress) {
      setBar(s.progress.done / s.progress.total);
      $("#p-title").textContent = "Embedding";
      $("#p-text").textContent = `${count(s.progress.done)} of `
        + `${count(s.progress.total)} papers embedded. This runs on the server:`
        + " you can close the tab, or open the index and search while it works.";
      $("#p-next").textContent = `${count(s.imported)} papers imported.`;
      $("#p-done").hidden = false;
    } else if (s.lines.length) {
      $("#p-text").textContent = s.lines[s.lines.length - 1].trim();
    }
    return;
  }
  stop();
  if (s.state === "failed") {
    $("#p-title").textContent = "Import failed";
    $("#p-err").textContent = s.error || "unknown error";
    unlock();
    return;
  }
  if (s.state !== "done") return;
  setBar(1);
  if (s.kind === "snapshot" && !s.imported) {
    $("#p-title").textContent = "Nothing imported";
    $("#p-err").textContent = "No papers in your categories were found in "
      + "that file. Check that it is the arXiv snapshot, and the category names.";
    unlock();
    return;
  }
  $("#p-title").textContent = "Done";
  if (s.kind === "snapshot") {
    $("#p-text").textContent = `Imported ${count(s.imported)} papers`
      + (s.embedded ? ` and embedded ${count(s.embedded)}.` : ".");
    $("#p-next").textContent = (s.embedded ? "" : "They are not embedded yet: "
      + "the index page offers to embed them. ")
      + "The snapshot is a few days or weeks old; Fetch new papers, under ⚙, "
      + "brings the index up to date.";
  } else {
    const uses = s.lines.find(l => l.startsWith("This index now uses")) || "";
    $("#p-text").textContent = "The export is installed. " + uses
      + (s.lines.some(l => l.startsWith("Took up"))
         ? " Its settings are in place too." : "");
    const pull = s.lines.find(l => l.includes("ollama pull"));
    if (pull) $("#p-err").textContent = pull;
    $("#p-next").textContent = "Fetch new papers, under ⚙, brings it up to "
      + "date with whatever was posted since it was exported.";
  }
  $("#p-done").hidden = false;
}

async function ensureModel() {
  const want = $("#model").value;
  $("#model-err").textContent = "";
  if (want === current) return true;
  const r = await fetch("/api/setup/model", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({model: want})});
  const s = await r.json();
  if (!r.ok) { $("#model-err").textContent = s.error; return false; }
  current = want;
  return true;
}

async function upload(kind, file, params) {
  // Locked at once, before the categories and model are saved, so a second
  // click cannot start a second import.
  busy = true;
  refresh();
  // An export brings its own model and categories; only building needs them.
  if (kind === "snapshot"
      && (!(await saveCategories()) || !(await ensureModel()))) {
    unlock();
    return;
  }
  uploading = true;
  watch(kind);
  setBar(0);
  $("#p-text").textContent = `Sending ${file.name}…`;
  let answer;
  try {
    const r = await fetch(
      `/api/import/${kind}?` + new URLSearchParams({name: file.name, ...params}),
      {method: "POST", body: file});
    answer = await r.json();
  } catch (e) {
    answer = {error: e.message};
  }
  uploading = false;
  if (answer.state) { render(answer); return; }
  // Refused, or cut off: the server's own state says which.
  try {
    const s = await (await fetch("/api/update")).json();
    if (s.kind === kind && s.state !== "idle") { render(s); return; }
  } catch (e) { /* fall through */ }
  stop();
  $("#p-title").textContent = "Import not started";
  $("#p-err").textContent = answer.error;
  unlock();
}

$("#snap-go").onclick = () => upload("snapshot", $("#snap").files[0],
                                     {embed: $("#embed").checked ? "1" : "0"});
// Nothing to merge into yet, so the export simply becomes the index.
$("#tar-go").onclick = () => upload("index", $("#tar").files[0],
  {mode: "replace", settings: $("#take").checked ? "1" : "0"});
$("#open").onclick = () => { location.href = "/"; };
init();
