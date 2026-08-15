"""Local web UI for searching the index.

    python -m arxiv_index serve

Runs on the standard library alone. The point of a resident server is that the
vector matrix is loaded once and stays mapped, so a search costs one embedding
call plus one matrix-vector product -- rather than the CLI's re-open of the
whole file on every invocation.

Filtering is applied *after* scoring here, unlike the CLI. The CLI pre-filters
in SQL to avoid touching rows it does not need, but that gathers the matching
rows into a fresh array, which for a resident server would mean copying up to
several hundred MB per query. Scoring everything and masking is both simpler and
faster once the matrix is already in memory.

Binds to localhost only: the server exposes the index and, indirectly, Ollama.
"""

import datetime as dt
import json
import mimetypes
import pathlib
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

from . import (cite, config, rerank as rerank_mod, search as search_mod,
               store, textnorm)

# Vendored KaTeX (js, css, woff2 subset). Kept local rather than pulled from a
# CDN so the UI still works offline and does not phone home.
STATIC = pathlib.Path(__file__).resolve().parent / "static"


class ResidentIndex:
    """The matrix plus the per-row metadata needed for filtering, held in RAM."""

    def __init__(self):
        # One connection shared by every handler thread, so every touch of it
        # goes through this lock. Reentrant because refresh_if_stale() holds it
        # across reload().
        self._db_lock = threading.RLock()
        self.db = store.connect(check_same_thread=False)
        store.check_model(self.db)
        self.ids = []
        self.loaded = 0
        self.meta_loaded = -1
        self.gpu = None          # matrix in VRAM, when available
        self.torch = None
        self.reload()
        self.reload_metadata()

    def _to_gpu(self) -> None:
        """Mirror the matrix into VRAM. Falls back silently to the CPU path.

        Re-uploaded on every reload, so the old tensor is dropped first --
        during a build reload happens often, and leaking 747 MB each time would
        exhaust VRAM quickly.
        """
        self.gpu = None
        if not config.GPU_SEARCH or not len(self.ids):
            return
        try:
            import torch
        except ImportError:
            return
        try:
            if not torch.cuda.is_available():
                return
            self.torch = torch
            torch.cuda.empty_cache()
            self.gpu = torch.from_numpy(
                np.ascontiguousarray(self.matrix)).to("cuda")
        except Exception:  # noqa: BLE001 - VRAM pressure, driver issues, ...
            self.gpu = None

    def score(self, vector):
        """Cosine against every embedded paper, on the GPU when it is there."""
        if self.gpu is None:
            return search_mod.score_all(self.matrix, vector)
        query = self.torch.from_numpy(np.ascontiguousarray(vector)).to(
            "cuda").half()
        return (self.gpu @ query).float().cpu().numpy()

    def _rows(self, sql: str, params=()):
        with self._db_lock:
            return self.db.execute(sql, params).fetchall()

    def reload(self) -> None:
        with self._db_lock:
            rows = self.db.execute(
                "SELECT id, row, categories, update_date, authors FROM papers "
                "WHERE row IS NOT NULL ORDER BY row"
            ).fetchall()
            self.matrix, _ = store.load_matrix(self.db)
        self.ids = [r["id"] for r in rows]
        self.dates = np.array([r["update_date"] or "" for r in rows], dtype="U10")
        # Folded once at load (~0.5s for the full corpus) rather than per query.
        self.authors = [textnorm.fold(r["authors"] or "") for r in rows]
        # One boolean column per category beats re-parsing category strings on
        # every query. These index the *matrix*, so they must be built from the
        # embedded rows in row order -- never from the metadata table, which is
        # a different length and a different order.
        self.cat_masks = {
            cat: np.array([cat in (r["categories"] or "").split() for r in rows])
            for cat in config.CATEGORIES
        }
        self.loaded = len(self.ids)
        self._to_gpu()

    def reload_metadata(self) -> None:
        """Metadata for *every* paper, embedded or not.

        A semantic query can only reach embedded papers -- without a vector
        there is nothing to score. But an author or date lookup is pure
        metadata, and restricting it to embedded rows would silently hide
        papers: mid-build that is half the corpus. Held newest-first so a
        listing can stop as soon as it has enough.
        """
        with self._db_lock:
            rows = self.db.execute(
                "SELECT id, categories, update_date, authors FROM papers "
                "ORDER BY update_date DESC"
            ).fetchall()
        self.meta_ids = [r["id"] for r in rows]
        self.meta_dates = [r["update_date"] or "" for r in rows]
        self.meta_cats = [set((r["categories"] or "").split()) for r in rows]
        self.meta_authors = [textnorm.fold(r["authors"] or "") for r in rows]
        self.meta_loaded = len(rows)

    def refresh_if_stale(self) -> None:
        """Pick up papers embedded since load. Cheap: the matrix is a memmap, so
        re-mapping it does not copy, and during a build this keeps results
        current without restarting the server."""
        with self._db_lock:
            live = self.db.execute(
                "SELECT COUNT(*) FROM papers WHERE row IS NOT NULL"
            ).fetchone()[0]
            total = self.db.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
            # Tracked separately on purpose. During a build the embedded count
            # changes every few seconds while the metadata does not, and
            # re-folding 145k author strings each search would cost ~0.5s for
            # nothing. The lock is reentrant, so the reloads can retake it.
            if live != self.loaded:
                self.reload()
            if total != self.meta_loaded:
                self.reload_metadata()

    def stats(self) -> dict:
        with self._db_lock:
            total = store.count_papers(self.db)
            pending = store.count_pending(self.db)
        return {
            "papers": total,
            "embedded": total - pending,
            "pending": pending,
            "model": config.MODEL,
            "categories": list(config.CATEGORIES),
        }

    def query(self, text, k=20, categories=None, since=None, exclude=None,
              author=None, rerank=False):
        self.refresh_if_stale()
        if not self.ids:
            return [], 0.0, None

        started = time.monotonic()

        if not text:
            # No query to be similar to, so this is a metadata listing and has
            # no business consulting the vectors at all.
            return (self.browse(k, categories, since, author),
                    time.monotonic() - started, None)

        keep = None
        if categories:
            keep = np.zeros(len(self.ids), dtype=bool)
            for cat in categories:
                mask = self.cat_masks.get(cat)
                if mask is not None:
                    keep |= mask
        if since:
            recent = self.dates >= since
            keep = recent if keep is None else (keep & recent)
        terms = textnorm.fold_terms(author)
        if terms:
            by = np.fromiter(
                (textnorm.matches_terms(a, terms) for a in self.authors),
                dtype=bool, count=len(self.authors))
            keep = by if keep is None else (keep & by)

        vector = search_mod.embed_query_normalised(text)
        scores = self.score(vector)

        if keep is not None:
            if not keep.any():
                return [], time.monotonic() - started, None
            # Push filtered-out rows below any real cosine rather than
            # compacting the array, which would cost a copy.
            scores = np.where(keep, scores, -2.0)

        # Reranking needs a shortlist bigger than the caller asked for; the
        # cross-encoder's job is to reorder it down to k.
        shortlist = max(k, config.RERANK_CANDIDATES) if rerank else k
        want = min(shortlist + (1 if exclude else 0), len(self.ids))
        top = np.argpartition(-scores, want - 1)[:want]
        top = top[np.argsort(-scores[top])]

        chosen = [self.ids[i] for i in top
                  if scores[i] > -2.0 and self.ids[i] != exclude][:shortlist]
        if not chosen:
            return [], time.monotonic() - started, None

        by_id = {self.ids[i]: float(scores[i]) for i in top}
        meta = self._meta(chosen)
        results = [meta[i] | {"score": by_id[i]} for i in chosen]
        if rerank and results:
            # A reranker failure must not take the search down with it: fall
            # back to the vector order and let the caller say so.
            try:
                results = rerank_mod.rerank(text, results)
            except rerank_mod.RerankUnavailable as exc:
                return results[:k], time.monotonic() - started, str(exc)
        return results[:k], time.monotonic() - started, None

    def browse(self, k, categories=None, since=None, author=None):
        """Newest-first listing by metadata alone, across the whole corpus.

        Covers papers that have not been embedded yet, which matters during a
        build and for anything the embedder has not caught up with. Rows are
        already date-sorted, so this stops as soon as it has k of them.
        """
        terms = textnorm.fold_terms(author)
        wanted = set(categories or ())
        chosen = []
        for i, paper_id in enumerate(self.meta_ids):
            if since and self.meta_dates[i] < since:
                continue
            if wanted and not (wanted & self.meta_cats[i]):
                continue
            if terms and not textnorm.matches_terms(self.meta_authors[i], terms):
                continue
            chosen.append(paper_id)
            if len(chosen) >= k:
                break
        if not chosen:
            return []
        meta = self._meta(chosen)
        # No relevance score exists here; null keeps the UI from showing a
        # number that would mean nothing.
        return [meta[i] | {"score": None} for i in chosen]

    def count_matching(self, categories=None, since=None, author=None) -> int:
        """How many papers match these filters, embedding or not.

        Used to explain an empty relevance search: during a build the filters
        may well select papers that simply have no vector yet.
        """
        terms = textnorm.fold_terms(author)
        wanted = set(categories or ())
        total = 0
        for i in range(len(self.meta_ids)):
            if since and self.meta_dates[i] < since:
                continue
            if wanted and not (wanted & self.meta_cats[i]):
                continue
            if terms and not textnorm.matches_terms(self.meta_authors[i], terms):
                continue
            total += 1
        return total

    def _meta(self, ids) -> dict:
        placeholders = ",".join("?" * len(ids))
        return {
            r["id"]: dict(r)
            for r in self._rows(
                f"SELECT * FROM papers WHERE id IN ({placeholders})", ids
            )
        }

    def vector_for(self, paper_id):
        found = self._rows("SELECT row FROM papers WHERE id = ?", (paper_id,))
        row = found[0] if found else None
        if row is None or row["row"] is None:
            return None
        total = store.vector_count()
        mm = np.memmap(config.VEC_PATH, dtype=config.VEC_DTYPE, mode="r",
                       shape=(total, config.DIM))
        return np.asarray(mm[row["row"]], dtype=np.float32)

    def similar(self, paper_id, k=20, rerank=False):
        """Papers closest to a given one.

        With `rerank`, the cross-encoder rescores the shortlist using the source
        paper's own title and abstract in place of a query. It is a text pair
        either way, so nothing about the model changes -- only that the left
        side is an abstract rather than a question.
        """
        self.refresh_if_stale()
        vector = self.vector_for(paper_id)
        if vector is None:
            return None, 0.0, None
        started = time.monotonic()
        scores = self.score(vector)
        # A wider net when reranking, for the same reason as in search: the
        # cross-encoder can only reorder what the index hands it.
        shortlist = max(k, config.RERANK_CANDIDATES) if rerank else k
        want = min(shortlist + 1, len(self.ids))
        top = np.argpartition(-scores, want - 1)[:want]
        top = top[np.argsort(-scores[top])]
        chosen = [self.ids[i] for i in top if self.ids[i] != paper_id][:shortlist]
        if not chosen:
            return [], time.monotonic() - started, None
        by_id = {self.ids[i]: float(scores[i]) for i in top}
        meta = self._meta(chosen)
        results = [meta[i] | {"score": by_id[i]} for i in chosen]

        if rerank:
            source = self._meta([paper_id]).get(paper_id)
            if source:
                try:
                    results = rerank_mod.rerank(
                        rerank_mod.document_text(source), results)
                except rerank_mod.RerankUnavailable as exc:
                    return results[:k], time.monotonic() - started, str(exc)
        return results[:k], time.monotonic() - started, None


def make_handler(index: ResidentIndex):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # quieter than the default
            pass

        def _send(self, body: bytes, content_type: str, status: int = 200,
                  no_store: bool = False):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            if no_store:
                self.send_header("Cache-Control", "no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload, status: int = 200):
            # Dynamic: results change as the build progresses.
            self._send(json.dumps(payload).encode("utf-8"),
                       "application/json; charset=utf-8", status, no_store=True)

        def _static(self, rel: str):
            """Serve a vendored asset, refusing anything outside STATIC."""
            target = (STATIC / unquote(rel)).resolve()
            if not target.is_file() or STATIC not in target.parents:
                self._send(b"not found", "text/plain", 404)
                return
            kind, _ = mimetypes.guess_type(target.name)
            if target.suffix == ".woff2":
                kind = "font/woff2"          # not in every mimetypes database
            elif target.suffix == ".js":
                kind = "application/javascript"
            body = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", kind or "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            # Vendored assets never change without a redeploy of the package.
            self.send_header("Cache-Control", "public, max-age=604800, immutable")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            one = lambda key, default=None: params.get(key, [default])[0]

            if parsed.path == "/":
                # Never cache the page. It is generated from web.py, so it
                # changes whenever the server is edited and restarted -- a
                # browser holding yesterday's copy would silently hide new UI.
                # (The vendored assets under /static are immutable and are
                # cached aggressively instead.)
                self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8",
                           no_store=True)
                return

            if parsed.path.startswith("/static/"):
                self._static(parsed.path[len("/static/"):])
                return

            if parsed.path == "/api/stats":
                self._json(index.stats())
                return

            if parsed.path == "/api/search":
                query = (one("q") or "").strip()
                author = (one("author") or "").strip()
                try:
                    # Generous, because listing a prolific author's whole
                    # output is a legitimate request (Sturmfels has 217).
                    k = max(1, min(500, int(one("k", "20"))))
                except ValueError:
                    k = 20
                cats = [c for c in params.get("cat", [])
                        if c in config.CATEGORIES]
                since = one("since") or None
                if since and not _valid_date(since):
                    self._json({"error": f"bad date: {since}"}, 400)
                    return
                # Any single criterion is a valid search on its own -- an
                # author, a category or a date each describe a listing. Only a
                # request with no criteria at all is rejected, matching the CLI.
                if not (query or author or since or cats):
                    self._json(
                        {"error": "give a query, author, category or date"}, 400)
                    return
                rerank = one("rerank", "0") in ("1", "true", "yes")
                try:
                    results, elapsed, warning = index.query(
                        query, k, cats, since, author=author or None,
                        rerank=rerank and bool(query))
                except Exception as exc:  # surfaced in the UI, not swallowed
                    self._json({"error": str(exc)}, 500)
                    return
                payload = {"results": results, "ms": round(elapsed * 1000),
                           "ranked": "relevance" if query else "date",
                           "reranked": bool(rerank and query and not warning)}
                if warning:
                    payload["warning"] = f"Reranker unavailable: {warning}"
                if query and not results:
                    # An empty relevance search is confusing while a build is
                    # running: the filters may match plenty of papers that
                    # simply have no vector to rank yet.
                    waiting = index.count_matching(cats, since, author or None)
                    if waiting:
                        payload["hint"] = (
                            f"{waiting:,} paper(s) match these filters but are "
                            "not embedded yet, so they cannot be ranked by "
                            "relevance. Clear the search box to list them."
                        )
                self._json(payload)
                return

            if parsed.path == "/api/bibtex":
                paper_id = one("id", "")
                rows = index._rows(
                    "SELECT * FROM papers WHERE id = ?", (paper_id,)
                )
                if not rows:
                    self._json({"error": f"{paper_id} is not in the index"}, 404)
                    return
                self._json({"entry": cite.biblatex(dict(rows[0]))})
                return

            if parsed.path == "/api/similar":
                paper_id = one("id", "")
                try:
                    # Generous, because listing a prolific author's whole
                    # output is a legitimate request (Sturmfels has 217).
                    k = max(1, min(500, int(one("k", "20"))))
                except ValueError:
                    k = 20
                rerank = one("rerank", "0") in ("1", "true", "yes")
                results, elapsed, warning = index.similar(paper_id, k, rerank)
                if results is None:
                    self._json({"error": f"{paper_id} has no vector yet"}, 404)
                    return
                payload = {"results": results, "ms": round(elapsed * 1000),
                           "reranked": bool(rerank and not warning)}
                if warning:
                    payload["warning"] = f"Reranker unavailable: {warning}"
                self._json(payload)
                return

            self._send(b"not found", "text/plain", 404)

    return Handler


def _valid_date(text: str) -> bool:
    try:
        dt.datetime.strptime(text, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def serve(port: int = 8000, host: str = "127.0.0.1", open_browser: bool = True):
    print("Loading index ...")
    index = ResidentIndex()
    stats = index.stats()
    size = index.matrix.nbytes / 1e6 if len(index.ids) else 0
    print(f"{stats['embedded']:,} papers resident ({size:,.0f} MB)"
          + (f", {stats['pending']:,} still embedding" if stats["pending"] else ""))

    server = ThreadingHTTPServer((host, port), make_handler(index))
    url = f"http://{host}:{port}/"
    print(f"\n  {url}\n\nCtrl-C to stop.")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>arXiv index</title>
<link rel="stylesheet" href="/static/katex.min.css">
<style>
:root {
  --bg: #fbfbfa; --panel: #fff; --ink: #1a1a1a; --muted: #6b6b6b;
  --line: #e3e3e0; --accent: #7c3f00; --accent-soft: #f0e6d8; --shadow: rgba(0,0,0,.06);
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #16161a; --panel: #1e1e23; --ink: #ececf0; --muted: #9a9aa4;
    --line: #2e2e36; --accent: #e0a35c; --accent-soft: #2a2118; --shadow: rgba(0,0,0,.3);
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font: 16px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
header {
  position: sticky; top: 0; z-index: 10; background: var(--panel);
  border-bottom: 1px solid var(--line); box-shadow: 0 1px 3px var(--shadow);
}
.wrap { max-width: 900px; margin: 0 auto; padding: 0 20px; }
h1 { font-size: 16px; font-weight: 600; margin: 0; padding: 14px 0 0; letter-spacing: .01em; }
h1 span { color: var(--muted); font-weight: 400; }
form { display: flex; gap: 8px; padding: 12px 0; flex-wrap: wrap; }
input[type=search] {
  flex: 1 1 320px; padding: 10px 13px; font-size: 16px; font-family: inherit;
  border: 1px solid var(--line); border-radius: 7px; background: var(--bg);
  color: var(--ink);
}
input[type=search]:focus { outline: 2px solid var(--accent); outline-offset: -1px; }
button {
  padding: 10px 18px; font-size: 16px; font-family: inherit; font-weight: 500;
  border: 0; border-radius: 7px; background: var(--accent); color: #fff;
  cursor: pointer;
}
button:disabled { opacity: .5; cursor: default; }
.opts {
  display: flex; gap: 16px; align-items: center; flex-wrap: wrap;
  padding-bottom: 12px; font-size: 14px; color: var(--muted);
}
.opts label { display: flex; gap: 5px; align-items: center; cursor: pointer; }
.opts input[type=date], .opts select, .opts input[type=text] {
  font: inherit; padding: 3px 6px; border: 1px solid var(--line);
  border-radius: 5px; background: var(--bg); color: var(--ink);
}
.opts input[type=text] { width: 150px; }
#status { padding: 14px 0 0; font-size: 14px; color: var(--muted); min-height: 20px; }
#results { padding: 6px 0 60px; }
article {
  background: var(--panel); border: 1px solid var(--line); border-radius: 9px;
  padding: 14px 16px; margin: 10px 0;
}
.top { display: flex; gap: 12px; align-items: baseline; }
.scores { display: flex; flex-direction: column; gap: 3px; flex: none;
          align-items: stretch; }
/* Scores are diagnostics, not reading material: hidden unless asked for.
   Toggled by a class rather than re-rendering, so it costs no re-search. */
body:not(.with-scores) .scores, body:not(.with-scores) .top > .score {
  display: none;
}
.score {
  font-variant-numeric: tabular-nums; font-size: 13px; font-weight: 600;
  color: var(--accent); background: var(--accent-soft); padding: 2px 7px;
  border-radius: 5px; flex: none; text-align: center;
}
/* The cosine is context for the reranked score, so it reads as secondary. */
.score.vec {
  color: var(--muted); background: transparent;
  border: 1px solid var(--line); font-weight: 500; font-size: 12px;
}
.score small { font-size: 9.5px; font-weight: 500; opacity: .75;
               display: block; letter-spacing: .04em; }
.title { font-size: 16.5px; font-weight: 600; margin: 0; line-height: 1.42; }
.title a { color: inherit; text-decoration: none; }
.title a:hover { text-decoration: underline; text-decoration-color: var(--accent); }
.meta { font-size: 13.5px; color: var(--muted); margin: 5px 0 0; }
.meta .cat {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12.5px;
}
.authors { font-size: 14px; color: var(--muted); margin: 3px 0 0; font-style: italic; }
/* Abstracts are the one block of sustained reading here, so they get more
   leading than the rest of the page. */
.abs { font-size: 14.5px; line-height: 1.72; margin: 10px 0 0; color: var(--ink);
       display: none; }
article.open .abs { display: block; }
.acts { margin: 8px 0 0; display: flex; gap: 14px; font-size: 13.5px; }
.acts a, .acts button.link {
  color: var(--accent); background: none; border: 0; padding: 0; font: inherit;
  cursor: pointer; text-decoration: none;
}
.acts a:hover, .acts button.link:hover { text-decoration: underline; }
.bib { display: none; margin: 10px 0 0; }
article.cited .bib { display: block; }
.bib pre {
  margin: 0; padding: 11px 13px; overflow-x: auto; border-radius: 7px;
  background: var(--bg); border: 1px solid var(--line);
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12.5px; line-height: 1.5; white-space: pre; color: var(--ink);
}
.bib .bibbar { display: flex; gap: 12px; align-items: center; margin: 6px 0 0;
               font-size: 13px; color: var(--muted); }
.empty { color: var(--muted); padding: 30px 0; text-align: center; }
mark { background: var(--accent-soft); color: inherit; }
/* KaTeX defaults to 1.21em, which makes inline math tower over the body text.
   Nudged up from parity so symbols and subscripts stay legible without the
   line spacing going ragged. */
.katex { font-size: 1.09em; }
/* A stray display equation in an abstract must not widen the page. */
.katex-display { overflow-x: auto; overflow-y: hidden; padding: 2px 0; }
/* Unparseable LaTeX is shown as-is rather than throwing; keep it legible. */
.katex-error { color: var(--muted) !important; font-family: ui-monospace, monospace; }
</style>
</head>
<body>
<header><div class="wrap">
  <h1>arXiv index <span id="scope"></span></h1>
  <form id="f">
    <input type="search" id="q" placeholder="Describe what you are looking for…"
           autofocus autocomplete="off">
    <button type="submit" id="go">Search</button>
  </form>
  <div class="opts">
    <label>Author <input type="text" id="author" placeholder="Kollár  ·  Larson, Payne"
                        title="Several names, comma-separated, match papers they wrote together"
                        autocomplete="off"></label>
    <span>Categories:</span>
    <label><input type="checkbox" class="cat" value="math.AC"> math.AC</label>
    <label><input type="checkbox" class="cat" value="math.AG"> math.AG</label>
    <label><input type="checkbox" class="cat" value="math.CO"> math.CO</label>
    <label title="Rescores the top 50 hits with a cross-encoder that reads query and abstract together. Slower, better ordered.">
      <input type="checkbox" id="rerank" checked> Rerank top 50</label>
    <label title="Show the relevance logit and cosine for each hit">
      <input type="checkbox" id="showscores"> Scores</label>
    <label>Since <input type="date" id="since"></label>
    <label>Results
      <select id="k">
        <option>10</option><option selected>20</option>
        <option>50</option><option>100</option><option>250</option>
      </select>
    </label>
  </div>
</div></header>

<div class="wrap">
  <div id="status"></div>
  <div id="results"></div>
</div>

<script src="/static/katex.min.js"></script>
<script src="/static/auto-render.min.js"></script>
<script>
const $ = s => document.querySelector(s);

/* ---- LaTeX -------------------------------------------------------------
   arXiv metadata is raw LaTeX in two distinct flavours, and they need
   different treatment:

     1. Real maths between $…$ or \(…\) — handed to KaTeX.
     2. Accents and special letters in ordinary prose, above all in author
        names and titles: Gr\"obner, Erd\H{o}s, \c{c}, \ss. These sit OUTSIDE
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

let stats = null;
fetch("/api/stats").then(r => r.json()).then(s => {
  stats = s;
  $("#scope").textContent = "· " + s.categories.join(" · ");
  note();
});

function note(extra) {
  let bits = [];
  if (stats) {
    bits.push(stats.embedded.toLocaleString() + " papers searchable");
    if (stats.pending > 0)
      bits.push(stats.pending.toLocaleString() + " still embedding — "
                + "results improve as the build finishes");
  }
  if (extra) bits.unshift(extra);
  $("#status").textContent = bits.join("  ·  ");
}

/* Reranked hits carry two scores on different scales, shown stacked so they can
   be read against each other: the cross-encoder's log-odds (what the order is
   based on) and the cosine the index started from (what it was before). A
   result high on one and low on the other is exactly where reranking earned
   its keep. Unreranked hits have only the cosine, and it needs no label. */
function scoreBadges(p) {
  if (p.score == null) return "";
  if (p.rerank_margin == null)
    return '<span class="score" title="Cosine similarity of the embeddings, '
         + '-1 to 1">' + p.score.toFixed(3) + "</span>";
  return '<div class="scores">'
       + '<span class="score" title="Cross-encoder log-odds that this answers '
       + 'the query. Higher is better; the ordering is based on this.">'
       + '<small>RERANK</small>' + p.rerank_margin.toFixed(2) + "</span>"
       + '<span class="score vec" title="Cosine similarity from the embedding '
       + 'index, before reranking.">'
       + '<small>COS</small>' + p.vector_score.toFixed(3) + "</span>"
       + "</div>";
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
  const bits = [`${data.results.length} results in ${data.ms} ms`];
  if (data.reranked) bits.push("cross-encoder reranked");
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

$("#f").onsubmit = e => {
  e.preventDefault();
  const q = $("#q").value.trim(), author = $("#author").value.trim();
  const cats = [...document.querySelectorAll(".cat:checked")].map(c => c.value);
  const since = $("#since").value;
  // Any single criterion is a valid search; only nothing at all is a no-op.
  if (!q && !author && !since && !cats.length) return;
  const p = new URLSearchParams({q, k: $("#k").value});
  if (author) p.set("author", author);
  if ($("#rerank").checked && q) p.set("rerank", "1");
  document.querySelectorAll(".cat:checked").forEach(c => p.append("cat", c.value));
  if ($("#since").value) p.set("since", $("#since").value);
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

function similar(id, title) {
  // Honour the same checkbox as search: the cross-encoder scores a text pair
  // either way, with this paper's abstract standing in for the query.
  const rr = $("#rerank").checked ? "&rerank=1" : "";
  run(`/api/similar?id=${encodeURIComponent(id)}&k=${$("#k").value}` + rr,
      "similar to " + deTeX(title).slice(0, 60));
}
</script>
</body>
</html>
"""
