"""Local web UI for searching the index.

    python -m arxiv_index serve

Runs on the standard library alone. The point of a resident server is that the
vector matrix is loaded once and stays put -- in VRAM when there is a GPU,
mapped from the file otherwise -- so a search costs one embedding call plus one
matrix-vector product, rather than the CLI's re-open of the whole file on every
invocation.

Filtering is applied *after* scoring here, unlike the CLI. The CLI pre-filters
in SQL to avoid touching rows it does not need, but that gathers the matching
rows into a fresh array, which for a resident server would mean copying up to
several hundred MB per query. Scoring everything and masking is both simpler and
faster once the matrix is already in memory.

Binds to localhost only: the server exposes the index and, indirectly, Ollama.
"""

import collections
import datetime as dt
import ipaddress
import json
import mimetypes
import os
import pathlib
import signal
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

from . import (cite, config, embedder, ingest, profile as profile_mod,
               schedule as schedule_mod, search as search_mod, settings,
               store, textnorm, transfer, update as update_mod)

# The pages, their styles and scripts, and a vendored KaTeX (js, css, woff2
# subset), kept local rather than pulled from a CDN so the UI still works
# offline and does not phone home. index.html is the search page; setup.html
# is what `/` serves instead while the index holds no papers.
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
        self._db_file = os.stat(config.DB_PATH).st_ino
        self._vec_file = self._vector_file()
        # Read once rather than per request: the category masks are built from
        # this list, so a category added to the settings file since would
        # filter on a mask that does not exist. set_categories changes it.
        self.categories = settings.categories()
        self._read_holdings()
        self.ids = []
        self.matrix = None
        self.gpu = None          # matrix in VRAM, when available
        self.torch = None
        # Values that repeat across rows, held once. See _shared().
        self._pool = {}
        self._version = self._data_version()
        self._vec_print, self._meta_print = self._fingerprints()
        self.reload()
        self.reload_metadata()

    def _shared(self, value):
        """The canonical instance of `value`, so equal values cost one object.

        The per-row metadata is mostly repetition: 145k rows carry 5.2k distinct
        dates and 4.6k distinct category sets between them, and every folded
        author string is built twice -- once for the embedded rows, once for the
        metadata table. Pooling them turns 47 MB of category sets into well
        under one, and is why the two author lists share their strings rather
        than holding a copy each.

        The pool is never pruned. It only ever holds values still present in the
        corpus, minus whatever a deleted paper leaves behind, which is bounded
        by the vocabulary rather than by the number of rows.
        """
        return self._pool.setdefault(value, value)

    def _to_gpu(self) -> None:
        """Mirror the matrix into VRAM. Falls back silently to the CPU path.

        Re-uploaded on every reload, so the old tensor is dropped first --
        during a build reload happens often, and leaking 747 MB each time would
        exhaust VRAM quickly.

        On success the host copy is released. Once the vectors are in VRAM
        nothing reads them from RAM again, and the upload has just touched every
        page of the file: keeping the mapping would hold 747 MB resident for a
        fallback that cannot be taken while `gpu` is set. Dropping it is free --
        `reload()` re-maps from scratch anyway.
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
        else:
            self.matrix = None

    def score(self, vector):
        """Cosine against every embedded paper, on the GPU when it is there."""
        # Snapshot both, matrix first: a reload running in another thread swaps
        # the pair, and the local reference keeps whichever one this query picks
        # alive for the duration of the scan.
        matrix, gpu = self.matrix, self.gpu
        if gpu is None:
            return search_mod.score_all(matrix, vector)
        query = self.torch.from_numpy(np.ascontiguousarray(vector)).to(
            "cuda").half()
        return (gpu @ query).float().cpu().numpy()

    def _rows(self, sql: str, params=()):
        with self._db_lock:
            return self.db.execute(sql, params).fetchall()

    def reload(self) -> None:
        ids, dates, authors = [], [], []
        # One boolean column per category beats re-parsing category strings on
        # every query. These index the *matrix*, so they must be built from the
        # embedded rows in row order -- never from the metadata table, which is
        # a different length and a different order.
        masks = {cat: [] for cat in self.categories}
        with self._db_lock:
            # Streamed rather than fetchall()'d. The full result is ~65 MB of
            # sqlite3.Row objects, and freeing them does not hand the memory
            # back: glibc keeps the arena and the process stays that size. Never
            # allocating it is the only way not to pay for it -- which matters
            # here because a build calls this every few seconds.
            for row in self.db.execute(
                "SELECT id, row, categories, update_date, authors FROM papers "
                "WHERE row IS NOT NULL ORDER BY row"
            ):
                ids.append(row["id"])
                dates.append(self._shared(row["update_date"] or ""))
                # Folded once at load (~0.5s for the full corpus) rather than
                # per query.
                authors.append(
                    self._shared(textnorm.fold(row["authors"] or "")))
                cats = (row["categories"] or "").split()
                for cat, mask in masks.items():
                    mask.append(cat in cats)
            self.matrix, _ = store.load_matrix(self.db)
        self.ids = ids
        self.dates = np.array(dates, dtype="U10")
        self.authors = authors
        self.cat_masks = {cat: np.array(mask, dtype=bool)
                          for cat, mask in masks.items()}
        self._to_gpu()

    def reload_metadata(self) -> None:
        """Metadata for *every* paper, embedded or not.

        A semantic query can only reach embedded papers -- without a vector
        there is nothing to score. But an author or date lookup is pure
        metadata, and restricting it to embedded rows would silently hide
        papers: mid-build that is half the corpus. Held newest-first so a
        listing can stop as soon as it has enough.
        """
        ids, dates, cats, authors = [], [], [], []
        with self._db_lock:
            # Streamed and pooled, for the reasons given in reload().
            for row in self.db.execute(
                "SELECT id, categories, update_date, authors FROM papers "
                "ORDER BY update_date DESC"
            ):
                ids.append(row["id"])
                dates.append(self._shared(row["update_date"] or ""))
                # frozenset rather than set only so it can be pooled; the one
                # use is an intersection, which works the same either way.
                cats.append(
                    self._shared(frozenset((row["categories"] or "").split())))
                authors.append(
                    self._shared(textnorm.fold(row["authors"] or "")))
        self.meta_ids = ids
        self.meta_dates = dates
        self.meta_cats = cats
        self.meta_authors = authors

    def set_categories(self, categories) -> None:
        """Switch to other categories: on setting up, or on taking up an
        export's settings. The category masks are rebuilt with the matrix."""
        with self._db_lock:
            self.categories = list(categories)
            self._read_holdings()
            self.reload()

    def _read_holdings(self) -> None:
        """Which categories the index holds, as they bear on the reader's."""
        held = update_mod.cursors(self.db)
        self.missing = [c for c in self.categories if c not in held]
        # With nothing ticked a search covers the reader's categories -- which
        # on an index holding only those is everything, and needs no mask. An
        # index may hold others too, and those should not appear here.
        self.default_categories = (
            self.categories if set(held) - set(self.categories) else None)

    @staticmethod
    def _vector_file():
        """Which vector file is in place. `compact` swaps in a new one after
        renumbering the rows, with no commit to announce it."""
        try:
            return os.stat(config.VEC_PATH).st_ino
        except FileNotFoundError:
            return None

    def _data_version(self) -> int:
        # Moves whenever another connection -- another process, or the
        # server's own background runs -- commits to the database.
        return self.db.execute("PRAGMA data_version").fetchone()[0]

    def _fingerprints(self):
        """(vectors, metadata): cheap summaries that change when either does.

        The first covers which papers have vectors and in which slots, so it
        moves when a paper is embedded, re-embedded after a revision, replaced
        by a merge, or renumbered by `compact`; it reads only the index on
        `row`. The second covers what reload_metadata() holds -- how many
        papers, their dates, authors and categories -- and scans the table,
        tens of milliseconds, so it is only taken once data_version has said
        something changed.
        """
        vectors = self.db.execute(
            "SELECT COUNT(row), TOTAL(row) FROM papers WHERE row IS NOT NULL"
        ).fetchone()
        metadata = self.db.execute(
            "SELECT COUNT(*), TOTAL(length(authors)), TOTAL(length(categories)),"
            " TOTAL(julianday(update_date)) FROM papers").fetchone()
        return tuple(vectors), tuple(metadata)

    def refresh_if_stale(self) -> None:
        """Pick up whatever changed in the index since it was loaded.

        Called before every search and every few seconds by the server's
        watcher, so it has to be cheap when nothing changed: one pragma. When
        something did, only the half that changed is reloaded. During a build
        the vectors change every few seconds while the metadata does not, and
        re-folding 145k author strings each time would cost ~0.5s for nothing.
        Re-mapping the matrix is cheap: it is a memmap, so it does not copy.

        An index replaced outright -- `import --replace` -- is a new file, which
        the open connection would never see; it is reopened, provided it was
        built with the model this server searches with, and reloaded whole.
        """
        with self._db_lock:
            try:
                current = os.stat(config.DB_PATH).st_ino
            except FileNotFoundError:
                return          # mid-replacement; the next check finds it
            if current != self._db_file:
                fresh = store.connect(check_same_thread=False)
                try:
                    store.check_model(fresh)
                except SystemExit as exc:
                    fresh.close()
                    self._db_file = current     # say so once, not every check
                    print(f"The index was replaced, but not reloaded: {exc}",
                          flush=True)
                    return
                self.db.close()
                self.db, self._db_file, self._version = fresh, current, None
                self._vec_print = self._meta_print = None
            version, vec_file = self._data_version(), self._vector_file()
            if version == self._version and vec_file == self._vec_file:
                return
            if vec_file != self._vec_file:
                self._vec_print = None
            self._version, self._vec_file = version, vec_file
            vectors, metadata = self._fingerprints()
            # The lock is reentrant, so the reloads can retake it.
            if metadata != self._meta_print:
                # reload() carries per-paper metadata too, so both.
                self.reload()
                self.reload_metadata()
            elif vectors != self._vec_print:
                self.reload()
            self._vec_print, self._meta_print = vectors, metadata
            self._read_holdings()

    def stats(self) -> dict:
        self.refresh_if_stale()
        with self._db_lock:
            total = store.count_papers(self.db)
            pending = store.count_pending(self.db)
        return {
            "papers": total,
            "embedded": total - pending,
            "pending": pending,
            "model": config.model() if config.ready() else None,
            "categories": self.categories,
            "missing": self.missing,
        }

    def _mask(self, categories=None, since=None, until=None, author=None):
        """Boolean mask over the matrix rows, or None when nothing filters.

        This indexes the *matrix*, so it is built from `dates`, `cat_masks` and
        `authors` -- the embedded rows, in row order -- and never from the
        metadata tables, which are a different length and a different order.
        """
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
        if until:
            earlier = self.dates <= until
            keep = earlier if keep is None else (keep & earlier)
        terms = textnorm.fold_terms(author)
        if terms:
            by = np.fromiter(
                (textnorm.matches_terms(a, terms) for a in self.authors),
                dtype=bool, count=len(self.authors))
            keep = by if keep is None else (keep & by)
        return keep

    def query(self, text, k=20, categories=None, since=None, exclude=None,
              author=None, until=None):
        self.refresh_if_stale()
        started = time.monotonic()
        if not text:
            # No query to be similar to, so this is a metadata listing and has
            # no business consulting the vectors -- which may not exist yet.
            return (self.browse(k, categories, since, author, until),
                    time.monotonic() - started)
        if not self.ids:
            return [], 0.0

        keep = self._mask(categories, since, until, author)

        vector = search_mod.embed_query_normalised(text)
        scores = self.score(vector)

        if keep is not None:
            if not keep.any():
                return [], time.monotonic() - started
            # Push filtered-out rows below any real score rather than
            # compacting the array, which would cost a copy.
            scores = np.where(keep, scores, -np.inf)
        return self._pick(scores, k, exclude), time.monotonic() - started

    def _pick(self, scores, k, exclude=None) -> list:
        """The k best-scoring papers, best first, as dicts with their scores.
        Rows scored -inf are filtered out; `exclude` is an id to leave out."""
        best = [i for i in search_mod.top(scores, k + (1 if exclude else 0))
                if np.isfinite(scores[i]) and self.ids[i] != exclude][:k]
        found = self._meta([self.ids[i] for i in best])
        return [found[self.ids[i]] | {"score": float(scores[i])} for i in best]

    def _matching_rows(self, categories=None, since=None, until=None,
                       match_author=None):
        """Indices into the metadata tables, newest first, passing the filters.

        Covers papers with no vector yet, which matters during a build and for
        anything the embedder has not caught up with.

        The rows are held in update_date order, so a `since` bound stops the
        scan at the first row older than it instead of walking the rest of the
        corpus. That is what lets "everything by these authors in the last
        week" cost a few hundred comparisons rather than 146,000 -- and rows
        with no date at all sort last, where the same break discards them,
        which is right: an undated paper cannot be shown to fall in a window.

        `match_author` is a predicate on the folded author string rather than a
        term list, because the callers disagree about what several names mean:
        the author box ANDs them, a follow list unions them.
        """
        wanted = set(categories or ())
        for i, date in enumerate(self.meta_dates):
            if since and date < since:
                break
            if until and date > until:
                continue
            if wanted and not (wanted & self.meta_cats[i]):
                continue
            if match_author and not match_author(self.meta_authors[i]):
                continue
            yield i

    @staticmethod
    def _all_terms(author):
        """Predicate for the author box: every term must appear."""
        terms = textnorm.fold_terms(author)
        if not terms:
            return None
        return lambda folded: textnorm.matches_terms(folded, terms)

    def browse(self, k, categories=None, since=None, author=None, until=None):
        """Newest-first listing by metadata alone, across the whole corpus."""
        chosen = []
        for i in self._matching_rows(categories, since, until,
                                     self._all_terms(author)):
            chosen.append(self.meta_ids[i])
            if len(chosen) >= k:
                break
        if not chosen:
            return []
        meta = self._meta(chosen)
        # No relevance score exists here; null keeps the UI from showing a
        # number that would mean nothing.
        return [meta[i] | {"score": None} for i in chosen]

    def count_matching(self, categories=None, since=None, author=None,
                       until=None) -> int:
        """How many papers match these filters, embedded or not.

        Used to explain an empty relevance search: during a build the filters
        may well select papers that simply have no vector yet.
        """
        return sum(1 for _ in self._matching_rows(
            categories, since, until, self._all_terms(author)))

    def followed(self, authors, k=500, categories=None, since=None,
                 until=None):
        """Newest-first listing of papers by *any* of `authors` in the window.

        The union is the point, and it is where this parts company with the
        author box: that ANDs its names, because "Hardy, Littlewood" asks for
        their joint work. A follow list is the other thing -- one name per
        line, each worth seeing on its own -- so the lines are OR-ed.

        Returns (results, total). The cap is a display limit, and a listing
        that was truncated has to be able to say so rather than look complete.
        """
        self.refresh_if_stale()
        groups = [terms for terms in
                  (textnorm.fold_terms(name) for name in authors) if terms]
        if not groups:
            return [], 0

        def match(folded):
            return any(textnorm.matches_terms(folded, terms)
                       for terms in groups)

        chosen, total = [], 0
        for i in self._matching_rows(categories, since, until, match):
            total += 1
            if len(chosen) < k:
                chosen.append(self.meta_ids[i])
        if not chosen:
            return [], 0
        meta = self._meta(chosen)
        return [meta[i] | {"score": None} for i in chosen], total

    def ranked(self, queries, weights, blend, k=20, categories=None,
               since=None, until=None):
        """The window's papers ordered by closeness to a set of interests.

        `queries` is a stack of unit interest vectors, `weights` the weight
        beside each. A paper is scored against every interest, those scores are
        multiplied by their weights and sorted best-first, and the result is
        summed under profile.blend_decay -- so the best match counts in full
        and each further one counts less. `blend` picks how much less: 0 scores
        a paper by its single best interest, 1 by all of them equally.

        Only embedded papers can appear -- ranking needs a vector -- so the
        caller is left to say how much of the window is still waiting.
        """
        self.refresh_if_stale()
        if queries is None or not len(queries) or not self.ids:
            return [], 0.0
        started = time.monotonic()

        keep = self._mask(categories, since, until)
        if keep is not None and not keep.any():
            return [], time.monotonic() - started

        # (papers, interests), filled in a single pass over the matrix.
        per = np.atleast_2d(self.score(np.ascontiguousarray(queries.T)))
        if per.shape[0] != len(self.ids):  # a lone interest comes back 1-D
            per = per.reshape(len(self.ids), -1)
        per *= np.asarray(weights, dtype=np.float32)
        per.sort(axis=1)
        scores = per[:, ::-1] @ profile_mod.blend_decay(blend, per.shape[1])

        # -inf, since weights scale these past [-1, 1].
        if keep is not None:
            scores = np.where(keep, scores, -np.inf)
        return self._pick(scores, max(k, 1)), time.monotonic() - started

    # --- When the index was last topped up ---------------------------------
    # The profile and the schedule setting live in the settings file and need
    # no connection; this record is about the index, so it stays in it.

    def last_run(self) -> float:
        with self._db_lock:
            return schedule_mod.last_run(self.db)

    def note_run(self, when) -> None:
        with self._db_lock:
            schedule_mod.note_run(self.db, when)

    def newest_date(self) -> str:
        """The most recent update_date held. The rows are date-sorted."""
        return self.meta_dates[0] if self.meta_dates else ""

    def _meta(self, ids) -> dict:
        with self._db_lock:
            return search_mod.papers(self.db, ids)

    def similar(self, paper_id, k=20):
        """Papers closest to a given one; (None, 0) if it has no vector."""
        self.refresh_if_stale()
        found = self._rows("SELECT row FROM papers WHERE id = ?", (paper_id,))
        if not found or found[0]["row"] is None:
            return None, 0.0
        started = time.monotonic()
        scores = self.score(store.read_vector(found[0]["row"]))
        return self._pick(scores, k, paper_id), time.monotonic() - started


class Updater:
    """Runs `update` in the background, for the UI's "Fetch new papers" button,
    or just the embedding, for its "Embed them now" (papers imported with
    `build --scan-only` and not embedded yet), or an import uploaded from the
    settings panel: the arXiv snapshot, or an exported index.

    An import reads its upload off the request that carries it, which has to
    stay open until then; see _Upload.

    A top-up walks the arXiv API and then embeds what came back. A week's
    worth is three or four pages and about a minute all told, most of it
    embedding; a long absence is many minutes of paging at one request per
    three seconds. Either way it is far too long to hold an HTTP request open,
    so the request that starts one returns immediately and the page polls for
    progress instead. Only one runs at a time; a second click while one is in
    flight is refused rather than queued.

    The thread opens its *own* SQLite connection rather than borrowing the
    resident index's. WAL lets one writer and many readers coexist, whereas
    holding the index lock for the length of a run would stall every search
    behind it. Nothing else is needed to make the new papers searchable: the
    watcher, or the next query, calls refresh_if_stale(), which sees the
    commits and re-maps.
    """

    KEEP_LINES = 200        # a normal run prints a handful; a backlog, more

    def __init__(self, index=None):
        self._index = index         # told of categories an import brings
        self._lock = threading.Lock()
        self._thread = None
        self.state = "idle"         # idle | running | done | failed
        self.kind = "update"        # update | embed | snapshot | index
        self.upload = None          # an import's _Upload
        self.imported = 0           # papers in scope, once a snapshot is read
        self.matched = 0            # the same, so far, while it is read
        self.lines = collections.deque(maxlen=self.KEEP_LINES)
        self.started = None
        self.finished = None
        self.embedded = 0
        self.progress = None        # (done, total) while embedding
        self.error = None

    def _alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _in_flight(self) -> bool:
        """Whether a run is still going.

        Both halves are needed. `_run` records its outcome and only then
        returns, so between those two moments the thread is alive but the run
        is over -- asking liveness alone would refuse a click made in that
        window, which is exactly when someone who just watched a run finish
        clicks again.
        """
        return self.state == "running" and self._alive()

    def start(self, kind: str = "update", upload=None, embed=True,
              mode="merge", categories=(), take_settings=False) -> bool:
        """Kick off a run. False if one is already going.

        One at a time whatever the kind: they all write to the index, and two
        embedding runs would only have the second refused by the embed lock.
        A snapshot import scans for `categories` and embeds afterwards if
        `embed`; an index import merges or replaces as `mode` says.
        """
        with self._lock:
            if self._in_flight():
                return False
            self.state, self.kind = "running", kind
            self.upload, self._embed, self._mode = upload, embed, mode
            self._categories = list(categories)
            self._take_settings = take_settings
            self.imported = self.matched = 0
            self.lines.clear()
            self.started = time.time()
            self.finished = None
            self.embedded = 0
            self.progress = None
            self.error = None
            # Daemonic, so Ctrl-C on the server is not held hostage by a long
            # embedding run. Nothing is lost by cutting one short: batches are
            # committed as they land, and `update` advances the cursor only
            # after the work is stored, so the next run resumes from there.
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            return True

    def _log(self, *args) -> None:
        text = " ".join(str(a) for a in args).rstrip()
        with self._lock:
            self.lines.extend(text.split("\n"))
        # The terminal running `serve` keeps seeing the run as it always did.
        print(text, flush=True)

    def _progress(self, done: int, total: int) -> None:
        with self._lock:
            self.progress = (done, total)

    def _run(self) -> None:
        db = None
        try:
            if self.kind == "index":
                # Its own connections: a replaced index must not be held open.
                embedded = self._import_index()
            else:
                db = store.connect()
                store.check_model(db)
                if self.kind == "snapshot":
                    embedded = self._import_snapshot(db)
                elif self.kind == "embed":
                    embedded = ingest.embed_pending(db, log=self._log,
                                                    progress=self._progress)
                else:
                    embedded = update_mod.update(db, log=self._log,
                                                 progress=self._progress)
        except (Exception, SystemExit) as exc:  # noqa: BLE001
            # SystemExit deliberately included: the embed lock, the model check
            # and the Ollama probe all raise it to end a CLI run, and none of
            # them is a reason to take the server down.
            message = str(exc) or exc.__class__.__name__
            with self._lock:
                self.state, self.error = "failed", message
                self.finished, self.progress = time.time(), None
            self._log(f"{self.kind} failed: {message}")
        else:
            with self._lock:
                self.state, self.embedded = "done", embedded
                self.finished, self.progress = time.time(), None
        finally:
            if self.upload is not None:
                # Whatever happened, the request carrying it can now answer.
                self.upload.finished()
            if db is not None:
                db.close()

    def _import_snapshot(self, db) -> int:
        """Scan the uploaded snapshot for the reader's categories not held yet,
        then embed if asked. Returns the count embedded."""
        imported = ingest.scan_lines(
            db, self._categories, self.upload, self.upload.name,
            log=self._log, progress=self._matched)
        with self._lock:
            self.imported = imported
        self.upload.finished()
        if not self._embed:
            return 0
        return ingest.embed_pending(db, log=self._log, progress=self._progress)

    def _import_index(self) -> int:
        """Import the uploaded export. Into an empty index it brings its own
        model and categories; see transfer.import_stream."""
        transfer.import_stream(
            self.upload, self.upload.name, replace=self._mode == "replace",
            merge=self._mode == "merge", take_settings=self._take_settings,
            log=self._log, on_read=self.upload.finished)
        if self._index is not None:
            # A replaced index is a new file: open it before reloading.
            self._index.refresh_if_stale()
            # The categories are the one setting the server reads once; the
            # rest are read afresh wherever they are used.
            self._index.set_categories(settings.categories())
        return 0

    def _matched(self, matched: int) -> None:
        with self._lock:
            self.matched = matched

    def snapshot(self) -> dict:
        with self._lock:
            state = self.state
            if state == "running" and not self._alive():
                # The thread went without recording an outcome -- only a
                # BaseException other than SystemExit can do that. Say so
                # rather than leaving the page polling a run that is over.
                state = self.state = "failed"
                self.error = self.error or "the update thread stopped"
            payload = {
                "state": state,
                "kind": self.kind,
                "lines": list(self.lines),
                "embedded": self.embedded,
                "imported": self.imported,
                "error": self.error,
            }
            if self.started:
                payload["elapsed"] = round(
                    (self.finished or time.time()) - self.started)
            if state == "running" and self.progress:
                done, total = self.progress
                payload["progress"] = {"done": done, "total": total}
            elif (state == "running" and self.upload is not None
                    and not self.upload.done.is_set()):
                payload["read"] = {"done": self.upload.read_bytes,
                                   "total": self.upload.total,
                                   "matched": self.matched}
        return payload


class _Upload:
    """An import's file as it arrives in a request body.

    Read straight off the socket and never held whole: a snapshot is 5.5 GB.
    Serves both readers -- line by line for the snapshot scan, `read(n)` for
    tarfile's stream mode -- and never reads past the body. A connection that
    closes early raises rather than ending the file, so neither reader can
    mistake half an upload for all of it.

    `done` is set once the upload is no longer being read, which is when the
    request can answer: until then its body is still in the socket.
    """

    LINE_CAP = 1 << 24      # a snapshot line is a few KB; this is a backstop

    def __init__(self, rfile, length: int, name: str):
        self.rfile, self.total, self.name = rfile, length, name
        self.read_bytes = 0
        self.done = threading.Event()

    def _short(self):
        return ConnectionError(f"the upload stopped after {self.read_bytes:,} "
                               f"of {self.total:,} bytes")

    def read(self, size: int = -1) -> bytes:
        left = self.total - self.read_bytes
        if size < 0 or size > left:
            size = left
        if not size:
            return b""
        data = self.rfile.read(size)
        if not data:
            raise self._short()
        self.read_bytes += len(data)
        return data

    def __iter__(self):
        while self.read_bytes < self.total:
            line = self.rfile.readline(
                min(self.total - self.read_bytes, self.LINE_CAP))
            if not line:
                raise self._short()
            self.read_bytes += len(line)
            yield line.decode("utf-8")

    def finished(self) -> None:
        """The reader is done with it. What little is left (a tar's closing
        padding) is read off, so the request can answer on a clean socket."""
        if self.done.is_set():
            return
        try:
            if self.total - self.read_bytes <= transfer.RECORD:
                while self.read(1 << 16):
                    pass
        except OSError:
            pass
        self.done.set()


class Scheduler:
    """Presses "Fetch new papers" on a timer, for as long as `serve` is up.

    The button exists because an index goes stale behind a server that is left
    running; this is the same button, pressed by the clock instead. All of the
    "when" lives in `schedule`, as pure functions over the setting and two
    timestamps -- this class only supplies the clock, the thread and the
    refusal to start a second run on top of a first.

    Each tick re-reads the setting, so changing it in the UI takes effect
    within the tick rather than at the next restart. The read is a small file,
    which is why polling is affordable enough to keep the alternative --
    waking exactly at the due moment, and rearming whenever the setting
    changes -- from being worth its extra machinery.

    Runs are timed from when one last *started*, manual runs included: someone
    who has just fetched by hand does not want the clock doing it again a
    moment later. That record is persisted, so it survives a restart.
    """

    TICK = 30.0     # the finest the setting can express is an hour

    def __init__(self, index, updater):
        self._index = index
        self._updater = updater
        self._stop = threading.Event()
        self._thread = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        # wait() rather than sleep(): a stop lands within the tick instead of
        # at the end of it. The first check is one tick in, which keeps the
        # scheduler off the back of a server still loading its index.
        while not self._stop.wait(self.TICK):
            try:
                self.tick()
            # SettingsError is a SystemExit, and would otherwise end the thread
            # quietly over a typo in the settings file.
            except (Exception, settings.SettingsError) as exc:  # noqa: BLE001
                print(f"auto-update check failed: {exc}", flush=True)

    def _last(self) -> float:
        """When a run last started, by either route."""
        return max(self._index.last_run(), self._updater.started or 0)

    def tick(self, now=None) -> bool:
        """Start a run if one is due. Returns whether it did."""
        now = time.time() if now is None else now
        setting = schedule_mod.load()
        if setting["mode"] == "off":
            return False
        if not schedule_mod.due(setting, self._last(), now):
            return False
        if not self._updater.start():
            # One is already in flight -- a long backlog, or a manual run
            # started seconds ago. Leave the record alone and ask again next
            # tick, by which time that run will have set it.
            return False
        self._index.note_run(now)
        print("auto-update: starting a scheduled run", flush=True)
        return True

    def status(self, now=None) -> dict:
        now = time.time() if now is None else now
        setting = schedule_mod.load()
        last = self._last()
        return setting | {
            "last_run": last or None,
            "next_run": schedule_mod.next_run(setting, last, now),
            "now": now,
        }


# A profile is two short text fields; anything near this is a mistake.
MAX_BODY = 256 * 1024

GRACE_PERIOD = 10.0     # seconds to let in-flight requests finish on shutdown


class GracefulHTTPServer(ThreadingHTTPServer):
    """Threading server that lets in-flight requests finish before it closes.

    Handler threads stay daemonic on purpose. With HTTP/1.1 keep-alive most of
    them sit blocked on a read from an idle browser connection, and joining
    those -- what ``daemon_threads = False`` would do -- would stall the exit
    for as long as a tab stays open. What matters for a clean stop is the
    requests actually being served, so those are counted here and waited on for
    a bounded time instead.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._in_flight = 0
        self._idle = threading.Condition()

    def request_started(self):
        with self._idle:
            self._in_flight += 1

    def request_finished(self):
        with self._idle:
            self._in_flight -= 1
            if not self._in_flight:
                self._idle.notify_all()

    def drain(self, timeout: float = GRACE_PERIOD) -> int:
        """Wait for in-flight requests; return how many were still running."""
        deadline = time.monotonic() + timeout
        with self._idle:
            while self._in_flight:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self._idle.wait(remaining):
                    break
            return self._in_flight


def make_handler(index: ResidentIndex, updater: Updater,
                 scheduler: "Scheduler"):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # quieter than the default
            pass

        def _local(self) -> bool:
            """Whether the page was opened on this machine. Importing and
            exporting move files between the browser and the index, which is
            only offered there."""
            try:
                return ipaddress.ip_address(self.client_address[0]).is_loopback
            except ValueError:
                return False

        def _trusted(self) -> bool:
            """On this machine, and not a request from another site open in
            the same browser: a cross-site request carries its Origin."""
            origin = self.headers.get("Origin")
            return self._local() and (
                not origin or urlparse(origin).netloc == self.headers.get("Host"))

        def _receive(self, kind: str, query) -> None:
            """Take an uploaded import and start it, answering once the upload
            has been read. What follows -- the embedding, a merge -- carries on
            in the background, watched through GET /api/update."""
            # The body may be left unread (a refusal, a failed run); the
            # connection cannot be reused after that.
            self.close_connection = True
            if not self._trusted():
                self._json({"error": "Importing is only offered on the "
                                     "machine running the server."}, 403)
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = 0
            if length <= 0:
                self._json({"error": "No file was sent."}, 411)
                return
            if kind == "snapshot":
                index.refresh_if_stale()
                if not index.missing:
                    self._json({"error": "The index already holds all your "
                                         "categories, so there is nothing to "
                                         "import from the snapshot."}, 409)
                    return
            name = (query.get("name") or ["the upload"])[0]
            upload = _Upload(self.rfile, length, name)
            # The server's own categories, not the settings file's: they
            # are what its searches and checkboxes know.
            if not updater.start(kind, upload=upload,
                                 embed=query.get("embed") == ["1"],
                                 mode=(query.get("mode") or ["merge"])[0],
                                 categories=index.missing,
                                 take_settings=query.get("settings") == ["1"]):
                self._json(updater.snapshot() |
                           {"error": "An update is already running."}, 409)
                return
            upload.done.wait()
            self._json(updater.snapshot())

        def _choose_model(self, model) -> None:
            """Choose the embedding model of an index with no papers yet."""
            if not self._trusted():
                self._json({"error": "Setting up is only offered on the "
                                     "machine running the server."}, 403)
                return
            if config.ready() and model == config.model():
                self._json({"model": model})
                return
            if index.stats()["papers"]:
                self._json({"error": "The index already has papers, so its "
                                     "embedding model cannot change."}, 409)
                return
            try:
                chosen = next((m for m in embedder.embedding_models()
                               if m["name"] == model), None)
            except SystemExit as exc:
                self._json({"error": str(exc)}, 502)
                return
            if chosen is None:
                self._json({"error": f"{model} is not an installed embedding "
                                     "model."}, 400)
                return
            # The default's prompts are known; another model gets none.
            config.use({"model": model}
                       if model == config.DEFAULT_EMBEDDING["model"]
                       else {"model": model, "dim": chosen["dim"]})
            self._json({"model": model})

        def _export(self) -> None:
            """Stream the index out as an export, the same file `export` writes."""
            if not self._trusted():
                self._json({"error": "Exporting is only offered on the "
                                     "machine running the server."}, 403)
                return
            query = parse_qs(urlparse(self.path).query)
            try:
                with transfer.exporting(query.get("settings") == ["1"]) as ex:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/x-tar")
                    self.send_header("Content-Length", str(ex.size))
                    self.send_header(
                        "Content-Disposition", "attachment; filename="
                        f'"arxiv_index-{dt.date.today().isoformat()}.tar"')
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    try:
                        ex.write(self.wfile)
                    except OSError:
                        # The download was cancelled; nothing to clean up
                        # beyond what exporting() does.
                        self.close_connection = True
            except SystemExit as exc:
                # No index, or an embedding run holds the lock. Raised before
                # the headers, so it can still be answered.
                self._json({"error": str(exc)}, 409)

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
            """Serve a file from STATIC, refusing anything outside it."""
            target = (STATIC / unquote(rel)).resolve()
            if not target.is_file() or STATIC not in target.parents:
                self._send(b"not found", "text/plain", 404)
                return
            kind, _ = mimetypes.guess_type(target.name)
            if target.suffix == ".woff2":
                kind = "font/woff2"          # not in every mimetypes database
            elif target.suffix == ".js":
                kind = "application/javascript"
            elif target.suffix == ".html":
                kind = "text/html; charset=utf-8"
            body = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", kind or "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            # KaTeX's files never change without a new version of it, so they
            # are cached for a week. The app's own change with the code, and a
            # browser holding yesterday's copy would silently hide new UI, so
            # those are fetched afresh every time -- they are local and small.
            vendored = ".min." in target.name or target.suffix == ".woff2"
            self.send_header("Cache-Control",
                             "public, max-age=604800, immutable" if vendored
                             else "no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            # Counted rather than hooking handle_one_request, which spends most
            # of its life blocked waiting for the *next* request on an idle
            # keep-alive connection -- that is not work worth draining for.
            self.server.request_started()
            try:
                self._route()
            except settings.SettingsError as exc:
                # A hand edit broke the file while the server was up.
                self._json({"error": str(exc)}, 500)
            finally:
                self.server.request_finished()

        def do_POST(self):
            self.server.request_started()
            try:
                # Uploads first, before the body is read: it is the file.
                target = urlparse(self.path)
                if target.path in ("/api/import/snapshot", "/api/import/index"):
                    self._receive(target.path.rsplit("/", 1)[1],
                                  parse_qs(target.query))
                    return
                # The body is always read, even where it is not wanted:
                # leaving it in the socket desynchronises the keep-alive
                # connection. Capped so a stray upload cannot be buffered whole.
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    length = 0
                if length > MAX_BODY:
                    self._json({"error": "request body too large"}, 413)
                    return
                raw = self.rfile.read(length) if length else b""
                body = None
                if raw:
                    try:
                        body = json.loads(raw)
                    except ValueError:
                        body = None
                if body is not None and not isinstance(body, dict):
                    body = None
                path = urlparse(self.path).path
                if path == "/api/setup/categories":
                    if not self._trusted():
                        self._json({"error": "Setting up is only offered on "
                                             "the machine running the server."},
                                   403)
                        return
                    try:
                        chosen = settings.parse_categories(
                            str((body or {}).get("categories", "")))
                    except ValueError as exc:
                        self._json({"error": str(exc)}, 400)
                        return
                    if index.stats()["papers"] and chosen != index.categories:
                        # The masks the page filters with are built per
                        # category at start; see ResidentIndex.categories. The
                        # same list again is fine: a retry after a failed import.
                        self._json({"error": "The index already has papers; "
                                             "change the categories in the "
                                             "settings file and restart."}, 409)
                        return
                    settings.update(categories=chosen)
                    index.set_categories(chosen)
                    self._json({"categories": chosen})
                    return

                if path == "/api/setup/model":
                    self._choose_model((body or {}).get("model"))
                    return

                if path == "/api/profile":
                    if body is None:
                        self._json({"error": "expected a JSON body"}, 400)
                        return
                    authors = body.get("authors", [])
                    interests = body.get("interests", [])
                    if not isinstance(authors, list) or not isinstance(
                            interests, list):
                        self._json({"error": "authors and interests must both "
                                             "be lists"}, 400)
                        return
                    saved, error = profile_mod.save(
                        authors, interests, body.get("blend"))
                    if error:
                        # The text is stored either way; only the embedding
                        # failed, so this is a warning on a successful save
                        # rather than a failed request.
                        saved = saved | {
                            "warning": f"Interests saved, but embedding them "
                                       f"failed: {error}"}
                    self._json(saved)
                    return

                if path == "/api/schedule":
                    if body is None:
                        self._json({"error": "expected a JSON body"}, 400)
                        return
                    schedule_mod.save(body)
                    self._json(scheduler.status())
                    return

                if path == "/api/update":
                    # POST, not GET: this walks the arXiv API and writes to the
                    # index, which is no business of a reload or a prefetch.
                    if not updater.start():
                        self._json(updater.snapshot() |
                                   {"error": "An update is already running."},
                                   409)
                        return
                    # A manual run resets the clock too, so the scheduler does
                    # not follow it with one of its own minutes later.
                    index.note_run(time.time())
                    self._json(updater.snapshot())
                    return

                if path == "/api/embed":
                    # Progress is read from GET /api/update, as for a fetch.
                    # Not a fetch, so the schedule's clock is left alone.
                    if not updater.start("embed"):
                        self._json(updater.snapshot() |
                                   {"error": "An update is already running."},
                                   409)
                        return
                    self._json(updater.snapshot())
                    return
                self._send(b"not found", "text/plain", 404)
            except settings.SettingsError as exc:
                self._json({"error": str(exc)}, 500)
            finally:
                self.server.request_finished()

        def _stale_hint(self, since):
            """Why a window came back empty, when the reason is the index.

            A range that starts after the newest paper held cannot match
            anything, and "No matches" reads as "nothing was posted" rather
            than "this index stopped three weeks ago" -- which is the common
            case for the default last-7-days window on an index that has not
            been updated in a while.
            """
            newest = index.newest_date()
            if since and newest and since > newest:
                return (f"The index holds nothing newer than {newest}. "
                        f"Fetch new papers to bring it up to date.")
            return None

        def _window(self, params, one):
            """(categories, since, until, k) for the listing endpoints.

            With no category ticked, `categories` is the reader's own when the
            index holds others as well; see ResidentIndex.default_categories.
            Returns None after answering with a 400, so the caller just stops.
            """
            cats = [c for c in params.get("cat", [])
                    if c in index.categories] or index.default_categories
            since = one("since") or None
            until = one("until") or None
            for label, value in (("since", since), ("until", until)):
                if value and not _valid_date(value):
                    self._json({"error": f"bad {label} date: {value}"}, 400)
                    return None
            if since and until and since > until:
                self._json({"error": f"{since} is after {until}"}, 400)
                return None
            return cats, since, until, _count(one)

        def _route(self):
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            one = lambda key, default=None: params.get(key, [default])[0]

            if parsed.path == "/":
                # An index with no papers yet gets the setup page instead:
                # there is nothing to search.
                self._static("setup.html" if index.stats()["papers"] == 0
                             else "index.html")
                return

            if parsed.path == "/api/setup":
                try:
                    models, problem = embedder.embedding_models(), None
                except SystemExit as exc:
                    models, problem = [], str(exc)
                self._json({"categories": index.categories,
                            "model": config.model() if config.ready() else None,
                            "default": config.DEFAULT_EMBEDDING["model"],
                            "models": models,
                            "ollama_error": problem,
                            "local": self._local(),
                            "papers": index.stats()["papers"]})
                return

            if parsed.path.startswith("/static/"):
                self._static(parsed.path[len("/static/"):])
                return

            if parsed.path == "/api/stats":
                self._json(index.stats() | {"local": self._local()})
                return

            if parsed.path == "/api/export":
                self._export()
                return

            if parsed.path == "/api/update":
                # Progress of the current or most recent run. The server is the
                # only source of truth about whether one is in flight, so a
                # reloaded page asks here rather than assuming it is idle.
                self._json(updater.snapshot())
                return

            if parsed.path == "/api/schedule":
                self._json(scheduler.status())
                return

            if parsed.path == "/api/profile":
                self._json(profile_mod.load())
                return

            if parsed.path == "/api/followed":
                window = self._window(params, one)
                if window is None:
                    return
                cats, since, until, _ = window
                authors = profile_mod.load()["authors"]
                if not authors:
                    self._json({"error": "No followed authors yet. Add some "
                                         "under ⚙."}, 400)
                    return
                started = time.monotonic()
                # Everything in the window, not the page's result count: the
                # question is "what did these people post", which a top-20
                # would answer wrongly by dropping the rest without saying so.
                results, total = index.followed(
                    authors, k=500, categories=cats, since=since, until=until)
                payload = {"results": results,
                           "ms": round((time.monotonic() - started) * 1000),
                           "ranked": "date", "total": total,
                           "authors": len(authors)}
                if not results:
                    payload["hint"] = self._stale_hint(since) or (
                        f"No papers by your {len(authors)} followed author(s) "
                        f"{_scope(since, until)}.")
                self._json(payload)
                return

            if parsed.path == "/api/interests":
                window = self._window(params, one)
                if window is None:
                    return
                cats, since, until, k = window
                # Picks up interests typed into the settings file by hand. A
                # no-op when every entry is already cached.
                profile_mod.embed_missing()
                profile = profile_mod.load()
                if not profile["interests"]:
                    self._json({"error": "No research interests yet. Describe "
                                         "them under ⚙."}, 400)
                    return
                queries, weights = profile_mod.vectors()
                if queries is None:
                    # Either nothing is embedded yet, or every entry that is
                    # has been turned off with a zero weight. Both leave
                    # nothing to rank by, but they are not the same mistake.
                    self._json({"error": (
                        "None of your interests can be ranked by: they are "
                        "all switched off with a weight of 0."
                        if profile["embedded"] else
                        "Your interests have not been embedded yet -- save "
                        "them again once Ollama is reachable.")}, 409)
                    return
                results, elapsed = index.ranked(
                    queries, weights, profile["blend"], k,
                    categories=cats, since=since, until=until)
                payload = {"results": results, "ms": round(elapsed * 1000),
                           "ranked": "relevance",
                           "interests": len(weights)}
                if not results:
                    # Same trap as an empty relevance search: the window may be
                    # full of papers that simply have no vector to rank yet.
                    waiting = index.count_matching(cats, since, until=until)
                    if waiting:
                        payload["hint"] = (
                            f"{waiting:,} paper(s) fall {_scope(since, until)} "
                            "but are not embedded yet, so they cannot be "
                            "ranked. Fetch new papers, or list them by date "
                            "instead."
                        )
                    else:
                        payload["hint"] = self._stale_hint(since) or (
                            f"Nothing {_scope(since, until)}.")
                self._json(payload)
                return

            if parsed.path == "/api/search":
                query = (one("q") or "").strip()
                author = (one("author") or "").strip()
                window = self._window(params, one)
                if window is None:
                    return
                cats, since, until, k = window
                # Any single criterion is a valid search on its own -- an
                # author, a category or a date each describe a listing. Only a
                # request with no criteria at all is rejected, matching the CLI.
                # Ticked boxes, not `cats`, which may be the default scope.
                ticked = any(c in index.categories
                             for c in params.get("cat", []))
                if not (query or author or since or until or ticked):
                    self._json(
                        {"error": "give a query, author, category or date"}, 400)
                    return
                try:
                    results, elapsed = index.query(
                        query, k, cats, since, author=author or None,
                        until=until)
                except Exception as exc:  # surfaced in the UI, not swallowed
                    self._json({"error": str(exc)}, 500)
                    return
                payload = {"results": results, "ms": round(elapsed * 1000),
                           "ranked": "relevance" if query else "date"}
                if query and not results:
                    # An empty relevance search is confusing while a build is
                    # running: the filters may match plenty of papers that
                    # simply have no vector to rank yet.
                    waiting = index.count_matching(cats, since, author or None,
                                                   until=until)
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
                results, elapsed = index.similar(paper_id, _count(one))
                if results is None:
                    self._json({"error": f"{paper_id} has no vector yet"}, 404)
                    return
                self._json({"results": results, "ms": round(elapsed * 1000)})
                return

            self._send(b"not found", "text/plain", 404)

    return Handler


def _scope(since, until) -> str:
    """How to refer to the window in a message, now that it may be unbounded.

    Both date fields empty means the whole index, and "in this range" would
    then be describing a range the reader never set.
    """
    return "in this range" if (since or until) else "anywhere in the index"


def _count(one) -> int:
    """The `k` a request asks for. Generous, because listing a prolific
    author's whole output is a legitimate request (Sturmfels has 217)."""
    try:
        return max(1, min(500, int(one("k", "20"))))
    except ValueError:
        return 20


def _valid_date(text: str) -> bool:
    try:
        dt.datetime.strptime(text, "%Y-%m-%d")
        return True
    except ValueError:
        return False


WATCH_INTERVAL = 5.0    # seconds between looks at the index for changes


def _watch(index: ResidentIndex) -> None:
    """Keep the resident index in step with the files, whoever changes them:
    `update` from cron, a `build` or `import` in a terminal, a run started from
    this page. Searches check too, but a change picked up here is loaded before
    anyone searches, and the page's counts are right without one."""
    while True:
        time.sleep(WATCH_INTERVAL)
        try:
            index.refresh_if_stale()
        except Exception as exc:  # noqa: BLE001 - keep watching regardless
            print(f"Could not reload the index: {exc}", flush=True)


def serve(port: int = 8000, host: str = "127.0.0.1", open_browser: bool = True):
    print("Loading index ...")
    index = ResidentIndex()
    stats = index.stats()
    # Computed rather than read off the matrix, which is gone on the GPU path.
    size = len(index.ids) * config.slot_bytes() / 1e6 if index.ids else 0
    where = "VRAM" if index.gpu is not None else "RAM"
    print(f"{stats['embedded']:,} papers resident ({size:,.0f} MB in {where})"
          + (f", {stats['pending']:,} still embedding" if stats["pending"] else ""))

    updater = Updater(index)
    scheduler = Scheduler(index, updater)
    server = GracefulHTTPServer((host, port),
                                make_handler(index, updater, scheduler))
    url = f"http://{host}:{port}/"
    setting = schedule_mod.load()
    if setting["mode"] == "interval":
        print(f"Automatic updates: every {setting['hours']:g}h")
    elif setting["mode"] == "daily":
        print(f"Automatic updates: daily at {setting['at']}")
    print(f"\n  {url}\n\nCtrl-C (or SIGTERM) to stop.")
    scheduler.start()
    threading.Thread(target=_watch, args=(index,), daemon=True).start()
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    stopping = threading.Event()

    def stop(signum, _frame):
        # Signal handlers run in the main thread, which is the one parked in
        # serve_forever(); calling shutdown() from here would deadlock waiting
        # on itself, so hand it to a helper thread. A second signal is ignored
        # rather than escalated -- the grace period already bounds the wait.
        if stopping.is_set():
            return
        stopping.set()
        print(f"\n{signal.Signals(signum).name} -- finishing in-flight "
              f"requests ...", flush=True)
        threading.Thread(target=server.shutdown, daemon=True).start()

    try:
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
    except ValueError:
        pass            # not the main thread: fall back to KeyboardInterrupt

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()

    scheduler.stop()
    cut_short = server.drain()
    server.server_close()
    print(f"Stopped, {cut_short} request(s) cut short."
          if cut_short else "Stopped.")
