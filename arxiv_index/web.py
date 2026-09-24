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
import html
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

from . import (cite, config, ingest, profile as profile_mod,
               schedule as schedule_mod, search as search_mod, settings,
               store, textnorm, update as update_mod)

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
        self._db_file = os.stat(config.DB_PATH).st_ino
        self._vec_file = self._vector_file()
        # Fixed for the life of the server: the category masks below are built
        # from this list, so a checkbox for a category added to the settings
        # since would filter on a mask that does not exist.
        self.categories = settings.categories()
        self._read_holdings()
        self.ids = []
        self.matrix = None
        self.loaded = 0
        self.meta_loaded = -1
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
        self.loaded = len(ids)
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
        self.meta_loaded = len(ids)

    def _read_holdings(self) -> None:
        """Which categories the index holds, as they bear on the reader's."""
        held = update_mod.cursors(self.db)
        self.missing = [c for c in self.categories if c not in held]
        # With nothing ticked a search covers the reader's categories -- which
        # on an index holding only those is everything, and needs no mask. A
        # shared index may hold others', and those should not appear here.
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
            "model": config.MODEL,
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
        if not self.ids:
            return [], 0.0

        started = time.monotonic()

        if not text:
            # No query to be similar to, so this is a metadata listing and has
            # no business consulting the vectors at all.
            return (self.browse(k, categories, since, author, until),
                    time.monotonic() - started)

        keep = self._mask(categories, since, until, author)

        vector = search_mod.embed_query_normalised(text)
        scores = self.score(vector)

        if keep is not None:
            if not keep.any():
                return [], time.monotonic() - started
            # Push filtered-out rows below any real cosine rather than
            # compacting the array, which would cost a copy.
            scores = np.where(keep, scores, -2.0)

        want = min(k + (1 if exclude else 0), len(self.ids))
        top = np.argpartition(-scores, want - 1)[:want]
        top = top[np.argsort(-scores[top])]

        chosen = [self.ids[i] for i in top
                  if scores[i] > -2.0 and self.ids[i] != exclude][:k]
        if not chosen:
            return [], time.monotonic() - started

        by_id = {self.ids[i]: float(scores[i]) for i in top}
        meta = self._meta(chosen)
        return ([meta[i] | {"score": by_id[i]} for i in chosen],
                time.monotonic() - started)

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

        # -inf rather than a sentinel below every real score: weights scale
        # these past [-1, 1], so no finite floor is safe to assume any more.
        if keep is not None:
            scores = np.where(keep, scores, -np.inf)

        want = min(max(k, 1), len(self.ids))
        top = np.argpartition(-scores, want - 1)[:want]
        top = top[np.argsort(-scores[top])]
        chosen = [self.ids[i] for i in top if np.isfinite(scores[i])]
        if not chosen:
            return [], time.monotonic() - started
        by_id = {self.ids[i]: float(scores[i]) for i in top}
        meta = self._meta(chosen)
        return ([meta[i] | {"score": by_id[i]} for i in chosen],
                time.monotonic() - started)

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

    def similar(self, paper_id, k=20):
        """Papers closest to a given one."""
        self.refresh_if_stale()
        vector = self.vector_for(paper_id)
        if vector is None:
            return None, 0.0
        started = time.monotonic()
        scores = self.score(vector)
        # +1 because the paper matches itself.
        want = min(k + 1, len(self.ids))
        top = np.argpartition(-scores, want - 1)[:want]
        top = top[np.argsort(-scores[top])]
        chosen = [self.ids[i] for i in top if self.ids[i] != paper_id][:k]
        if not chosen:
            return [], time.monotonic() - started
        by_id = {self.ids[i]: float(scores[i]) for i in top}
        meta = self._meta(chosen)
        return ([meta[i] | {"score": by_id[i]} for i in chosen],
                time.monotonic() - started)


class Updater:
    """Runs `update` in the background, for the UI's "Fetch new papers" button,
    or just the embedding, for its "Embed them now" (papers imported with
    `build --scan-only` and not embedded yet).

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

    def __init__(self):
        self._lock = threading.Lock()
        self._thread = None
        self.state = "idle"         # idle | running | done | failed
        self.kind = "update"        # update | embed
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

    def start(self, kind: str = "update") -> bool:
        """Kick off a run. False if one is already going.

        One at a time whatever the kind: both embed, and two embedding runs
        would only have the second refused by the embed lock.
        """
        with self._lock:
            if self._in_flight():
                return False
            self.state, self.kind = "running", kind
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
            db = store.connect()
            if self.kind == "embed":
                store.check_model(db)
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
            if db is not None:
                db.close()

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
                "error": self.error,
            }
            if self.started:
                payload["elapsed"] = round(
                    (self.finished or time.time()) - self.started)
            if state == "running" and self.progress:
                done, total = self.progress
                payload["progress"] = {"done": done, "total": total}
        return payload


class Scheduler:
    """Presses "Fetch new papers" on a timer, for as long as `serve` is up.

    The button exists because an index goes stale behind a server that is left
    running; this is the same button, pressed by the clock instead. All of the
    "when" lives in `schedule`, as pure functions over the setting and two
    timestamps -- this class only supplies the clock, the thread and the
    refusal to start a second run on top of a first.

    Each tick re-reads the setting, so changing it in the UI takes effect
    within the tick rather than at the next restart. The read is one `meta`
    row, which is why polling is affordable enough to keep the alternative --
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
            try:
                # Generous, because listing a prolific author's whole output
                # is a legitimate request (Sturmfels has 217).
                k = max(1, min(500, int(one("k", "20"))))
            except ValueError:
                k = 20
            return cats, since, until, k

        def _route(self):
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            one = lambda key, default=None: params.get(key, [default])[0]

            if parsed.path == "/":
                # Never cache the page. It is generated from web.py, so it
                # changes whenever the server is edited and restarted -- a
                # browser holding yesterday's copy would silently hide new UI.
                # (The vendored assets under /static are immutable and are
                # cached aggressively instead.)
                self._send(page(index.categories).encode("utf-8"),
                           "text/html; charset=utf-8", no_store=True)
                return

            if parsed.path.startswith("/static/"):
                self._static(parsed.path[len("/static/"):])
                return

            if parsed.path == "/api/stats":
                self._json(index.stats())
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
                                         "under Profile."}, 400)
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
                                         "them under Profile."}, 400)
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
                try:
                    # Generous, because listing a prolific author's whole
                    # output is a legitimate request (Sturmfels has 217).
                    k = max(1, min(500, int(one("k", "20"))))
                except ValueError:
                    k = 20
                results, elapsed = index.similar(paper_id, k)
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
    size = (len(index.ids) * config.DIM
            * np.dtype(config.VEC_DTYPE).itemsize / 1e6)
    where = "VRAM" if index.gpu is not None else "RAM"
    print(f"{stats['embedded']:,} papers resident ({size:,.0f} MB in {where})"
          + (f", {stats['pending']:,} still embedding" if stats["pending"] else ""))

    updater = Updater()
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
  --line: #e3e3e0; --accent: #13396b; --accent-soft: #dde7f4; --shadow: rgba(0,0,0,.06);
  --warn: #a5251b;
  /* Two blues, because the accent has two jobs. --accent is drawn *on* the
     page (the cog, ghost labels, focus rings) and has to carry against the
     background; --accent-fill is the filled button, and has to carry white
     text. On a light page one blue does both; on a dark one they part ways. */
  --accent-fill: #13396b;
  --accent-hover: #0c2749;
  /* A lift on the dark accent, for a glyph that has to be found rather
     than read. */
  --accent-bright: #1f5ba8;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #16161a; --panel: #1e1e23; --ink: #ececf0; --muted: #9a9aa4;
    --line: #2e2e36; --accent: #5b95d8; --accent-soft: #172030; --shadow: rgba(0,0,0,.3);
    --warn: #f0847c;
    --accent-fill: #1d4f8a;
    --accent-hover: #163d6b;
    --accent-bright: #7fb4f0;
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
/* The settings panel makes the header taller than the viewport, and a sticky
   box that tall pins its top and puts its own bottom out of reach: the page
   scrolls the results past it and only reaches the panel once the list is
   spent. While the panel is open the header is an ordinary block, so it
   scrolls with the page and the panel is where it was left. */
body.settings-open header { position: static; }
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
  border: 0; border-radius: 7px; background: var(--accent-fill); color: #fff;
  cursor: pointer;
}
button:hover:not(:disabled) { background: var(--accent-hover); }
button:disabled { opacity: .5; cursor: default; }
/* Secondary action. It lives among the filters and must not compete with
   Search, which is what the page is actually for. */
button.ghost {
  padding: 4px 11px; font-size: 13px; background: none; color: var(--accent);
  border: 1px solid var(--line);
}
button.ghost:hover:not(:disabled) { background: var(--accent-soft); }
.grow { flex: 1 1 auto; }
/* One line, and only while there is something to say. Monospace because what
   it carries is the updater's own output. */
#update {
  padding: 0 0 12px; font-size: 13px; color: var(--muted);
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
#update.bad { color: var(--warn); }
/* The action row: things that run a listing, as against the filters above
   that describe one. */
.acts-bar { padding-bottom: 12px; gap: 10px; }
.acts-bar button {
  padding: 6px 13px; font-size: 14px;
}
.acts-bar button.primary {
  color: #fff; background: var(--accent-fill); border-color: var(--accent-fill);
}
.acts-bar button.primary:hover:not(:disabled) {
  background: var(--accent-hover); border-color: var(--accent-hover);
}
/* Authors are one short name a line; interests are sentences, and take the
   room. minmax(0,...) so a long line scrolls the field rather than widening
   its column. */
#settings {
  display: grid; gap: 14px 22px;
  grid-template-columns: minmax(0, 1fr) minmax(0, 2fr);
  padding: 4px 0 14px;
}
/* An author `display` beats the UA rule for [hidden], so the panel needs to be
   told explicitly to stay shut. */
#settings[hidden] { display: none; }
@media (max-width: 720px) { #settings { grid-template-columns: 1fr; } }
#settings label { display: flex; flex-direction: column; gap: 5px;
                 font-size: 13px; color: var(--muted); }
#settings label b { font-weight: 600; color: var(--ink); font-size: 14px; }
#settings textarea {
  font: inherit; font-size: 14px; line-height: 1.5; padding: 9px 11px;
  border: 1px solid var(--line); border-radius: 7px; background: var(--bg);
  color: var(--ink); resize: vertical; min-height: 336px;
}
#settings textarea:focus { outline: 2px solid var(--accent); outline-offset: -1px; }
.pbar { grid-column: 1 / -1; display: flex; gap: 12px; align-items: center;
        font-size: 13px; color: var(--muted); }
.pbar .bad { color: var(--warn); }
/* The interests side is a list of rows rather than one field, so it is a
   plain container: #settings label lays its children out in a column, which is
   right for a captioned textarea and wrong for a weight beside its text. */
.pfield { display: flex; flex-direction: column; gap: 5px;
          font-size: 13px; color: var(--muted); }
.pfield b { font-weight: 600; color: var(--ink); font-size: 14px; }
.ihead { display: flex; gap: 8px; font-size: 12px; padding-top: 3px; }
.ihead span:first-child { width: 72px; flex: none; }
#p-interests { display: flex; flex-direction: column; gap: 6px; }
.interest { display: flex; gap: 8px; align-items: stretch; }
/* A stepper, not a spinner: the UA's arrows are a few pixels tall and drawn
   in the UA's own style. One box instead, the field spanning both rows with
   the two arrows stacked in a column at its right edge, so the whole thing
   reads as a single control the width of its column. Its own height, since a
   number does not get easier to read by being stretched to the height of the
   description beside it. */
.interest .weight {
  flex: none; align-self: flex-start; width: 72px; height: 58px;
  display: grid; grid-template-columns: 1fr 24px; grid-template-rows: 1fr 1fr;
  overflow: hidden;
  border: 1px solid var(--line); border-radius: 7px; background: var(--bg);
}
.interest .weight:focus-within { outline: 2px solid var(--accent); outline-offset: -1px; }
.interest .weight input[type=number] {
  grid-row: 1 / -1; width: 100%; min-width: 0; font: inherit; font-size: 15px;
  padding: 0; border: 0; background: none; color: var(--ink); text-align: center;
  font-variant-numeric: tabular-nums;
  /* The field keeps the keyboard and the validation, and loses the arrows. */
  -moz-appearance: textfield; appearance: textfield;
}
.interest .weight input[type=number]::-webkit-outer-spin-button,
.interest .weight input[type=number]::-webkit-inner-spin-button {
  -webkit-appearance: none; margin: 0;
}
.interest .weight input[type=number]:focus { outline: none; }
.interest .weight button {
  padding: 0; border: 0; border-left: 1px solid var(--line); border-radius: 0;
  background: none; color: var(--accent-bright); cursor: pointer;
  font: inherit; font-size: 13px; line-height: 1;
  display: flex; align-items: center; justify-content: center;
}
.interest .weight button:last-child { border-top: 1px solid var(--line); }
.interest .weight button:hover:not(:disabled) {
  background: var(--accent-soft); color: var(--ink);
}
.interest .weight button:disabled { opacity: .3; cursor: default; }
/* Overrides the tall single-field default; a description is a line or two.
   Needs the id to outrank `#settings textarea`, which sets min-height. */
#settings .interest textarea {
  flex: 1 1 auto; min-height: 0; height: 84px; padding: 7px 10px;
}
.interest .drop {
  flex: none; align-self: stretch; width: 30px; padding: 0;
  font: inherit; font-size: 17px; line-height: 1;
  border: 1px solid var(--line); border-radius: 7px;
  background: none; color: var(--muted); cursor: pointer;
}
.interest .drop:hover { background: var(--accent-soft); color: var(--ink); }
#p-add { align-self: flex-start; margin-top: 2px; }
#settings .blendrow {
  flex-direction: row; align-items: center; gap: 9px; margin-top: 6px;
}
.blendrow input[type=range] { flex: 0 1 150px; accent-color: var(--accent); }
.blendrow output { color: var(--ink); font-variant-numeric: tabular-nums; }
/* The one control that opens the panel. Square, so the glyph sits centred
   rather than being letter-spaced like a word: the padding is even, and
   aspect-ratio takes the width from the height the glyph and padding make. */
button.cog {
  font-size: 21px; line-height: 1; padding: 8px; aspect-ratio: 1;
  display: inline-flex; align-items: center; justify-content: center;
  color: var(--accent-bright);
}
button.cog[aria-expanded="true"] {
  background: var(--accent-soft); color: var(--ink);
}
/* Updates span the panel's full width, under both profile columns. */
.sfield {
  grid-column: 1 / -1; display: flex; flex-direction: column; gap: 5px;
  padding-top: 12px; border-top: 1px solid var(--line);
  font-size: 13px; color: var(--muted);
}
.sfield b { font-weight: 600; color: var(--ink); font-size: 14px; }
.urow { display: flex; gap: 9px; align-items: center; flex-wrap: wrap;
        padding-top: 3px; }
.urow select, .urow input[type=number], .urow input[type=time] {
  font: inherit; font-size: 14px; padding: 5px 7px; border: 1px solid var(--line);
  border-radius: 6px; background: var(--bg); color: var(--ink);
}
.urow input[type=number] { width: 66px; text-align: center; }
#s-every[hidden], #s-at[hidden] { display: none; }
#s-next { color: var(--muted); }
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
/* One flex item holding both date fields: .opts wraps around the pair,
   never between them. */
.dates { display: flex; gap: 10px; align-items: center; }
#status { padding: 14px 0 0; font-size: 14px; color: var(--muted); min-height: 20px; }
#embed { margin-left: 8px; }
#results { padding: 6px 0 60px; }
article {
  background: var(--panel); border: 1px solid var(--line); border-radius: 9px;
  padding: 14px 16px; margin: 10px 0;
}
.top { display: flex; gap: 12px; align-items: baseline; }
/* Scores are diagnostics, not reading material: hidden unless asked for.
   Toggled by a class rather than re-rendering, so it costs no re-search. */
body:not(.with-scores) .top > .score {
  display: none;
}
.score {
  font-variant-numeric: tabular-nums; font-size: 13px; font-weight: 600;
  color: var(--accent); background: var(--accent-soft); padding: 2px 7px;
  border-radius: 5px; flex: none; text-align: center;
}
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
    <label>Author <input type="text" id="author" placeholder="Noether  ·  Hardy, Littlewood"
                        title="Several names, comma-separated, match papers they wrote together"
                        autocomplete="off"></label>
    <span>Categories:</span>
<!--CATEGORIES-->
    <label title="Show the relevance logit and cosine for each hit">
      <input type="checkbox" id="showscores"> Scores</label>
    <span class="dates">
      <label>Since <input type="date" id="since"></label>
      <label>Until <input type="date" id="until"></label>
    </span>
    <label>Results
      <select id="k">
        <option>10</option><option selected>20</option>
        <option>50</option><option>100</option><option>250</option>
      </select>
    </label>
  </div>
  <div class="opts acts-bar">
    <button type="button" id="followed" class="ghost primary"
            title="Every paper by a followed author in the date range above">Followed authors</button>
    <button type="button" id="byinterest" class="ghost primary"
            title="The date range above, ordered by closeness to your research interests">Rank by my interests</button>
    <span class="grow"></span>
    <button type="button" id="cog" class="ghost cog" aria-controls="settings"
            aria-expanded="false"
            title="Settings — who you follow, what you work on, and when the index tops itself up">⚙</button>
  </div>
  <div id="settings" hidden>
    <label><b>Followed authors</b> one name per line
      <textarea id="p-authors" rows="6" spellcheck="false"
                placeholder="Emmy Noether&#10;David Hilbert"></textarea></label>
    <div class="pfield">
      <b>Research interests</b>
      <span>one short description per thing you work on — each is embedded and
        matched separately, so distinct projects stay distinct instead of
        averaging into one blur. Weight 0 switches an entry off.</span>
      <div class="ihead"><span>Weight</span><span>Description</span></div>
      <div id="p-interests"></div>
      <button type="button" id="p-add" class="ghost">Add interest</button>
      <label class="blendrow" title="A paper is scored by its best-matching
interest, plus a diminishing share of every further match. At 0 only the best
match counts, so a paper squarely on one project wins. At 1 every interest
counts in full, which favours papers near the middle of all of them.">
        Reward for matching several
        <input type="range" id="p-blend" min="0" max="1" step="0.05">
        <output id="p-blendout"></output>
      </label>
    </div>
    <div class="pbar">
      <button type="button" id="p-save">Save</button>
      <button type="button" id="p-cancel" class="ghost">Cancel</button>
      <span id="p-note"></span>
    </div>
    <div class="sfield">
      <b>Automatic updates</b>
      <span>the same top-up as the button, on a clock, for as long as this
        server is running. Times are this machine's local time. A missed run
        is caught up rather than skipped.</span>
      <div class="urow">
        <select id="s-mode">
          <option value="off">Off</option>
          <option value="interval">Every</option>
          <option value="daily">Daily at</option>
        </select>
        <span id="s-every" hidden>
          <input type="number" id="s-hours" min="1" max="168" step="1"> hours
        </span>
        <input type="time" id="s-at" hidden>
        <span id="s-next"></span>
        <span class="grow"></span>
        <button type="button" id="fetch" class="ghost"
                title="Walk the arXiv API back to where the last run stopped, then embed whatever is new">Fetch new papers</button>
      </div>
    </div>
  </div>
  <div id="update" hidden></div>
</div></header>

<div class="wrap">
  <div id="status"><span id="status-text"></span>
    <button type="button" id="embed" class="ghost" hidden
            title="Embed the papers that are in the index but not yet searchable">Embed them now</button></div>
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
    note();
  });
}
refreshStats();

/* `extra` is remembered rather than passed through, so re-reading the counts
   after a fetch finishes does not wipe the line describing what is on screen. */
function note(extra) {
  if (extra !== undefined) lastNote = extra;
  let bits = [];
  if (stats) {
    bits.push(stats.embedded.toLocaleString() + " papers searchable");
    if (stats.missing.length)
      bits.push(stats.missing.join(", ") + " not in the index yet — "
                + "run build to add");
    if (stats.pending > 0)
      bits.push(stats.pending.toLocaleString() + (embedding
        ? " still embedding — results improve as it goes"
        : " not embedded yet, so not searchable"));
  }
  if (lastNote) bits.unshift(lastNote);
  $("#status-text").textContent = bits.join("  ·  ");
  // Offered only when nothing is running: a fetch embeds what is pending too.
  $("#embed").hidden = !(stats && stats.pending > 0) || running;
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
  running = s.state === "running";
  embedding = running && (s.kind === "embed" || !!s.progress);
  fetchBtn.disabled = running;
  fetchBtn.textContent = running && s.kind !== "embed"
    ? "Fetching…" : "Fetch new papers";
  note();
  if (running) watched = true;

  if (running) {
    // Embedding is the long half and reports a count; the walk before it only
    // has its own narration to offer, so show whichever exists.
    showUpdate(s.progress
      ? `Embedding ${s.progress.done.toLocaleString()} of `
        + `${s.progress.total.toLocaleString()}…`
      : (s.lines.length ? s.lines[s.lines.length - 1].trim() : "Starting…"));
  } else if (!watched) {
    updBox.hidden = true;
  } else if (s.state === "failed") {
    showUpdate((s.kind === "embed" ? "Embedding" : "Update")
               + " failed: " + (s.error || "unknown error"), true);
  } else if (s.state === "done") {
    showUpdate((s.kind === "embed"
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

pollUpdate();

function similar(id, title) {
  run(`/api/similar?id=${encodeURIComponent(id)}&k=${$("#k").value}`,
      "similar to " + deTeX(title).slice(0, 60));
}
</script>
</body>
</html>
"""

def page(categories) -> str:
    """The UI, with a checkbox per category."""
    boxes = "\n".join(
        f'    <label><input type="checkbox" class="cat" value="{c}"> '
        f"{c}</label>" for c in map(html.escape, categories))
    return PAGE.replace("<!--CATEGORIES-->", boxes)
