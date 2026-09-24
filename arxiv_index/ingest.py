"""Load in-scope metadata from the Kaggle snapshot, and embed what's pending.

Metadata loading and embedding are deliberately separate phases. Scanning the
snapshot takes a couple of minutes; embedding takes hours. Splitting them means
the slow phase is a simple resumable loop over `row IS NULL`, and it is shared
verbatim with the incremental arXiv update path.
"""

import contextlib
import fcntl
import json
import sys
import time

from . import config, embedder, store


@contextlib.contextmanager
def _embed_lock():
    """Serialise embedding runs across processes.

    Two concurrent runs would both see the same `row IS NULL` rows and embed
    them twice, appending duplicate slots and wasting GPU time. A cron `update`
    firing during a long `build` is the obvious way to hit this.
    """
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    path = config.INDEX_DIR / "embed.lock"
    with open(path, "w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(
                "Another embedding run is already in progress "
                f"(lock held on {path}).\nWait for it to finish, or check it is "
                "still alive with: pgrep -af arxiv_index"
            ) from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _record(paper: dict) -> dict:
    versions = paper.get("versions") or []
    return {
        "id": paper["id"],
        "version": versions[-1]["version"] if versions else None,
        "title": paper["title"],
        "abstract": paper["abstract"],
        "authors": paper.get("authors"),
        "categories": paper["categories"],
        "update_date": paper.get("update_date"),
        "doi": paper.get("doi"),
        "journal_ref": paper.get("journal-ref"),
    }


def scan_snapshot(db, categories, path=None, chunk: int = 20_000) -> int:
    """Backfill `categories` from the snapshot. Returns the count in scope.

    Papers already held are left alone (see store.insert_new_papers), so this
    can add a category to an index that has others, and an interrupted scan
    can simply be run again. Each category gets its update cursor only once
    the scan has finished -- that is what marks it as held -- set to the
    newest date seen, so the next `update` fills in everything since the
    snapshot was taken.
    """
    path = path or config.SNAPSHOT
    if not path.exists():
        raise SystemExit(
            f"Snapshot not found at {path}\nDownload it from "
            "https://www.kaggle.com/datasets/Cornell-University/arxiv and "
            "give its path to `build`.")

    print(f"Scanning {path.name} for {', '.join(categories)} ...")
    matched = 0
    seen = 0
    newest = ""
    found = dict.fromkeys(categories, 0)
    buffer = []
    started = time.monotonic()

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            seen += 1
            if seen % 250_000 == 0:
                print(f"  {seen:,} lines, {matched:,} in scope", flush=True)
            # Cheap substring reject before paying for json.loads on 3.1M lines.
            if not any(c in line for c in categories):
                continue
            paper = json.loads(line)
            if not config.in_scope(paper["categories"], categories):
                continue
            matched += 1
            newest = max(newest, paper.get("update_date") or "")
            for c in paper["categories"].split():
                if c in found:
                    found[c] += 1
            buffer.append(_record(paper))
            if len(buffer) >= chunk:
                store.insert_new_papers(db, buffer)
                buffer.clear()

    if buffer:
        store.insert_new_papers(db, buffer)

    elapsed = time.monotonic() - started
    print(f"Scanned {seen:,} records in {elapsed:.0f}s; {matched:,} in scope.")
    empty = [c for c, n in found.items() if not n]
    if empty:
        # Most likely a misspelling. Left without a cursor, so it keeps being
        # reported as not held rather than silently fetching nothing.
        print(f"No papers at all in {', '.join(empty)} -- check the name "
              f"against https://arxiv.org/category_taxonomy.")
    if newest:
        from . import update

        update.set_cursors(db, {c: update.day_cursor(newest)
                                for c, n in found.items() if n})
    return matched


def embed_pending(db, batch_size: int = None, log=print, progress=None) -> int:
    """Embed every paper with no vector yet. Safe to interrupt and re-run.

    `log` receives the framing lines and `progress` the running (done, total)
    count, so a caller that is not a terminal -- the web UI's Fetch button --
    can show the same thing without parsing stdout. Left alone, both keep the
    CLI's behaviour: prose on stdout, a rewriting progress line on stderr.
    """
    batch_size = batch_size or config.BATCH_SIZE
    total = store.count_pending(db)
    if not total:
        log("Nothing to embed; index is up to date.")
        return 0

    embedder.check_available()
    log(f"Embedding {total:,} papers with {config.MODEL} ...")
    done = 0
    started = time.monotonic()

    with _embed_lock():
        # Re-read under the lock: a run that just finished may have drained it.
        total = store.count_pending(db) or total
        for rows in store.pending_batches(db, batch_size):
            vectors = embedder.embed_documents(
                [(r["title"], r["abstract"]) for r in rows]
            )
            store.append_vectors(db, [r["id"] for r in rows], vectors)

            done += len(rows)
            if progress is not None:
                progress(done, total)
                continue
            elapsed = time.monotonic() - started
            rate = done / elapsed
            remaining = (total - done) / rate if rate else 0
            print(
                f"\r  {done:,}/{total:,} ({done / total:6.1%})  "
                f"{rate:5.1f} docs/s  eta {remaining / 60:5.1f} min   ",
                end="",
                file=sys.stderr,
                flush=True,
            )

    if progress is None and done:
        # Close the rewriting stderr line so the summary does not land on it.
        print(file=sys.stderr)
    log(f"Embedded {done:,} papers in {(time.monotonic() - started) / 60:.1f} min.")
    return done
