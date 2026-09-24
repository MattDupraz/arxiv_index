"""Export an index to one file, and import one.

The file is an uncompressed tar holding the index's two files, papers.db and
vectors.f16 -- the vectors are float16 noise to a compressor, so compressing
would cost minutes to save almost nothing. It holds no settings: the profile
lives in the settings file, beside the index, and does not travel with it.

The pair has to be consistent: every `row` in papers.db must name a slot that
vectors.f16 holds. Exporting takes the embedding lock, so nothing is appended
meanwhile, and copies the database before the vectors. Since vectors are
written before the database points at them (store.append_vectors), that order
would be safe even without the lock.

Importing either installs the export in place of the index (with nothing
there, or with --replace), or merges it in (--merge): see merge().
"""

import os
import shutil
import sqlite3
import tarfile

import numpy as np

from . import config, ingest, settings, store, update as update_mod

MEMBERS = ("papers.db", "vectors.f16")


def export(target) -> None:
    target = target.expanduser()
    if target.exists():
        raise SystemExit(f"{target} already exists; choose another name or "
                         "remove it first.")
    if not config.DB_PATH.exists():
        raise SystemExit(f"There is no index at {config.INDEX_DIR} to export.")

    db = store.connect()
    copy = config.INDEX_DIR / "papers.db.exporting"
    partial = target.with_name(target.name + ".partial")
    try:
        with ingest._embed_lock():
            # The backup API gives a consistent copy of a live database,
            # WAL and all, which copying the file would not.
            out = sqlite3.connect(copy)
            with out:
                db.backup(out)
            out.close()
            with tarfile.open(partial, "w") as tar:
                tar.add(copy, arcname="papers.db")
                if config.VEC_PATH.exists():
                    tar.add(config.VEC_PATH, arcname="vectors.f16")
        os.replace(partial, target)
    finally:
        copy.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)
    print(f"Exported {store.count_papers(db):,} papers to {target} "
          f"({target.stat().st_size / 1e6:,.0f} MB).")


def import_(source, replace: bool = False, merge: bool = False) -> None:
    source = source.expanduser()
    if not source.is_file():
        raise SystemExit(f"{source} not found.")
    existing = config.DB_PATH.exists()
    if existing and not (replace or merge):
        raise SystemExit(
            f"There is already an index at {config.INDEX_DIR}. Import with "
            "--merge to add the export's papers to it, --replace to overwrite "
            "it, or set $ARXIV_INDEX_DIR to import into another directory.")

    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    staged = {name: config.INDEX_DIR / (name + ".importing") for name in MEMBERS}
    try:
        with tarfile.open(source, "r") as tar:
            found = {m.name: m for m in tar.getmembers() if m.isfile()}
            if "papers.db" not in found:
                raise SystemExit(f"{source} is not an index export: it has no "
                                 "papers.db.")
            for name, path in staged.items():
                if name in found:
                    with tar.extractfile(found[name]) as src, \
                            open(path, "wb") as dst:
                        shutil.copyfileobj(src, dst, 1 << 20)
        _check(staged, source)
        if existing and merge:
            _merge(staged)
            return
        with ingest._embed_lock():
            # A replaced database's WAL would be replayed into the new one.
            for suffix in ("-wal", "-shm"):
                config.DB_PATH.with_name(config.DB_PATH.name + suffix).unlink(
                    missing_ok=True)
            os.replace(staged["papers.db"], config.DB_PATH)
            if staged["vectors.f16"].exists():
                os.replace(staged["vectors.f16"], config.VEC_PATH)
            else:
                config.VEC_PATH.unlink(missing_ok=True)
    finally:
        for path in staged.values():
            path.unlink(missing_ok=True)
    print(f"Imported the index from {source} into {config.INDEX_DIR}.")


def _key(paper) -> tuple:
    """How recent a copy of a paper is: its arXiv version, then its date, so
    a metadata change (a journal reference added) counts within a version."""
    version = paper["version"] or ""
    number = int(version[1:]) if version[1:].isdigit() else 0
    return number, paper["update_date"] or ""


COLUMNS = ("id", "row", "version", "title", "abstract", "authors",
           "categories", "update_date", "doi", "journal_ref", "embedded_at")

MERGE = f"""
INSERT INTO papers ({", ".join(COLUMNS)})
VALUES ({", ".join(":" + c for c in COLUMNS)})
ON CONFLICT(id) DO UPDATE SET
    {", ".join(f"{c} = excluded.{c}" for c in COLUMNS if c != "id")}
"""


def _merge(staged, batch: int = 5000) -> None:
    """Add the export's papers to the index already here.

    A paper only in the export is added. A paper in both keeps the more
    recent copy (see _key); on a tie, the one that is embedded, and otherwise
    the index's own. A copy taken from the export brings its vector, appended
    under a new slot; the slot it replaces is left to `compact`. As always,
    vectors are written before the database points at them, and the papers are
    committed in one transaction, so an interruption leaves the index as it
    was, bar orphan slots.

    A category held by both is complete up to the later of the two cursors,
    since each copy is complete up to its own; one held by either alone keeps
    its cursor.
    """
    db = store.connect()
    store.check_model(db)
    theirs = sqlite3.connect(staged["papers.db"])
    theirs.row_factory = sqlite3.Row
    width = config.DIM
    slots = (staged["vectors.f16"].stat().st_size
             // (width * np.dtype(config.VEC_DTYPE).itemsize)
             if staged["vectors.f16"].exists() else 0)
    source = (np.memmap(staged["vectors.f16"], dtype=config.VEC_DTYPE,
                        mode="r", shape=(slots, width)) if slots else None)

    with ingest._embed_lock():
        ours = {r["id"]: (_key(r), r["row"] is not None) for r in db.execute(
            "SELECT id, version, update_date, row FROM papers")}
        added = updated = kept = 0
        next_slot = store.vector_count()
        pending = []

        def flush(fh):
            nonlocal next_slot
            rows = [p["row"] for p in pending if p["row"] is not None]
            if rows:
                fh.write(np.ascontiguousarray(source[rows]).tobytes())
                fh.flush()
            for paper in pending:
                if paper["row"] is not None:
                    paper["row"], next_slot = next_slot, next_slot + 1
            db.executemany(MERGE, pending)
            pending.clear()

        config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
        with open(config.VEC_PATH, "ab") as fh:
            for row in theirs.execute(f"SELECT {', '.join(COLUMNS)} FROM papers"):
                mine = ours.get(row["id"])
                if mine is None:
                    added += 1
                else:
                    key, embedded = _key(row), row["row"] is not None
                    if key < mine[0] or (key == mine[0] and
                                         (mine[1] or not embedded)):
                        kept += 1
                        continue
                    updated += 1
                pending.append(dict(row))
                if len(pending) >= batch:
                    flush(fh)
            if pending:
                flush(fh)
        db.commit()

        mine, others = update_mod.cursors(db), update_mod.cursors(theirs)
        advanced = {cat: stamp for cat, stamp in others.items()
                    if cat not in mine or stamp > mine[cat]}
        if advanced:
            update_mod.set_cursors(db, advanced)
            db.commit()
    theirs.close()

    print(f"Merged: {added:,} papers added, {updated:,} replaced by a more "
          f"recent copy, {kept:,} already as recent here.")
    for cat, stamp in sorted(advanced.items()):
        print(f"  {cat} now complete to {stamp:%Y-%m-%d %H:%M} UTC"
              + ("" if cat in mine else " (new to this index)"))
    reclaimable = store.vector_count() - (
        store.count_papers(db) - store.count_pending(db))
    if reclaimable > 0:
        print(f"{reclaimable:,} vector slots are no longer used; `compact` "
              "reclaims them.")


def _check(staged, source) -> None:
    """Refuse an export whose vectors this reader cannot use, or whose two
    files do not match, before anything is replaced."""
    db = sqlite3.connect(staged["papers.db"])
    try:
        meta = dict(db.execute("SELECT key, value FROM meta").fetchall())
        highest = db.execute("SELECT MAX(row) FROM papers").fetchone()[0]
    except sqlite3.DatabaseError as exc:
        raise SystemExit(f"{source} does not hold a usable papers.db: {exc}")
    finally:
        db.close()

    for key, current in (("model", config.MODEL), ("dim", str(config.DIM)),
                         ("document_prefix", config.DOCUMENT_PREFIX)):
        if meta.get(key) is not None and meta[key] != current:
            raise SystemExit(
                f"{source} was built with {key} {meta[key]!r}, but your "
                f"settings give {current!r}. Set \"embedding\" in "
                f"{settings.path()} to match it, then import again.")

    vectors = staged["vectors.f16"]
    size = vectors.stat().st_size if vectors.exists() else 0
    width = config.DIM * np.dtype(config.VEC_DTYPE).itemsize
    if size % width or (highest is not None and highest >= size // width):
        raise SystemExit(f"{source} is damaged: its vectors do not match its "
                         "papers.")
