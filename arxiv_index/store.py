"""Storage layer: SQLite for metadata, a flat append-only file for vectors.

Layout
------
index/papers.db    one row per in-scope paper. `row` is that paper's slot in
                   the vector file, or NULL if it has not been embedded yet.
index/vectors.f16  DIM float16 values per slot, packed back to back. Slot n
                   lives at byte offset n * DIM * 2.

The vector file is append-only. Re-embedding a revised paper appends a new slot
and repoints `papers.row` at it; the old slot becomes unreferenced and is
reclaimed by `compact`. Nothing is ever rewritten in place, so an interrupted
run can only ever lose the tail, which the next run redoes.
"""

import sqlite3
from datetime import datetime, timezone

import numpy as np

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    id           TEXT PRIMARY KEY,
    row          INTEGER UNIQUE,
    version      TEXT,
    title        TEXT NOT NULL,
    abstract     TEXT NOT NULL,
    authors      TEXT,
    categories   TEXT NOT NULL,
    update_date  TEXT,
    doi          TEXT,
    journal_ref  TEXT,
    embedded_at  TEXT
);
CREATE INDEX IF NOT EXISTS papers_pending ON papers(row) WHERE row IS NULL;
CREATE INDEX IF NOT EXISTS papers_update_date ON papers(update_date);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


def connect(check_same_thread: bool = True) -> sqlite3.Connection:
    """Open the index, creating it if absent.

    Pass check_same_thread=False to share one connection across threads (the web
    server does this); the caller is then responsible for serialising access.
    """
    config.INDEX_DIR.mkdir(exist_ok=True)
    db = sqlite3.connect(config.DB_PATH, check_same_thread=check_same_thread)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    # A search issued while a long embedding run is committing should wait its
    # turn rather than fail outright.
    db.execute("PRAGMA busy_timeout=10000")
    db.executescript(SCHEMA)
    set_meta_defaults(db)
    return db


def set_meta_defaults(db: sqlite3.Connection) -> None:
    for key, value in (("model", config.MODEL), ("dim", str(config.DIM))):
        db.execute("INSERT OR IGNORE INTO meta VALUES (?, ?)", (key, value))
    db.commit()


def get_meta(db: sqlite3.Connection, key: str, default=None):
    row = db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


def set_meta(db: sqlite3.Connection, key: str, value: str) -> None:
    db.execute(
        "INSERT INTO meta VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = ?",
        (key, value, value),
    )
    db.commit()


def check_model(db: sqlite3.Connection) -> None:
    """Refuse to mix vectors from different models in one file."""
    stored = get_meta(db, "model")
    if stored and stored != config.MODEL:
        raise SystemExit(
            f"Index was built with {stored!r} but config.MODEL is {config.MODEL!r}.\n"
            "Vectors from different models are not comparable. Either restore the "
            "old model name or rebuild the index from scratch."
        )


# --- Metadata ---------------------------------------------------------------

UPSERT = """
INSERT INTO papers (id, version, title, abstract, authors, categories,
                    update_date, doi, journal_ref, row)
VALUES (:id, :version, :title, :abstract, :authors, :categories,
        :update_date, :doi, :journal_ref, NULL)
ON CONFLICT(id) DO UPDATE SET
    title       = excluded.title,
    abstract    = excluded.abstract,
    authors     = excluded.authors,
    categories  = excluded.categories,
    update_date = excluded.update_date,
    doi         = excluded.doi,
    journal_ref = excluded.journal_ref,
    version     = excluded.version,
    -- Only re-embed when the text we embed could have changed.
    row = CASE
        WHEN papers.version IS NOT excluded.version
          OR papers.title   IS NOT excluded.title
          OR papers.abstract IS NOT excluded.abstract
        THEN NULL ELSE papers.row END
"""


def upsert_papers(db: sqlite3.Connection, records) -> int:
    """Insert or refresh metadata. Returns how many rows now await embedding."""
    db.executemany(UPSERT, records)
    db.commit()
    return count_pending(db)


def count_pending(db: sqlite3.Connection) -> int:
    return db.execute("SELECT COUNT(*) FROM papers WHERE row IS NULL").fetchone()[0]


def count_papers(db: sqlite3.Connection) -> int:
    return db.execute("SELECT COUNT(*) FROM papers").fetchone()[0]


def pending_batches(db: sqlite3.Connection, size: int):
    """Yield lists of unembedded papers, re-querying each time so that rows
    committed by the previous batch drop out and a resumed run picks up where
    it left off."""
    while True:
        rows = db.execute(
            "SELECT id, title, abstract FROM papers WHERE row IS NULL LIMIT ?",
            (size,),
        ).fetchall()
        if not rows:
            return
        yield rows


# --- Vectors ----------------------------------------------------------------


def vector_count() -> int:
    """Number of slots currently in the vector file."""
    if not config.VEC_PATH.exists():
        return 0
    itemsize = np.dtype(config.VEC_DTYPE).itemsize
    return config.VEC_PATH.stat().st_size // (config.DIM * itemsize)


def normalise(vectors: np.ndarray) -> np.ndarray:
    """L2-normalise in float32, then cast to the storage dtype."""
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(config.VEC_DTYPE)


def append_vectors(db: sqlite3.Connection, ids, vectors: np.ndarray) -> None:
    """Append vectors and point their papers at the new slots.

    The file is flushed before the DB commits, so a crash between the two leaves
    orphan slots (harmless, reclaimed by `compact`) rather than papers pointing
    at vectors that were never written.
    """
    vectors = normalise(vectors)
    if vectors.shape != (len(ids), config.DIM):
        raise ValueError(f"expected {(len(ids), config.DIM)}, got {vectors.shape}")

    start = vector_count()
    config.INDEX_DIR.mkdir(exist_ok=True)
    with open(config.VEC_PATH, "ab") as fh:
        fh.write(vectors.tobytes())
        fh.flush()

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    db.executemany(
        "UPDATE papers SET row = ?, embedded_at = ? WHERE id = ?",
        [(start + i, now, pid) for i, pid in enumerate(ids)],
    )
    db.commit()


def load_matrix(db: sqlite3.Connection, where: str = None, params=(),
                keep_ids=None):
    """Return (matrix, ids) for every embedded paper, optionally filtered.

    Slots are read in file order and the unreferenced ones dropped, so the
    result is contiguous and `ids[i]` names row `i` of the matrix. `where` is
    extra SQL ANDed onto the selection, so a filtered search scores only the
    rows that survive the filter rather than scoring everything and discarding.

    `keep_ids` narrows the selection further in Python. Author matching needs
    it: names are stored as LaTeX, so the comparison happens after normalising
    both sides and cannot be expressed as SQL.
    """
    sql = "SELECT id, row FROM papers WHERE row IS NOT NULL"
    if where:
        sql += f" AND ({where})"
    rows = db.execute(sql + " ORDER BY row", params).fetchall()
    if keep_ids is not None:
        rows = [r for r in rows if r["id"] in keep_ids]
    if not rows:
        return np.empty((0, config.DIM), dtype=config.VEC_DTYPE), []

    slots = np.fromiter((r["row"] for r in rows), dtype=np.int64, count=len(rows))
    ids = [r["id"] for r in rows]

    total = vector_count()
    if slots[-1] >= total:
        raise SystemExit(
            f"Index is inconsistent: paper points at slot {slots[-1]} but the "
            f"vector file only holds {total}. Run `compact` to rebuild it."
        )

    mm = np.memmap(config.VEC_PATH, dtype=config.VEC_DTYPE, mode="r",
                   shape=(total, config.DIM))
    # Contiguous run is the common case (no revisions yet) and avoids a copy.
    if len(slots) == total and slots[0] == 0 and slots[-1] == total - 1:
        return mm, ids
    return mm[slots], ids
