"""Export an index to one file, and import one.

The file is an uncompressed tar holding the index's two files, papers.db and
vectors.f16 -- the vectors are float16 noise to a compressor, so compressing
would cost minutes to save almost nothing. It holds the settings only if asked
to: config.json, with the categories and the profile, and the cache of the
interests' embeddings beside it, so they need not be embedded again. An import
takes them up only if asked to as well, keeping the settings it replaces.

The pair has to be consistent: every `row` in papers.db must name a slot that
vectors.f16 holds. Exporting takes the embedding lock, so nothing is appended
meanwhile, and copies the database before the vectors. Since vectors are
written before the database points at them (store.append_vectors), that order
would be safe even without the lock.

Importing either installs the export in place of the index (with nothing
there, or with --replace), or merges it in (--merge): see _merge. Into an index
with no papers yet -- a fresh install -- the export also brings its embedding
model and its categories, rather than having to match what the settings say.

Both work on streams as well as files, for the web UI: an export is written
straight into the HTTP response, with its exact size known beforehand, and an
import is unpacked as the upload arrives, never held whole.
"""

import contextlib
import io
import json
import os
import shutil
import sqlite3
import tarfile
import time

import numpy as np

from . import config, embedder, ingest, settings, store, update as update_mod

MEMBERS = ("papers.db", "vectors.f16")
# Only when the settings are exported too.
SETTINGS = "config.json"
CACHE = "interest-vectors.json"

BLOCK = tarfile.BLOCKSIZE          # 512
RECORD = tarfile.RECORDSIZE        # the archive is padded to a multiple of this


class Export:
    """An export ready to be written: its members and its exact size.

    The size is known before a byte is written, so the web UI can send it as
    Content-Length and the browser can show real progress. Each member is a
    GNU-format header block and its data padded to a block -- GNU because it
    stores sizes of 8 GB and over without the extra header PAX would add --
    then two empty blocks, the whole padded to a record, as tarfile writes it.
    """

    def __init__(self, members, papers: int):
        # [(name in the archive, a path or the bytes themselves, size)]
        self.members = members
        self.papers = papers
        data = sum(BLOCK + -(-size // BLOCK) * BLOCK for _, _, size in members)
        self.size = -(-(data + 2 * BLOCK) // RECORD) * RECORD

    def write(self, fileobj) -> None:
        """Write the archive to `fileobj`, sequentially: a file or a socket."""
        with tarfile.open(fileobj=fileobj, mode="w|",
                          format=tarfile.GNU_FORMAT) as tar:
            for name, path, size in self.members:
                info = tarfile.TarInfo(name)
                info.size, info.mtime, info.mode = size, int(time.time()), 0o644
                # Exactly `size` bytes, even of a file still growing: the
                # vectors past it are not referenced by the copied database.
                with (io.BytesIO(path) if isinstance(path, bytes)
                      else open(path, "rb")) as src:
                    tar.addfile(info, src)


@contextlib.contextmanager
def exporting(with_settings: bool = False):
    """Prepare an export and yield it, holding the embedding lock until it has
    been written, so nothing is appended or renumbered meanwhile.

    The settings, if wanted, are read now and held: the web UI may save the
    profile while the export downloads, and a file changing size under the
    archive's header would corrupt it."""
    if not config.DB_PATH.exists():
        raise SystemExit(f"There is no index at {config.INDEX_DIR} to export.")
    db = store.connect()
    copy = config.INDEX_DIR / "papers.db.exporting"
    try:
        with ingest.embed_lock():
            # The backup API gives a consistent copy of a live database,
            # WAL and all, which copying the file would not.
            out = sqlite3.connect(copy)
            with out:
                db.backup(out)
            out.close()
            members = [("papers.db", copy, copy.stat().st_size)]
            if config.VEC_PATH.exists():
                members.append(("vectors.f16", config.VEC_PATH,
                                config.VEC_PATH.stat().st_size))
            if with_settings:
                for name, path in ((SETTINGS, settings.path()),
                                   (CACHE, settings.cache_path())):
                    if path.is_file():
                        data = path.read_bytes()
                        members.append((name, data, len(data)))
            yield Export(members, store.count_papers(db))
    finally:
        copy.unlink(missing_ok=True)
        db.close()


def export(target, with_settings: bool = False) -> None:
    target = target.expanduser()
    if target.exists():
        raise SystemExit(f"{target} already exists; choose another name or "
                         "remove it first.")
    partial = target.with_name(target.name + ".partial")
    try:
        with exporting(with_settings) as ex, open(partial, "wb") as out:
            ex.write(out)
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)
    print(f"Exported {ex.papers:,} papers"
          + (" and your settings" if with_settings else "")
          + f" to {target} ({target.stat().st_size / 1e6:,.0f} MB).")


def import_(source, replace: bool = False, merge: bool = False,
            take_settings: bool = False) -> None:
    source = source.expanduser()
    if not source.is_file():
        raise SystemExit(f"{source} not found.")
    with open(source, "rb") as fileobj:
        import_stream(fileobj, str(source), replace=replace, merge=merge,
                      take_settings=take_settings)


def _is_empty() -> bool:
    """Whether there is no index here yet, or one with no papers."""
    if not config.DB_PATH.exists():
        return True
    db = sqlite3.connect(config.DB_PATH)
    try:
        return not db.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    except sqlite3.DatabaseError:
        return False
    finally:
        db.close()


def import_stream(fileobj, source: str, replace: bool = False,
                  merge: bool = False, take_settings: bool = False, log=print,
                  on_read=None) -> dict:
    """Import the export read from `fileobj`, sequentially, as it comes.

    Everything is unpacked beside the index first; `on_read` is called once
    the stream has been read to the end of the archive, and only then is the
    export checked and merged or installed. A stream that ends early raises
    before anything has been replaced. So do settings that cannot be used,
    when they were asked for.

    Returns {"settings": taken up, "model": the embedding model the settings
    now name, "model_changed": whether that differs from this process's}.
    """
    fresh = _is_empty()
    existing = config.DB_PATH.exists()
    if existing and not fresh and not (replace or merge):
        raise SystemExit(
            f"There is already an index at {config.INDEX_DIR}. Import with "
            "--merge to add the export's papers to it, --replace to overwrite "
            "it, or set $ARXIV_INDEX_DIR to import into another directory.")

    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    staged = {name: config.INDEX_DIR / (name + ".importing")
              for name in MEMBERS + (SETTINGS, CACHE)}
    try:
        seen = set()
        try:
            with tarfile.open(fileobj=fileobj, mode="r|") as tar:
                for member in tar:
                    if member.isfile() and member.name in staged:
                        with tar.extractfile(member) as src, \
                                open(staged[member.name], "wb") as dst:
                            shutil.copyfileobj(src, dst, 1 << 20)
                        seen.add(member.name)
        except tarfile.TarError as exc:
            raise SystemExit(f"{source} is not an index export: {exc}")
        if "papers.db" not in seen:
            raise SystemExit(f"{source} is not an index export: it has no "
                             "papers.db.")
        if on_read:
            on_read()
        meta = _check(staged, source, adopt=fresh)
        has_settings = SETTINGS in seen
        taking = take_settings and has_settings
        if taking:
            _check_settings(staged[SETTINGS], source, adopt=fresh)
        if fresh:
            embedding = _index_embedding(
                meta, staged[SETTINGS] if has_settings else None)
            held = _held_categories(staged["papers.db"])
        if existing and merge and not fresh:
            _merge(staged, log=log)
        else:
            _install(staged)
            log(f"Imported the index from {source} into {config.INDEX_DIR}.")
        if taking:
            _take_settings(staged, log)
        elif take_settings:
            log(f"{source} holds no settings; yours are unchanged.")
        elif has_settings:
            log(f"{source} also holds its settings; they were left out, "
                "and yours are unchanged.")
        model = config.MODEL
        if fresh:
            model = _adopt(embedding, None if taking else held, log)
        return {"settings": taking, "model": model,
                "model_changed": model != config.MODEL}
    finally:
        for path in staged.values():
            path.unlink(missing_ok=True)


def _install(staged) -> None:
    """Put the export's two files in place of the index's."""
    with ingest.embed_lock():
        # A replaced database's WAL would be replayed into the new one.
        for suffix in ("-wal", "-shm"):
            config.DB_PATH.with_name(config.DB_PATH.name + suffix).unlink(
                missing_ok=True)
        os.replace(staged["papers.db"], config.DB_PATH)
        if staged["vectors.f16"].exists():
            os.replace(staged["vectors.f16"], config.VEC_PATH)
        else:
            config.VEC_PATH.unlink(missing_ok=True)


def _check(staged, source, adopt: bool = False) -> dict:
    """Refuse an export whose vectors this reader cannot use, or whose two
    files do not match, before anything is replaced. Returns its meta table.

    With `adopt` the export's model is taken up rather than checked against
    the settings: the index here is empty, so there is nothing to match."""
    db = sqlite3.connect(staged["papers.db"])
    try:
        meta = dict(db.execute("SELECT key, value FROM meta").fetchall())
        highest = db.execute("SELECT MAX(row) FROM papers").fetchone()[0]
    except sqlite3.DatabaseError as exc:
        raise SystemExit(f"{source} does not hold a usable papers.db: {exc}")
    finally:
        db.close()

    wrong = None if adopt else store.embedding_mismatch(meta)
    if wrong:
        key, stored, current = wrong
        raise SystemExit(
            f"{source} was built with {key} {stored!r}, but your settings "
            f"give {current!r}. Set \"embedding\" in {settings.path()} to "
            "match it, then import again.")

    vectors = staged["vectors.f16"]
    size = vectors.stat().st_size if vectors.exists() else 0
    try:
        dim = int(meta.get("dim") or config.DIM)
    except ValueError:
        raise SystemExit(f"{source} is damaged: its dim is not a number.")
    width = dim * np.dtype(config.VEC_DTYPE).itemsize
    if size % width or (highest is not None and highest >= size // width):
        raise SystemExit(f"{source} is damaged: its vectors do not match its "
                         "papers.")
    return meta


def _index_embedding(meta: dict, settings_file) -> dict:
    """The "embedding" setting that describes an export's vectors.

    The model, dimension and document prefix are recorded in its index. The
    query prefix is not -- it only shapes searches -- so it comes from the
    export's own settings if they name the same model, and otherwise is the
    default's for the default model and none for any other.
    """
    default = config.DEFAULT_EMBEDDING
    model = meta.get("model") or default["model"]
    out = {"model": model, "dim": int(meta.get("dim") or default["dim"]),
           "document_prefix": meta.get("document_prefix", "")}
    theirs = {}
    if settings_file is not None:
        try:
            theirs = json.loads(settings_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    theirs = theirs.get("embedding") if isinstance(theirs, dict) else None
    if (isinstance(theirs, dict) and isinstance(theirs.get("query_prefix"), str)
            and theirs.get("model", default["model"]) == model):
        out["query_prefix"] = theirs["query_prefix"]
    else:
        out["query_prefix"] = (default["query_prefix"]
                               if model == default["model"] else "")
    if out == default:
        return {"model": model}     # the defaults need not be spelled out
    return out


def _held_categories(db_path) -> list:
    """The categories an export's index holds, in their usual order."""
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    try:
        return sorted(update_mod.cursors(db))
    finally:
        db.close()


def _adopt(embedding: dict, categories, log) -> str:
    """Name the imported index's model in the settings, and its categories
    unless its settings were taken up with their own. Returns the model."""
    values = {"embedding": embedding}
    if categories:
        values["categories"] = categories
    settings.update(**values)
    model = embedding["model"]
    log(f"This index now uses the embedding model {model}"
        + (f", and covers {', '.join(categories)}." if categories else "."))
    try:
        installed = {m["name"] for m in embedder.embedding_models()}
    except SystemExit:
        installed = None
    if installed is not None and model not in installed:
        log(f"{model} is not installed in Ollama; searching needs it: "
            f"ollama pull {model}")
    return model


def _check_settings(path, source, adopt: bool = False) -> None:
    """Refuse an export's settings that could not be used here, before
    anything is replaced: not a settings file, categories that are not, or a
    different embedding model from the one the index here is searched with
    (which the export's own index must match, so this is a file edited since).
    With `adopt` the model is the export's anyway; see _index_embedding."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"The settings in {source} are not readable: {exc}")
    if not isinstance(data, dict):
        raise SystemExit(f"The settings in {source} are not a settings file.")
    where = f"the settings in {source}"
    try:
        settings.clean_categories(
            data.get("categories", list(settings.DEFAULT_CATEGORIES)), where)
        theirs = settings.embedding(config.DEFAULT_EMBEDDING, data, where)
    except settings.SettingsError as exc:
        raise SystemExit(str(exc))
    if not adopt and theirs != config.EMBEDDING:
        raise SystemExit(
            f"The settings in {source} name the embedding model "
            f"{theirs['model']!r}, but this index is searched with "
            f"{config.MODEL!r}. Import without taking its settings.")


def _take_settings(staged, log) -> None:
    """Put the export's settings, and its cache of interest embeddings, in
    place of these, keeping the settings replaced as config.json.bak."""
    target = settings.path()
    kept = target.exists()
    if kept:
        shutil.copy2(target, target.with_name(target.name + ".bak"))
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staged[SETTINGS], target)
    if staged[CACHE].exists():
        os.replace(staged[CACHE], settings.cache_path())
    log("Took up the export's settings"
        + (f"; yours are kept in {target.name}.bak." if kept else "."))


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


def _merge(staged, batch: int = 5000, log=print) -> None:
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
    slots = (staged["vectors.f16"].stat().st_size // store.SLOT_BYTES
             if staged["vectors.f16"].exists() else 0)
    source = (np.memmap(staged["vectors.f16"], dtype=config.VEC_DTYPE,
                        mode="r", shape=(slots, config.DIM)) if slots else None)

    with ingest.embed_lock():
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

    log(f"Merged: {added:,} papers added, {updated:,} replaced by a more "
        f"recent copy, {kept:,} already as recent here.")
    for cat, stamp in sorted(advanced.items()):
        log(f"  {cat} now complete to {stamp:%Y-%m-%d %H:%M} UTC"
            + ("" if cat in mine else " (new to this index)"))
    reclaimable = store.vector_count() - (
        store.count_papers(db) - store.count_pending(db))
    if reclaimable > 0:
        log(f"{reclaimable:,} vector slots are no longer used; `compact` "
            "reclaims them.")

