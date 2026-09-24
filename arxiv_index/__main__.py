"""Command line entry point: python -m arxiv_index <command>"""

import argparse
import json
import sys
import textwrap
from pathlib import Path

import numpy as np

from . import (config, ingest, search as search_mod, settings, store,
               textnorm, transfer, update as update_mod)


# --- Output -----------------------------------------------------------------


def _abs_url(paper_id: str) -> str:
    return f"https://arxiv.org/abs/{paper_id}"


def print_results(results, full: bool = False, scores: bool = False) -> None:
    """Print ranked results. Scores are diagnostics, so they are off by default."""
    if not results:
        print("No matches.")
        return
    for rank, paper in enumerate(results, 1):
        title = " ".join(paper["title"].split())
        score = ""
        if scores and paper.get("score") is not None:
            score = f"[cos {paper['score']:.3f}] "
        print(f"\n{rank:2d}. {score}{title}")
        print(f"    {paper['categories']}  ·  {paper['update_date']}  ·  "
              f"{_abs_url(paper['id'])}")
        if paper.get("authors"):
            # Names only: titles are left as stored, since un-escaping them
            # would strip the braces that their $…$ maths depends on.
            authors = textnorm.latex_to_unicode(paper["authors"])
            print(f"    {textwrap.shorten(authors, 100, placeholder=' et al.')}")
        if full:
            abstract = " ".join(paper["abstract"].split())
            print(textwrap.fill(abstract, 88, initial_indent="    ",
                                subsequent_indent="    "))
    print()


# --- Commands ---------------------------------------------------------------


def ask_categories() -> None:
    """On a first build, ask which categories to index and save the answer.

    Only when the settings name none yet and someone is at the terminal: run
    from a script, the build takes the defaults as it always has, and a
    settings file that names them is never second-guessed.
    """
    if "categories" in settings.load() or not sys.stdin.isatty():
        return
    default = " ".join(settings.DEFAULT_CATEGORIES)
    print("Which arXiv categories should the index cover? Give their full "
          "names,\nseparated by spaces or commas, e.g. math.AG hep-th cs.LG "
          "(the list is at\nhttps://arxiv.org/category_taxonomy). More "
          "categories make a longer build.\n")
    while True:
        try:
            answer = input(f"Categories [{default}]: ")
        except (EOFError, KeyboardInterrupt):
            print()
            raise SystemExit(1)
        try:
            chosen = settings.parse_categories(answer)
            break
        except ValueError as exc:
            print(f"{exc}. Try again.")
    settings.update(categories=chosen)
    print(f"Saved to {settings.path()}; edit \"categories\" there to change "
          "them later.\n")


def cmd_build(args) -> None:
    db = store.connect()
    store.check_model(db)
    if args.embed_only and args.snapshot:
        raise SystemExit("--embed-only does not read the snapshot; "
                         "give one or the other.")
    snapshot = args.snapshot.expanduser() if args.snapshot else None
    if snapshot and not snapshot.is_file():
        # Before asking for categories, not after.
        raise SystemExit(f"Snapshot not found at {snapshot}")
    if not args.embed_only:
        ask_categories()
        # Only the categories the index does not hold yet. The rest are kept
        # current by `update`, and the snapshot's copies would be older.
        missing = update_mod.missing(db)
        if missing:
            ingest.scan_snapshot(db, missing, path=snapshot, chunk=20_000)
        else:
            print(f"The index already holds "
                  f"{', '.join(settings.categories())}; nothing to scan.")
    if args.scan_only:
        pending = store.count_pending(db)
        if pending:
            print("\n" + textwrap.fill(
                f"{pending:,} papers are waiting to be embedded; "
                "`build --embed-only` does that. Until then they are listed "
                "by author but not found by searches, and the next `update` "
                "embeds them too.", 79))
    else:
        ingest.embed_pending(db)
    cmd_status(args, db)


def cmd_update(args) -> None:
    db = store.connect()
    update_mod.update(db, max_pages=args.max_pages)


def cmd_search(args) -> None:
    if not args.query and not (args.author or args.category or args.since):
        raise SystemExit(
            "Give a query, or at least one of --author / --category / --since."
        )
    db = store.connect()
    categories = args.category
    if not categories and set(update_mod.cursors(db)) - set(
            settings.categories()):
        # A shared index holding others' categories: keep to your own. On an
        # index holding only yours this would be a no-op filter that costs a
        # copy of the matrix, so it is skipped.
        categories = settings.categories()
    results = search_mod.search(
        db, args.query, k=args.k, categories=categories, since=args.since,
        author=args.author,
    )
    if args.json:
        json.dump(results, sys.stdout, indent=2)
        print()
    else:
        print_results(results, full=args.full, scores=args.scores)


def cmd_similar(args) -> None:
    db = store.connect()
    store.check_model(db)
    row = db.execute(
        "SELECT row, title FROM papers WHERE id = ?", (args.id,)
    ).fetchone()
    if row is None:
        raise SystemExit(f"{args.id} is not in the index.")
    if row["row"] is None:
        raise SystemExit(f"{args.id} has no vector yet; run `build` or `update`.")

    total = store.vector_count()
    mm = np.memmap(config.VEC_PATH, dtype=config.VEC_DTYPE, mode="r",
                   shape=(total, config.DIM))
    vector = np.asarray(mm[row["row"]], dtype=np.float32)

    matrix, ids = store.load_matrix(db)
    scores = search_mod.score_all(matrix, vector)

    # +1 because the paper matches itself.
    n = min(args.k + 1, len(ids))
    top = np.argpartition(-scores, n - 1)[:n]
    top = top[np.argsort(-scores[top])]

    chosen = [ids[i] for i in top if ids[i] != args.id][:args.k]
    meta = {
        r["id"]: dict(r)
        for r in db.execute(
            f"SELECT * FROM papers WHERE id IN ({','.join('?' * len(chosen))})",
            chosen,
        )
    }
    by_id = {ids[i]: float(scores[i]) for i in top}
    results = [meta[i] | {"score": by_id[i]} for i in chosen]

    print(f"\nSimilar to: {' '.join(row['title'].split())}")
    print_results(results, full=args.full, scores=args.scores)


def cmd_serve(args) -> None:
    from . import web

    web.serve(port=args.port, host=args.host, open_browser=not args.no_browser)


def cmd_status(args, db=None) -> None:
    db = db or store.connect()
    total = store.count_papers(db)
    pending = store.count_pending(db)
    slots = store.vector_count()
    size = config.VEC_PATH.stat().st_size / 1e6 if config.VEC_PATH.exists() else 0

    print(f"\nSettings:   {settings.path()}"
          + ("" if settings.path().exists() else "  (not created yet)"))
    print(f"Index:      {config.INDEX_DIR}")
    print(f"Model:      {store.get_meta(db, 'model')} ({config.DIM} dims, "
          f"{config.VEC_DTYPE})")
    print(f"Papers:     {total:,}   embedded {total - pending:,}, "
          f"pending {pending:,}")
    print(f"Vectors:    {slots:,} slots, {size:,.0f} MB"
          + (f"  ({slots - (total - pending):,} reclaimable)"
             if slots > total - pending else ""))

    # Every category either side knows of: yours, and whatever else the index
    # holds -- another reader's, or one since dropped from your settings.
    held = update_mod.cursors(db)
    mine = settings.categories()
    print("\nCategory          papers   complete to        (incl. cross-lists)")
    for cat in mine + sorted(set(held) - set(mine)):
        n = db.execute(
            "SELECT COUNT(*) FROM papers WHERE ' ' || categories || ' ' LIKE ?",
            (f"% {cat} %",),
        ).fetchone()[0]
        if cat not in held:
            note = "not in the index -- run `build` to add it"
        else:
            note = f"{held[cat]:%Y-%m-%d %H:%M} UTC"
            if cat not in mine:
                note += "   not in your settings"
        print(f"  {cat:<14}{n:>9,}   {note}")
    print()


def cmd_config(args) -> None:
    """Show the settings file, creating it with the defaults if absent."""
    if settings.write_default():
        print(f"Created {settings.path()} with the default categories.\n")
    print(f"# {settings.path()}")
    print(settings.path().read_text(encoding="utf-8"), end="")


def cmd_compact(args) -> None:
    """Rewrite the vector file with only live slots, in row order."""
    db = store.connect()
    rows = db.execute(
        "SELECT id, row FROM papers WHERE row IS NOT NULL ORDER BY row"
    ).fetchall()
    slots = store.vector_count()
    if len(rows) == slots:
        print(f"Nothing to reclaim ({slots:,} slots, all live).")
        return

    mm = np.memmap(config.VEC_PATH, dtype=config.VEC_DTYPE, mode="r",
                   shape=(slots, config.DIM))
    tmp = config.VEC_PATH.with_suffix(".compacting")
    with open(tmp, "wb") as fh:
        for start in range(0, len(rows), 8192):
            block = rows[start:start + 8192]
            fh.write(mm[[r["row"] for r in block]].tobytes())
    del mm

    # Renumber first, then swap the file in; if this is interrupted the old
    # file is still the one on disk and the DB transaction rolls back.
    db.executemany(
        "UPDATE papers SET row = ? WHERE id = ?",
        [(i, r["id"]) for i, r in enumerate(rows)],
    )
    tmp.replace(config.VEC_PATH)
    db.commit()
    print(f"Compacted {slots:,} -> {len(rows):,} slots "
          f"({(slots - len(rows)) * config.DIM * 2 / 1e6:.0f} MB reclaimed).")


# --- Argument parsing -------------------------------------------------------


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="arxiv_index",
        description="Semantic search over arXiv "
                    f"{', '.join(settings.categories())}. Settings are read "
                    f"from {settings.path()}, beside the index; "
                    "$ARXIV_INDEX_DIR moves both.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build", help="backfill from the Kaggle snapshot")
    p.add_argument("snapshot", nargs="?", type=Path,
                   help="arxiv-metadata-oai-snapshot.json; default: the "
                        "one in the current directory")
    only = p.add_mutually_exclusive_group()
    only.add_argument("--scan-only", action="store_true",
                      help="import the papers from the snapshot; embed them "
                           "later with --embed-only")
    only.add_argument("--embed-only", action="store_true",
                      help="skip the snapshot scan; just embed what is pending")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("update", help="fetch and embed new papers from arXiv")
    p.add_argument("--max-pages", type=int, default=update_mod.MAX_PAGES,
                   help="page cap for the API walk; raise it if a run reports "
                        "an incomplete walk")
    p.set_defaults(func=cmd_update)

    p = sub.add_parser("search", help="semantic search")
    p.add_argument("query", nargs="?", default="",
                   help="omit it to list by --author/--category/--since alone")
    p.add_argument("-k", type=int, default=10, help="number of results")
    p.add_argument("--category", action="append",
                   choices=settings.categories(),
                   help="restrict to a category (repeatable)")
    p.add_argument("--author", metavar="NAME[,NAME...]",
                   help="restrict to papers by these authors (all of them, so "
                        "'Hardy,Littlewood' finds their joint work); accents "
                        "and case are ignored")
    p.add_argument("--since", metavar="YYYY-MM-DD", help="only papers this recent")
    p.add_argument("--scores", action="store_true",
                   help="show relevance scores alongside each hit")
    p.add_argument("--full", action="store_true", help="print abstracts")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("similar", help="find papers like a given arXiv id")
    p.add_argument("id")
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--scores", action="store_true",
                   help="show relevance scores alongside each hit")
    p.add_argument("--full", action="store_true")
    p.set_defaults(func=cmd_similar)

    p = sub.add_parser("serve", help="open the web UI")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1",
                   help="default is localhost only; the server exposes the "
                        "index and, indirectly, Ollama")
    p.add_argument("--no-browser", action="store_true")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("status", help="show index statistics")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("config", help="show your settings file, creating it "
                                      "if it does not exist")
    p.set_defaults(func=cmd_config)

    p = sub.add_parser("compact", help="reclaim slots left by re-embedded papers")
    p.set_defaults(func=cmd_compact)

    p = sub.add_parser("export", help="write the index, embeddings included, "
                                      "to one file")
    p.add_argument("file", type=Path, help="e.g. arxiv_index.tar")
    p.add_argument("--settings", action="store_true",
                   help="include your settings: categories, profile, schedule")
    p.set_defaults(func=lambda args: transfer.export(args.file, args.settings))

    p = sub.add_parser("import", help="install an index written by export, "
                                      "or merge one into this one")
    p.add_argument("file", type=Path)
    how = p.add_mutually_exclusive_group()
    how.add_argument("--merge", action="store_true",
                     help="add its papers to the index already here, keeping "
                          "the more recent copy of any paper in both")
    how.add_argument("--replace", action="store_true",
                     help="overwrite the index already here")
    p.add_argument("--settings", action="store_true",
                   help="also take up the settings it holds, keeping yours "
                        "as config.json.bak")
    p.set_defaults(func=lambda args: transfer.import_(
        args.file, replace=args.replace, merge=args.merge,
        take_settings=args.settings))

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
