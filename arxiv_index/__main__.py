"""Command line entry point: python -m arxiv_index <command>"""

import argparse
import json
import sys
import textwrap

import numpy as np

from . import (config, ingest, search as search_mod, store, textnorm,
               update as update_mod)


# --- Output -----------------------------------------------------------------


def _abs_url(paper_id: str) -> str:
    return f"https://arxiv.org/abs/{paper_id}"


def print_results(results, full: bool = False, scores: bool = False) -> None:
    """Print ranked results. Scores are diagnostics, so they are off by default.

    With `scores`, a reranked hit shows both numbers -- the relevance logit the
    order is based on, and the cosine it started from -- since they are on
    unrelated scales and one without the other is misleading.
    """
    if not results:
        print("No matches.")
        return
    for rank, paper in enumerate(results, 1):
        title = " ".join(paper["title"].split())
        score = ""
        if scores and paper.get("score") is not None:
            if paper.get("rerank_margin") is not None:
                score = (f"[rr {paper['rerank_margin']:5.2f} · "
                         f"cos {paper['vector_score']:.3f}] ")
            else:
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


def cmd_build(args) -> None:
    db = store.connect()
    store.check_model(db)
    if not args.embed_only:
        ingest.scan_snapshot(db, chunk=20_000)
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
    results = search_mod.search(
        db, args.query, k=args.k, categories=args.category, since=args.since,
        author=args.author, rerank=args.rerank,
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

    # A wider net when reranking; the cross-encoder can only reorder what it is
    # given. +1 throughout because the paper matches itself.
    shortlist = max(args.k, config.RERANK_CANDIDATES) if args.rerank else args.k
    n = min(shortlist + 1, len(ids))
    top = np.argpartition(-scores, n - 1)[:n]
    top = top[np.argsort(-scores[top])]

    chosen = [ids[i] for i in top if ids[i] != args.id][:shortlist]
    meta = {
        r["id"]: dict(r)
        for r in db.execute(
            f"SELECT * FROM papers WHERE id IN ({','.join('?' * len(chosen))})",
            chosen,
        )
    }
    by_id = {ids[i]: float(scores[i]) for i in top}
    results = [meta[i] | {"score": by_id[i]} for i in chosen]

    if args.rerank:
        from . import rerank as rerank_mod

        source = db.execute("SELECT * FROM papers WHERE id = ?",
                            (args.id,)).fetchone()
        try:
            # The source paper's own text stands in for the query: the model
            # scores a text pair either way.
            results = rerank_mod.rerank(
                rerank_mod.document_text(dict(source)), results)
        except rerank_mod.RerankUnavailable as exc:
            print(f"warning: reranking unavailable, showing vector order "
                  f"({exc})", file=sys.stderr)

    print(f"\nSimilar to: {' '.join(row['title'].split())}")
    print_results(results[:args.k], full=args.full, scores=args.scores)


def cmd_serve(args) -> None:
    from . import web

    web.serve(port=args.port, host=args.host, open_browser=not args.no_browser)


def cmd_status(args, db=None) -> None:
    db = db or store.connect()
    total = store.count_papers(db)
    pending = store.count_pending(db)
    slots = store.vector_count()
    size = config.VEC_PATH.stat().st_size / 1e6 if config.VEC_PATH.exists() else 0

    print(f"\nIndex:      {config.INDEX_DIR}")
    print(f"Model:      {store.get_meta(db, 'model')} ({config.DIM} dims, "
          f"{config.VEC_DTYPE})")
    print(f"Scope:      {', '.join(config.CATEGORIES)} (incl. cross-lists)")
    print(f"Papers:     {total:,}   embedded {total - pending:,}, "
          f"pending {pending:,}")
    print(f"Vectors:    {slots:,} slots, {size:,.0f} MB"
          + (f"  ({slots - (total - pending):,} reclaimable)"
             if slots > total - pending else ""))
    print(f"Cursor:     {update_mod.default_cursor(db):%Y-%m-%d %H:%M} UTC")

    per_cat = []
    for cat in config.CATEGORIES:
        n = db.execute(
            "SELECT COUNT(*) FROM papers WHERE ' ' || categories || ' ' LIKE ?",
            (f"% {cat} %",),
        ).fetchone()[0]
        per_cat.append(f"{cat} {n:,}")
    print(f"By tag:     {'   '.join(per_cat)}\n")


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
                    f"{', '.join(config.CATEGORIES)}.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build", help="backfill from the Kaggle snapshot")
    p.add_argument("--embed-only", action="store_true",
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
    p.add_argument("--category", action="append", choices=config.CATEGORIES,
                   help="restrict to a category (repeatable)")
    p.add_argument("--author", metavar="NAME[,NAME...]",
                   help="restrict to papers by these authors (all of them, so "
                        "'Hardy,Littlewood' finds their joint work); accents "
                        "and case are ignored")
    p.add_argument("--since", metavar="YYYY-MM-DD", help="only papers this recent")
    p.add_argument("--rerank", action="store_true",
                   help=f"rescore the top {config.RERANK_CANDIDATES} hits with "
                        "a cross-encoder; better ordering, several seconds slower")
    p.add_argument("--scores", action="store_true",
                   help="show relevance scores alongside each hit")
    p.add_argument("--full", action="store_true", help="print abstracts")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("similar", help="find papers like a given arXiv id")
    p.add_argument("id")
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--rerank", action="store_true",
                   help="rescore with the cross-encoder, using this paper's "
                        "own text in place of a query")
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

    p = sub.add_parser("compact", help="reclaim slots left by re-embedded papers")
    p.set_defaults(func=cmd_compact)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
