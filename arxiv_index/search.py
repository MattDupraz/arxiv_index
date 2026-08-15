"""Exact nearest-neighbour search over the index.

At this corpus size a brute-force scan is the right call: 145k x 2560 float16 is
~745 MB, and scoring it is a single matrix-vector product that runs in well under
a second once the file is in page cache. The payoff is that results are exact and
there is no ANN structure to rebuild whenever papers are appended.
"""

import functools

import numpy as np

from . import config, embedder, store, textnorm

# Rows scored per pass. Bounds the float32 working copy to a few hundred MB
# regardless of how large the corpus grows.
CHUNK = 32_768


def score_all(matrix, query: np.ndarray) -> np.ndarray:
    """Cosine similarity of `query` against every row. Both sides are already
    L2-normalised, so the cosine is just a dot product."""
    scores = np.empty(len(matrix), dtype=np.float32)
    for start in range(0, len(matrix), CHUNK):
        block = np.asarray(matrix[start:start + CHUNK], dtype=np.float32)
        scores[start:start + len(block)] = block @ query
    return scores


def author_ids(db, author: str):
    """Ids whose author list contains `author`, compared on folded names.

    Both sides go through textnorm.fold, so "Kollar", "kollar" and "Kollár" all
    match the stored "Koll\\'ar". Scanning 145k rows takes well under a second,
    which is cheaper than maintaining a normalised column.
    """
    terms = textnorm.fold_terms(author)
    if not terms:
        return None
    return {
        row["id"]
        for row in db.execute("SELECT id, authors FROM papers")
        if textnorm.matches_terms(textnorm.fold(row["authors"] or ""), terms)
    }


def browse(db, k: int = 10, categories=None, since: str = None,
           author: str = None):
    """Newest-first listing by metadata alone, across the whole corpus.

    Deliberately does not touch the vectors: with no query there is nothing to
    score, and requiring an embedding would hide every paper the embedder has
    not reached yet -- half the corpus during a build.
    """
    clauses, params = [], []
    if categories:
        clauses.append(
            " OR ".join(["' ' || categories || ' ' LIKE ?"] * len(categories))
        )
        params += [f"% {c} %" for c in categories]
    if since:
        clauses.append("update_date >= ?")
        params.append(since)

    sql = "SELECT * FROM papers"
    if clauses:
        sql += " WHERE " + " AND ".join(f"({c})" for c in clauses)
    rows = db.execute(sql + " ORDER BY update_date DESC", params)

    terms = textnorm.fold_terms(author)
    out = []
    for row in rows:
        if terms and not textnorm.matches_terms(
                textnorm.fold(row["authors"] or ""), terms):
            continue
        out.append(dict(row) | {"score": None})
        if len(out) >= k:
            break
    return out


def search(db, query: str, k: int = 10, categories=None, since: str = None,
           author: str = None, rerank: bool = False):
    """Return the k best matches as a list of sqlite3.Row, each with `score`.

    With `rerank`, the index proposes RERANK_CANDIDATES hits and a cross-encoder
    reorders them, which is where the quality comes from -- the index only has
    to get the right papers into the shortlist, not order them well.
    """
    if not query:
        return browse(db, k, categories, since, author)
    store.check_model(db)
    want = max(k, config.RERANK_CANDIDATES) if rerank else k

    clauses, params = [], []
    if categories:
        # `categories` is a space-separated string; pad both sides so that a
        # search for math.CO cannot match a hypothetical math.COX.
        clauses.append(
            " OR ".join(["' ' || categories || ' ' LIKE ?"] * len(categories))
        )
        params += [f"% {c} %" for c in categories]
    if since:
        clauses.append("update_date >= ?")
        params.append(since)

    where = " AND ".join(f"({c})" for c in clauses) if clauses else None
    keep = author_ids(db, author) if author else None
    matrix, ids = store.load_matrix(db, where, params, keep_ids=keep)
    if len(ids) == 0:
        return []

    scores = score_all(matrix, embed_query_normalised(query))
    # Shortlist size, which is larger than k when reranking. `k` must survive
    # unchanged: it is what the caller actually asked for.
    shortlist = min(want, len(ids))
    # argpartition finds the top n without sorting the whole score array.
    top = np.argpartition(-scores, shortlist - 1)[:shortlist]
    top = top[np.argsort(-scores[top])]

    chosen = [ids[i] for i in top]
    placeholders = ",".join("?" * len(chosen))
    meta = {
        r["id"]: r
        for r in db.execute(
            f"SELECT * FROM papers WHERE id IN ({placeholders})", chosen
        )
    }
    hits = [(dict(meta[ids[i]]) | {"score": float(scores[i])}) for i in top]
    if rerank:
        import sys

        from . import rerank as rerank_mod

        try:
            hits = rerank_mod.rerank(query, hits)
        except rerank_mod.RerankUnavailable as exc:
            # Degrade to vector order rather than losing the search, and say
            # why. Usual causes: torch not installed, or no free VRAM.
            print(f"warning: reranking unavailable, showing vector order "
                  f"({exc})", file=sys.stderr)
    return hits[:k]


@functools.lru_cache(maxsize=512)
def _query_vector(query: str, model: str) -> np.ndarray:
    vector = embedder.embed_query(query)
    norm = np.linalg.norm(vector)
    unit = (vector / (norm or 1.0)).astype(np.float32)
    # Shared between callers, so freeze it rather than trust everyone.
    unit.flags.writeable = False
    return unit


def embed_query_normalised(query: str) -> np.ndarray:
    """Unit-length embedding of a query, cached per (query, model).

    Normalising keeps scores in [-1, 1] so they read as genuine cosines.

    The cache is a latency optimisation: ~104ms per repeat, more while a build
    competes for the GPU. It earns its keep because changing any filter -- the
    reranker toggle, the result count, a category -- resubmits the same query
    text, and those re-searches then skip the embedding call.

    It also makes repeated searches return identical cosines, because Ollama's
    embeddings are not deterministic (the reduction order depends on how a
    request is batched, giving ~4e-3 per-component variation under concurrent
    load). That is a nicety rather than a fix: papers within 3e-3 of cosine are
    ties, and either order is as good.

    The model is part of the key so that changing MODEL cannot serve vectors
    from the old one.
    """
    return _query_vector(query, config.MODEL)
