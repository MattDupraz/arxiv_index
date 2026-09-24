"""Exact nearest-neighbour search over the index.

At this corpus size a brute-force scan is the right call: 145k x 2560 float16 is
~745 MB, and scoring it is a single matrix-vector product that runs in well under
a second once the file is in page cache. The payoff is that results are exact and
there is no ANN structure to rebuild whenever papers are appended.
"""

import functools

import numpy as np

from . import embedder, store, textnorm

# Rows scored per pass. Bounds the float32 working copy to a few hundred MB
# regardless of how large the corpus grows.
CHUNK = 32_768


def score_all(matrix, query: np.ndarray) -> np.ndarray:
    """Cosine similarity of `query` against every row. Both sides are already
    L2-normalised, so the cosine is just a dot product.

    `query` is either one vector, shape (DIM,), giving one score per row, or a
    stack of them, shape (DIM, n), giving (rows, n). The stacked form exists so
    that ranking against several interests still reads the 745 MB matrix once.
    """
    scores = np.empty((len(matrix),) + query.shape[1:], dtype=np.float32)
    for start in range(0, len(matrix), CHUNK):
        block = np.asarray(matrix[start:start + CHUNK], dtype=np.float32)
        scores[start:start + len(block)] = block @ query
    return scores


def top(scores: np.ndarray, n: int) -> np.ndarray:
    """Indices of the n highest scores, best first. argpartition finds them
    without sorting the whole array."""
    n = min(n, len(scores))
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    best = np.argpartition(-scores, n - 1)[:n]
    return best[np.argsort(-scores[best])]


def papers(db, ids) -> dict:
    """{id: paper as a dict} for `ids`."""
    if not ids:
        return {}
    marks = ",".join("?" * len(ids))
    return {r["id"]: dict(r) for r in db.execute(
        f"SELECT * FROM papers WHERE id IN ({marks})", list(ids))}


def _filters(categories, since):
    """SQL for the category and date filters, and its parameters.

    `categories` is a space-separated column; padding both sides keeps a search
    for math.CO from matching a hypothetical math.COX."""
    clauses, params = [], []
    if categories:
        clauses.append(
            " OR ".join(["' ' || categories || ' ' LIKE ?"] * len(categories)))
        params += [f"% {c} %" for c in categories]
    if since:
        clauses.append("update_date >= ?")
        params.append(since)
    return " AND ".join(f"({c})" for c in clauses) or None, params


def author_ids(db, author: str):
    """Ids whose author list contains `author`, compared on folded names.

    Both sides go through textnorm.fold, so "Poincare", "poincare" and
    "Poincaré" all match the stored "Poincar\\'e". Scanning 145k rows takes well
    under a second, which is cheaper than maintaining a normalised column.
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
    where, params = _filters(categories, since)
    sql = "SELECT * FROM papers" + (f" WHERE {where}" if where else "")
    terms = textnorm.fold_terms(author)
    out = []
    for row in db.execute(sql + " ORDER BY update_date DESC", params):
        if terms and not textnorm.matches_terms(
                textnorm.fold(row["authors"] or ""), terms):
            continue
        out.append(dict(row) | {"score": None})
        if len(out) >= k:
            break
    return out


def search(db, query: str, k: int = 10, categories=None, since: str = None,
           author: str = None) -> list:
    """The k best matches, as paper dicts each with its `score`."""
    if not query:
        return browse(db, k, categories, since, author)
    store.check_model(db)
    where, params = _filters(categories, since)
    keep = author_ids(db, author) if author else None
    matrix, ids = store.load_matrix(db, where, params, keep_ids=keep)
    if not ids:
        return []
    scores = score_all(matrix, embed_query_normalised(query))
    best = top(scores, k)
    found = papers(db, [ids[i] for i in best])
    return [found[ids[i]] | {"score": float(scores[i])} for i in best]


@functools.lru_cache(maxsize=512)
def embed_query_normalised(query: str) -> np.ndarray:
    """Unit-length embedding of a query, cached.

    Normalising keeps scores in [-1, 1] so they read as genuine cosines.

    The cache is a latency optimisation: ~104ms per repeat, more while a build
    competes for the GPU. It earns its keep because changing any filter -- the
    result count, a category, the dates -- resubmits the same query text.

    It also makes repeated searches return identical cosines, because Ollama's
    embeddings are not deterministic (the reduction order depends on how a
    request is batched, giving ~4e-3 per-component variation under concurrent
    load). That is a nicety rather than a fix: papers within 3e-3 of cosine are
    ties, and either order is as good.
    """
    vector = embedder.embed_query(query)
    unit = (vector / (np.linalg.norm(vector) or 1.0)).astype(np.float32)
    # Shared between callers, so freeze it rather than trust everyone.
    unit.flags.writeable = False
    return unit
