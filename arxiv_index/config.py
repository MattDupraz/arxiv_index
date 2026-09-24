"""How the index is built: the model, the tuning, the measurements behind them.

What differs between the people using it -- which arXiv categories, their
profile -- is in their own settings file; see settings.py.
"""

import sqlite3
from pathlib import Path

from . import settings

# --- Embedding model --------------------------------------------------------
# The default, which the numbers below were measured with, and the one offered
# first. The settings file's `embedding` names the model in use; see
# embedding() for where it comes from when it does not.
#
# Documents are embedded raw. Queries get the Qwen3-Embedding instruct prefix,
# which is what the model was trained to expect on the query side.
DEFAULT_EMBEDDING = {
    "model": "qwen3-embedding:4b",
    "dim": 2560,
    "query_prefix": "Instruct: Given a research question, retrieve relevant "
                    "arXiv paper abstracts\nQuery: ",
    "document_prefix": "",
}


class NoModel(settings.SettingsError):
    """No embedding model has been chosen for this index yet."""


_embedding = None


def embedding() -> dict:
    """The embedding model, its dimension and its prefixes.

    Worked out when first needed rather than at import, because a fresh
    install has none: the model is chosen when the index is set up, by
    `build` or on the setup page, and until then there is nothing to assume.
    It comes from the settings if they name one, and otherwise from the index,
    which records the model its vectors were made with (see store.record).
    """
    global _embedding
    if _embedding is None:
        if "embedding" in settings.load():
            _embedding = settings.embedding(DEFAULT_EMBEDDING)
        else:
            _embedding = _recorded()
        if _embedding is None:
            raise NoModel("No embedding model has been chosen yet. Run `build`, "
                          "or open the setup page `serve` shows.")
    return _embedding


def ready() -> bool:
    """Whether an embedding model has been chosen."""
    try:
        embedding()
    except NoModel:
        return False
    return True


def use(entry: dict) -> dict:
    """Name `entry` as the model in the settings, and use it from now on."""
    global _embedding
    settings.update(embedding=entry)
    _embedding = None
    return embedding()


def _recorded():
    """The model the index's vectors were made with, if it has any."""
    if not DB_PATH.exists():
        return None
    db = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        meta = dict(db.execute("SELECT key, value FROM meta WHERE key IN "
                               "('model', 'dim', 'document_prefix')"))
    except sqlite3.DatabaseError:
        return None
    finally:
        db.close()
    if "model" not in meta:
        return None
    default = meta["model"] == DEFAULT_EMBEDDING["model"]
    return {"model": meta["model"],
            "dim": int(meta.get("dim") or DEFAULT_EMBEDDING["dim"]),
            "query_prefix": DEFAULT_EMBEDDING["query_prefix"] if default else "",
            "document_prefix": meta.get("document_prefix", "")}


def model() -> str:
    return embedding()["model"]


def dim() -> int:
    return embedding()["dim"]


def query_embedder() -> dict:
    """What an embedded query depends on, so caches of them can tell when it
    changes."""
    e = embedding()
    return {"model": e["model"], "query_prefix": e["query_prefix"]}

# Ollama runtime options for indexing.
#   num_ctx    abstracts top out around ~500 tokens; 2048 is generous headroom.
#   num_batch  physical batch; larger keeps the GPU busy across a doc batch.
#   num_gpu    99 = offload everything (see OLLAMA_QUERY_OPTIONS for why this
#              must be explicit rather than left to the default).
OLLAMA_OPTIONS = {"num_ctx": 2048, "num_batch": 8192, "num_gpu": 99}

# Query embedding runs on the CPU. One short text costs 175ms there against
# 89ms on the GPU, and it frees the 4.1 GB the embedder would otherwise hold
# for a build running alongside, or anything else using the GPU.
#
# `num_gpu` must be stated explicitly on BOTH paths. Ollama does not move a
# model back on its own: once loaded with num_gpu=0 it stays on the CPU, and a
# request that merely omits the option will not return it to the GPU.
#
# CPU and GPU vectors are not identical (cos ~0.998, components differ by up to
# 7e-3). Measured effect on retrieval: same top-1 and the same top-10 set, only
# minor reordering within it.
OLLAMA_QUERY_OPTIONS = {"num_ctx": 2048, "num_batch": 8192, "num_gpu": 0}

# Docs per ollama.embed() call. Throughput is flat from 64 upward on this GPU
# (compute-bound, not batching-bound), so 64 keeps checkpoints frequent.
BATCH_SIZE = 64

# --- Storage ----------------------------------------------------------------
# The index is ~/.arxiv_index or $ARXIV_INDEX_DIR. The snapshot is whatever
# `build` is given, else this one in the current directory.
SNAPSHOT = Path("arxiv-metadata-oai-snapshot.json")
INDEX_DIR = settings.home()
DB_PATH = INDEX_DIR / "papers.db"
VEC_PATH = INDEX_DIR / "vectors.f16"

# Vectors are stored L2-normalised as float16: 2560 dims x 2 bytes = 5 KB/paper.
# Normalisation makes cosine similarity a plain dot product; float16 halves the
# bytes read per search with no measurable effect on ranking.
VEC_DTYPE = "float16"


def slot_bytes() -> int:
    """Bytes per stored vector: two per float16."""
    return dim() * 2


def document_text(title: str, abstract: str) -> str:
    """The text that gets embedded for a paper. Title first, then abstract.

    arXiv metadata wraps both at ~80 chars and indents abstracts by two spaces;
    collapsing that whitespace keeps the tokenisation clean.
    """
    title = " ".join(title.split())
    abstract = " ".join(abstract.split())
    return f"{embedding()['document_prefix']}{title}\n\n{abstract}"


def query_text(query: str) -> str:
    """The text that gets embedded for a search query."""
    return f"{embedding()['query_prefix']}{query.strip()}"


def in_scope(categories: str, scope) -> bool:
    """True if a whitespace-separated category string touches `scope`.

    ANY of a paper's categories counts, so cross-listed papers are included,
    not just those whose primary category matches.
    """
    return any(c in scope for c in categories.split())
