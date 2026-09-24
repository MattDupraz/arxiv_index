"""How the index is built: the model, the tuning, the measurements behind them.

What differs between the people using it -- which arXiv categories, their
profile -- is in their own settings file; see settings.py.
"""

from pathlib import Path

from . import settings

# --- Embedding model --------------------------------------------------------
# The default, which the numbers below were measured with. The settings file's
# `embedding` can name another Ollama model; see settings.embedding.
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

EMBEDDING = settings.embedding(DEFAULT_EMBEDDING)
MODEL = EMBEDDING["model"]
DIM = EMBEDDING["dim"]
QUERY_PREFIX = EMBEDDING["query_prefix"]
DOCUMENT_PREFIX = EMBEDDING["document_prefix"]

# What an embedded query depends on, so caches of them can tell when it changes.
QUERY_EMBEDDER = {"model": MODEL, "query_prefix": QUERY_PREFIX}

# Ollama runtime options for indexing.
#   num_ctx    abstracts top out around ~500 tokens; 2048 is generous headroom.
#   num_batch  physical batch; larger keeps the GPU busy across a doc batch.
#   num_gpu    99 = offload everything (see OLLAMA_QUERY_OPTIONS for why this
#              must be explicit rather than left to the default).
OLLAMA_OPTIONS = {"num_ctx": 2048, "num_batch": 8192, "num_gpu": 99}

# Query embedding runs on the CPU. One short text costs 175ms there against
# 89ms on the GPU, and it frees the 4.1 GB the embedder would otherwise hold
# for the vector matrix (GPU_SEARCH) and for a build running alongside.
#
# `num_gpu` must be stated explicitly on BOTH paths. Ollama does not move a
# model back on its own: once loaded with num_gpu=0 it stays on the CPU, and a
# request that merely omits the option will not return it to the GPU.
#
# CPU and GPU vectors are not identical (cos ~0.998, components differ by up to
# 7e-3). Measured effect on retrieval: same top-1 and the same top-10 set, only
# minor reordering within it.
OLLAMA_QUERY_OPTIONS = {"num_ctx": 2048, "num_batch": 8192, "num_gpu": 0}

# Hold the vector matrix in VRAM and score there. Measured 418ms -> 2.8ms for
# 145k rows, with the top-10 unchanged (differences ~2.5e-4). Costs 747 MB of
# VRAM and a 0.14s upload whenever the index grows.
#
# Server only. A CLI search is a fresh process, so it would pay the torch import
# and the upload to save 0.4s -- a net loss. Set False to keep everything on CPU.
GPU_SEARCH = True

# Docs per ollama.embed() call. Throughput is flat from 64 upward on this GPU
# (compute-bound, not batching-bound), so 64 keeps checkpoints frequent.
BATCH_SIZE = 64

# --- Storage ----------------------------------------------------------------
# The index is ~/.arxiv_index or $ARXIV_INDEX_DIR. The snapshot is whatever
# `build` is given, else this one in the current directory.
SNAPSHOT = Path("arxiv-metadata-oai-snapshot.json")
INDEX_DIR = settings.index_dir()
DB_PATH = INDEX_DIR / "papers.db"
VEC_PATH = INDEX_DIR / "vectors.f16"

# Vectors are stored L2-normalised as float16: 2560 dims x 2 bytes = 5 KB/paper.
# Normalisation makes cosine similarity a plain dot product; float16 halves the
# bytes read per search with no measurable effect on ranking.
VEC_DTYPE = "float16"


def document_text(title: str, abstract: str) -> str:
    """The text that gets embedded for a paper. Title first, then abstract.

    arXiv metadata wraps both at ~80 chars and indents abstracts by two spaces;
    collapsing that whitespace keeps the tokenisation clean.
    """
    title = " ".join(title.split())
    abstract = " ".join(abstract.split())
    return f"{DOCUMENT_PREFIX}{title}\n\n{abstract}"


def query_text(query: str) -> str:
    """The text that gets embedded for a search query."""
    return f"{QUERY_PREFIX}{query.strip()}"


def in_scope(categories: str, scope) -> bool:
    """True if a whitespace-separated category string touches `scope`.

    ANY of a paper's categories counts, so cross-listed papers are included,
    not just those whose primary category matches.
    """
    return any(c in scope for c in categories.split())
