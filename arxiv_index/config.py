"""Central configuration for the arXiv index."""

from pathlib import Path

# --- Corpus scope -----------------------------------------------------------
# A paper is in scope if ANY of its categories is one of these (i.e. cross-listed
# papers are included, not just those whose primary category matches).
CATEGORIES = ("math.AC", "math.AG", "math.CO")

# --- Embedding model --------------------------------------------------------
MODEL = "qwen3-embedding:4b"
DIM = 2560

# Ollama runtime options for indexing.
#   num_ctx    abstracts top out around ~500 tokens; 2048 is generous headroom.
#   num_batch  physical batch; larger keeps the GPU busy across a doc batch.
#   num_gpu    99 = offload everything (see OLLAMA_QUERY_OPTIONS for why this
#              must be explicit rather than left to the default).
OLLAMA_OPTIONS = {"num_ctx": 2048, "num_batch": 8192, "num_gpu": 99}

# Query embedding runs on the CPU. One short text costs 175ms there against
# 89ms on the GPU -- +86ms on a search that takes ~2s with reranking -- and it
# frees the 4.1 GB the embedder would otherwise hold, which is what makes room
# for a larger reranker.
#
# `num_gpu` must be stated explicitly on BOTH paths. Ollama does not move a
# model back on its own: once loaded with num_gpu=0 it stays on the CPU, and a
# request that merely omits the option will not return it to the GPU.
#
# CPU and GPU vectors are not identical (cos ~0.998, components differ by up to
# 7e-3). Measured effect on retrieval: same top-1 and the same top-10 set, only
# minor reordering within it -- and the reranker rescores that shortlist anyway,
# so the vector stage only has to select the right papers, not order them.
OLLAMA_QUERY_OPTIONS = {"num_ctx": 2048, "num_batch": 8192, "num_gpu": 0}

# Documents are embedded raw. Queries get the Qwen3-Embedding instruct prefix,
# which is what the model was trained to expect on the query side.
QUERY_INSTRUCTION = (
    "Given a research question, retrieve relevant arXiv paper abstracts"
)

# --- Reranking ---------------------------------------------------------------
# A cross-encoder rescores the index's shortlist. It must be a
# `ForSequenceClassification` model: it takes a (query, document) pair and emits
# one logit, used directly as the ranking score.
#
# Chosen by measurement, on 50 known-item queries against identical shortlists:
#
#                                 recall@1  recall@5    MRR   50 docs   VRAM
#   vector only                      0.500     0.760   0.622
#   Qwen3-Reranker-0.6B              0.760     0.820   0.781     1.66s  1.11G
#   Qwen3-Reranker-4B                0.800     0.860   0.827     4.09s  7.54G
#   bge-reranker-v2-m3               0.820     0.860   0.835     0.65s  1.13G
#   gte-reranker-modernbert-base     0.860     0.860   0.860     0.38s  0.36G
#
# modernbert beat bge on 2 cases, lost 0, tied 41. Two discordant pairs is not
# statistical significance -- but every earlier comparison traded wins for
# losses, and this one is strictly non-worse while being 1.7x faster in a third
# of the VRAM.
#
# It also fixed a real failure. Asked for "K-rings of matroids", bge scored a
# paper on g-elements 2.35 against 2.26 for the actual K-rings paper -- wrong
# order, 0.09 apart in a 2.9 range, i.e. indifferent. modernbert scores them
# 3.79 and 1.74: it knows the difference.
#
# Models that did NOT work here, so they are not retried by accident:
#   zerank-2 / zerank-1-small   never finished loading (CPU-bound, >5 min)
#   gte-multilingual-base       crashed the GPU (HSA exception, custom kernels)
#   jina-reranker-v2            custom code needs a pre-5.x transformers
#   jina-reranker-v3            score head absent from the checkpoint
# The pattern: anything relying on custom remote code is a lottery on RDNA4 with
# transformers 5.x. Plain ForSequenceClassification on stock transformers works.
RERANK_MODEL = "Alibaba-NLP/gte-reranker-modernbert-base"

# Some rerankers define their architecture in Python shipped with the model
# rather than in transformers -- gte-multilingual-reranker-base is a
# `NewForSequenceClassification` from Alibaba-NLP/new-impl. Loading those
# executes downloaded code inside this process.
#
# An explicit switch rather than a blanket default: it should be a decision
# taken per model, by someone who has looked at where the model comes from, not
# something that silently applies to whatever RERANK_MODEL is set to next.
RERANK_TRUST_REMOTE_CODE = False

# How many of the index's hits get rescored. Cost is linear in this, and it
# also sets a ceiling: a paper outside the shortlist cannot be recovered no
# matter how good the reranker is.
#
# 50 rather than 100, on measurement. Doubling the shortlist moved the
# known-item ceiling only from 86% to 88% -- one paper in fifty -- because six
# of the seven misses are not in the top 100 either. Those are failures of the
# embedding, not of the cutoff, so the extra second of latency bought almost
# nothing.
RERANK_CANDIDATES = 50

# Documents per forward pass. Small beats large: every sequence in a batch is
# padded to the longest one in it, so oversized batches waste compute on padding.
RERANK_BATCH = 16

# Documents go to the cross-encoder whole. There is a token cap only as a
# backstop against pathological input; the longest title+abstract in the corpus
# is 3,928 characters, well inside it.
#
# There used to be a 1200-character cut, on the theory that cost grows with
# length. Measured, it does not: 50 documents took 0.77s at 1200 chars and 0.64s
# at 2000 (the difference is noise). Batching and length-bucketing mean 500
# tokens versus 300 is free on a 568M encoder. Meanwhile the cut was truncating
# 12.3% of papers -- 17,940 of them, losing ~287 characters each, biased toward
# the longest and most substantial abstracts. It was paying real quality for no
# speed.
RERANK_MAX_TOKENS = 2048


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
ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = ROOT / "arxiv-metadata-oai-snapshot.json"
INDEX_DIR = ROOT / "index"
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
    return f"{title}\n\n{abstract}"


def query_text(query: str) -> str:
    """The text that gets embedded for a search query."""
    return f"Instruct: {QUERY_INSTRUCTION}\nQuery: {query.strip()}"


def in_scope(categories: str) -> bool:
    """True if a whitespace-separated category string touches our scope."""
    return any(c in CATEGORIES for c in categories.split())
