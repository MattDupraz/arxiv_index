"""Cross-encoder reranking of the index's shortlist.

The index is a bi-encoder: query and document are embedded independently, so
their vectors never interact and the ranking is only as good as one dot product
can express. A cross-encoder reads the pair *together* and scores it jointly --
much better, and far too slow to run over 145k papers. So the index proposes
RERANK_CANDIDATES hits and the cross-encoder reorders them; the index only has
to get the right papers into the shortlist, not order them well.

The model is a `ForSequenceClassification` encoder: it takes a (query, document)
pair and emits one logit. Higher is more relevant, and the logit is used
directly as the ranking score.

Causal-LM rerankers (Qwen3-Reranker, prompted with a yes/no question and scored
on those two tokens' logprobs) were tried and dropped: slower for no measured
gain, because they project 2560 -> 151,936 per document only to read two of
those numbers.
"""

import importlib.util
import sys

from . import config

# torch and transformers are imported lazily inside _Model.load(): they are
# optional, and this module must stay importable without them so that indexing
# and plain search still work on a machine that has not installed them.


class RerankUnavailable(RuntimeError):
    pass


def document_text(paper) -> str:
    """What the cross-encoder reads: the whole title and abstract.

    Not truncated. The tokenizer's max_length is the only cap, and it exists as
    a backstop rather than an optimisation -- see RERANK_MAX_TOKENS.
    """
    return " ".join(f"{paper['title']}. {paper['abstract']}".split())


class _Model:
    """Loaded once and held for the life of the process. That is why the server
    benefits far more than the CLI, which pays the load on every invocation."""

    model = tokenizer = torch = None
    labels = 1
    failed = False
    reason = None

    @classmethod
    def load(cls) -> bool:
        if cls.model is not None or cls.failed:
            return not cls.failed
        try:
            import torch
            from transformers import (AutoConfig,
                                      AutoModelForSequenceClassification,
                                      AutoTokenizer)
        except ImportError as exc:
            cls.failed = True
            cls.reason = (f"{exc}; reranking needs torch and transformers "
                          "(see README)")
            return False
        try:
            cls.torch = torch
            if not torch.cuda.is_available():
                # 50 documents through a 568M encoder takes minutes on CPU.
                # Refuse rather than crawl; the caller falls back to vector
                # order, which is far better than a search that appears hung.
                raise RerankUnavailable("no GPU available")

            trust = config.RERANK_TRUST_REMOTE_CODE
            arch = (AutoConfig.from_pretrained(
                config.RERANK_MODEL,
                trust_remote_code=trust).architectures or ["?"])[0]
            if "ForSequenceClassification" not in arch:
                raise RerankUnavailable(
                    f"{config.RERANK_MODEL} is a {arch}; this backend needs a "
                    "ForSequenceClassification cross-encoder. Causal rerankers "
                    "were tried via sentence-transformers and were slower for "
                    "no gain -- see the notes in config.py."
                )

            # Right padding: the model reads CLS at position 0, so padding must
            # go on the end. Left padding would still return plausible numbers
            # while scoring the wrong position.
            cls.tokenizer = AutoTokenizer.from_pretrained(
                config.RERANK_MODEL, padding_side="right",
                trust_remote_code=trust)
            cls.model = AutoModelForSequenceClassification.from_pretrained(
                config.RERANK_MODEL, dtype=torch.float16,
                trust_remote_code=trust).to("cuda").eval()
            cls.labels = int(getattr(cls.model.config, "num_labels", 1))
            return True
        except Exception as exc:  # noqa: BLE001
            # Record why. Degrading to vector order is fine; doing it silently
            # is not -- an out-of-memory looks identical to a missing model.
            cls.failed = True
            cls.reason = f"{type(exc).__name__}: {exc}"
            return False

    @classmethod
    def score(cls, query: str, documents):
        """One logit per document, in batched forward passes.

        Documents are sorted by token length before batching and restored
        afterwards: every sequence is padded to the longest in its batch, and on
        a real shortlist that padding was 40% of the compute. Batches are kept
        small for the same reason -- bigger is not better here.
        """
        limit = config.RERANK_MAX_TOKENS
        lengths = [
            len(cls.tokenizer(query, d, truncation=True,
                              max_length=limit)["input_ids"])
            for d in documents
        ]
        order = sorted(range(len(documents)), key=lambda i: lengths[i])
        out = [0.0] * len(documents)

        with cls.torch.inference_mode():
            for start in range(0, len(order), config.RERANK_BATCH):
                idx = order[start:start + config.RERANK_BATCH]
                chunk = [documents[i] for i in idx]
                enc = cls.tokenizer(
                    [query] * len(chunk), chunk, return_tensors="pt",
                    padding=True, truncation=True, max_length=limit,
                ).to(cls.model.device)
                logits = cls.model(**enc).logits.float()
                # One label: the logit is the score. Two: the usual
                # relevant-minus-irrelevant contrast.
                scores = (logits[:, 0] if cls.labels == 1
                          else logits[:, 1] - logits[:, 0])
                for i, s in zip(idx, scores.tolist()):
                    out[i] = s
        return out


def available() -> bool:
    return _Model.load()


def offerable() -> bool:
    """Whether the UI should offer reranking at all. Cheap: loads nothing.

    `available()` answers the same question definitively, but it pays the model
    load -- several seconds, and ~480 MB of resident torch that a server
    configured without it would otherwise never hold. The page has to decide
    before any search happens, so it cannot pay that.

    So this answers only what can be known for free. A missing dependency is
    conclusive, and a load that has already failed in this process is too. A GPU
    that turns out to be absent is not detectable without importing torch, and
    is left to the run-time path, which falls back to vector order and says why.
    """
    if _Model.model is not None:
        return True
    if _Model.failed:
        return False
    try:
        if "torch" in sys.modules:
            # Already paid for -- GPU_SEARCH imports it -- so ask the real
            # question rather than only whether it is installed.
            return sys.modules["torch"].cuda.is_available()
        return all(importlib.util.find_spec(m) is not None
                   for m in ("torch", "transformers"))
    except Exception:  # noqa: BLE001
        # A half-installed package can make find_spec raise rather than return
        # None, and cuda.is_available() can fail on a broken driver. Either way
        # the answer is "do not offer it"; rendering the page must not depend
        # on the reranker being in a sane state.
        return False


def rerank(query: str, papers):
    """Return `papers` reordered by cross-encoder relevance.

    Each result gains `rerank_margin` (the score ordering is based on) and keeps
    its cosine as `vector_score`, so both are visible side by side.
    """
    if not papers or not query:
        return papers
    if not _Model.load():
        raise RerankUnavailable(_Model.reason or "reranker unavailable")

    scores = _Model.score(query, [document_text(p) for p in papers])

    ranked = []
    for paper, score in zip(papers, scores):
        entry = dict(paper)
        entry["vector_score"] = entry.get("score")
        entry["rerank_margin"] = score
        entry["score"] = score
        ranked.append(entry)
    ranked.sort(key=lambda p: p["rerank_margin"], reverse=True)
    return ranked
