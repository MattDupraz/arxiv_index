"""Thin wrapper around Ollama's embedding endpoint."""

import time

import numpy as np
import ollama

from . import config


class EmbedError(RuntimeError):
    pass


def embed(texts, *, retries: int = 4, options=None) -> np.ndarray:
    """Embed a list of texts, returning a (len(texts), DIM) float32 array.

    Ollama occasionally drops a request when the GPU is saturated; retrying with
    a short backoff is enough to ride that out. Retries are safe because
    embedding is a pure function of the input.
    """
    if not texts:
        return np.empty((0, config.DIM), dtype=np.float32)

    options = options if options is not None else config.OLLAMA_OPTIONS
    last = None
    for attempt in range(retries):
        try:
            response = ollama.embed(
                model=config.MODEL,
                input=list(texts),
                options=options,
                truncate=True,
            )
            vectors = np.asarray(response.embeddings, dtype=np.float32)
            if vectors.shape != (len(texts), config.DIM):
                raise EmbedError(
                    f"expected {(len(texts), config.DIM)}, got {vectors.shape}"
                )
            return vectors
        except Exception as exc:  # noqa: BLE001 - surfaced after the final retry
            last = exc
            if attempt < retries - 1:
                time.sleep(2 ** attempt)

    raise EmbedError(f"embedding failed after {retries} attempts: {last}") from last


def embed_documents(titles_and_abstracts) -> np.ndarray:
    return embed([config.document_text(t, a) for t, a in titles_and_abstracts])


def embed_query(query: str) -> np.ndarray:
    """Embed a search query on the CPU, keeping the GPU free for reranking."""
    return embed([config.query_text(query)],
                 options=config.OLLAMA_QUERY_OPTIONS)[0]


def check_available() -> None:
    """Fail early with a useful message if the model is not pulled."""
    try:
        names = {m.model for m in ollama.list().models}
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"Cannot reach Ollama ({exc}). Is the service running?"
        ) from exc
    if config.MODEL not in names:
        raise SystemExit(
            f"Model {config.MODEL!r} is not available. Pull it with:\n"
            f"    ollama pull {config.MODEL}"
        )
