"""Thin wrapper around Ollama's embedding endpoint."""

import time

import numpy as np
import ollama

from . import config, settings


class EmbedError(RuntimeError):
    pass


def embed(texts, *, retries: int = 4, options=None) -> np.ndarray:
    """Embed a list of texts, returning a (len(texts), DIM) float32 array.

    Ollama occasionally drops a request when the GPU is saturated; retrying with
    a short backoff is enough to ride that out. Retries are safe because
    embedding is a pure function of the input.
    """
    model, dim = config.model(), config.dim()
    if not texts:
        return np.empty((0, dim), dtype=np.float32)

    options = options if options is not None else config.OLLAMA_OPTIONS
    last = None
    for attempt in range(retries):
        try:
            response = ollama.embed(
                model=model,
                input=list(texts),
                options=options,
                truncate=True,
            )
            break
        except Exception as exc:  # noqa: BLE001 - surfaced after the final retry
            last = exc
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    else:
        raise EmbedError(
            f"embedding failed after {retries} attempts: {last}") from last

    # Outside the retries: a wrong dimension is a setting, not a hiccup.
    vectors = np.asarray(response.embeddings, dtype=np.float32)
    if vectors.ndim == 2 and vectors.shape[1] != dim:
        raise EmbedError(
            f"{model} gives {vectors.shape[1]} dimensions, but "
            f'"embedding.dim" is {dim}. Correct it in {settings.path()}.')
    if vectors.shape != (len(texts), dim):
        raise EmbedError(f"expected {(len(texts), dim)}, got {vectors.shape}")
    return vectors


def embed_documents(titles_and_abstracts) -> np.ndarray:
    return embed([config.document_text(t, a) for t, a in titles_and_abstracts])


def embed_query(query: str) -> np.ndarray:
    """Embed a search query on the CPU; see OLLAMA_QUERY_OPTIONS for why."""
    return embed([config.query_text(query)],
                 options=config.OLLAMA_QUERY_OPTIONS)[0]


def embedding_models() -> list:
    """The installed Ollama models that can embed, as {"name", "dim"}.

    Ollama says which models embed and how long their vectors are. One that
    does not say (an older Ollama) is included, its length found by embedding
    a word with it.
    """
    try:
        installed = [m.model for m in ollama.list().models]
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Cannot reach Ollama ({exc}). Is it running?") from exc
    out = []
    for name in installed:
        shown = ollama.show(name)
        capabilities = getattr(shown, "capabilities", None)
        if capabilities is not None and "embedding" not in capabilities:
            continue
        dim = next((v for k, v in (shown.modelinfo or {}).items()
                    if k.endswith(".embedding_length")), None)
        if dim is None:
            try:
                dim = len(ollama.embed(model=name, input="dimension").embeddings[0])
            except Exception:  # noqa: BLE001 - not an embedding model after all
                continue
        out.append({"name": name, "dim": int(dim)})
    return out


def check_available() -> None:
    """Fail early with a useful message if the model is not pulled."""
    try:
        names = {m.model for m in ollama.list().models}
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"Cannot reach Ollama ({exc}). Is the service running?"
        ) from exc
    if config.model() not in names:
        raise SystemExit(
            f"Model {config.model()!r} is not available. Pull it with:\n"
            f"    ollama pull {config.model()}"
        )
