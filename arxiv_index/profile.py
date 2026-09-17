"""What the reader follows, and what they work on.

Two fields: a list of authors whose papers are worth seeing whatever they are
about, and a paragraph describing the reader's own research interests. Both are
edited from the web UI.

They live in the index's `meta` table, alongside the model name and the update
cursor, rather than in a file of their own. That table is already where things
belonging to *this* index go, it is written transactionally with everything
else, and a profile then travels with the papers it describes when the index is
copied -- which is the behaviour the README already promises for the two index
files.

The interests paragraph is stored **with its embedding**. Ranking a date window
against it is one dot product against vectors already in memory, and paying an
Ollama round trip for the same unchanged paragraph every time would dominate
that. The vector is refreshed exactly when the text it came from changes.

Note the paragraph is embedded as a *query*, not as a document: it is being
compared against document vectors, so it needs the instruct prefix and the CPU
options that `search.embed_query_normalised` applies. Embedding it as a
document would put it in the wrong place in the space.
"""

import base64
import binascii
import contextlib
import json

import numpy as np

from . import config, search as search_mod, store, textnorm

AUTHORS_KEY = "followed_authors"
INTERESTS_KEY = "interests"
VECTOR_KEY = "interests_vector"

# Bounds on what the UI may submit. Neither is a limit anyone will reach by
# using the thing as intended; they exist so a runaway paste cannot put an
# unbounded blob in the metadata table.
MAX_AUTHORS = 500
MAX_INTERESTS = 20_000

# Little-endian float32, stated rather than left to the platform so that an
# index copied between machines reads back the vector it stored. float32 and
# not the corpus's float16: this is the query side, which the search path keeps
# in float32 throughout.
VECTOR_DTYPE = "<f4"


def clean_authors(raw) -> list:
    """Trim, drop blanks, de-duplicate and cap a submitted list of names.

    De-duplication is on the folded form, so "Noether" and "noether " are one
    entry -- they match exactly the same papers, and showing both in the list
    would just look like the UI had failed to save.
    """
    if not isinstance(raw, (list, tuple)):
        # A bare string would otherwise iterate into one "author" per
        # character, which is a worse answer than none.
        return []
    seen, out = set(), []
    for entry in raw:
        name = " ".join(str(entry).split())
        if not name:
            continue
        key = textnorm.fold(name)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(name)
        if len(out) >= MAX_AUTHORS:
            break
    return out


def load(db) -> dict:
    """The stored profile. Missing or corrupt values read as empty."""
    raw = store.get_meta(db, AUTHORS_KEY)
    authors = []
    if raw:
        try:
            parsed = json.loads(raw)
        except ValueError:
            parsed = None
        if isinstance(parsed, list):
            authors = [str(a) for a in parsed]
    return {
        "authors": authors,
        "interests": store.get_meta(db, INTERESTS_KEY, "") or "",
        "vector": bool(store.get_meta(db, VECTOR_KEY)),
    }


def vector(db):
    """The stored interests embedding as a unit float32 array, or None.

    Returns None rather than raising on anything unexpected -- absent, damaged,
    or the wrong length because config.DIM changed. The callers all treat "no
    vector" as "ranking is not available yet", which is the right answer in
    every one of those cases, and saving the profile again rebuilds it.
    """
    raw = store.get_meta(db, VECTOR_KEY)
    if not raw:
        return None
    try:
        buf = base64.b64decode(raw, validate=True)
    except (ValueError, binascii.Error):
        return None
    if len(buf) != config.DIM * np.dtype(VECTOR_DTYPE).itemsize:
        return None
    return np.frombuffer(buf, dtype=VECTOR_DTYPE)


def save(db, authors, interests, lock=None):
    """Store the profile. Returns (profile, error).

    The interests text is re-embedded only when it has changed, which is what
    makes ranking cheap afterwards.

    `lock` is the caller's database lock, taken around each touch of the
    connection and released for the embedding call. The web server shares one
    connection across its handler threads, and an Ollama round trip is far too
    long to hold that: every search on the page would queue behind it.

    An embedding failure is reported, not raised, and leaves the profile saved
    with no vector. That is the honest state -- the text is stored, ranking by
    it is not available until Ollama can be reached -- and it is why the old
    vector is dropped *before* the new text is embedded rather than after: the
    one thing that must never happen is ranking a new paragraph by the vector
    of the old one.
    """
    lock = lock if lock is not None else contextlib.nullcontext()
    authors = clean_authors(authors)
    interests = interests.strip()[:MAX_INTERESTS] if isinstance(
        interests, str) else ""

    with lock:
        before = load(db)
        store.set_meta(db, AUTHORS_KEY, json.dumps(authors))
        store.set_meta(db, INTERESTS_KEY, interests)
        reuse = interests == before["interests"] and before["vector"]
        if not reuse:
            store.set_meta(db, VECTOR_KEY, "")

    error = None
    if not reuse and interests:
        try:
            unit = search_mod.embed_query_normalised(interests)
        except Exception as exc:  # noqa: BLE001 - Ollama down, model missing
            error = str(exc) or exc.__class__.__name__
        else:
            encoded = base64.b64encode(
                np.asarray(unit, dtype=VECTOR_DTYPE).tobytes()).decode("ascii")
            with lock:
                store.set_meta(db, VECTOR_KEY, encoded)

    with lock:
        return load(db), error
