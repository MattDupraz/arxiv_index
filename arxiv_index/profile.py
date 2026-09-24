"""What the reader follows, and what they work on.

Two fields: a list of authors whose papers are worth seeing whatever they are
about, and a list of research interests -- short descriptions, one per thing
the reader actually works on, each carrying a weight. Both are edited from the
web UI.

They live in the reader's settings file (see settings.py), not in the index:
the profile belongs to a person, and an index may be shared between several,
or handed to someone else.

Each interest's embedding is cached **keyed by its exact text**. That is what
makes the invariant cheap to hold: an entry can never be ranked by the vector
of some earlier wording, because a different wording is a different key. An
entry whose text is unchanged finds its vector and costs nothing; only new or
edited entries -- including ones typed into the settings file by hand -- are
sent to Ollama.

Note the descriptions are embedded as *queries*, not as documents: they are
compared against document vectors, so they need the instruct prefix and the CPU
options that `search.embed_query_normalised` applies. Embedding them as
documents would put them in the wrong place in the space.

Why a list rather than one paragraph. A single blob has to be embedded as a
single point, which lands somewhere in the middle of everything the reader
does and is squarely none of it. Separate descriptions each keep their own
direction, and the ranking rule in `web.ResidentIndex.ranked` is then free to
say "closest to any one of these" rather than "closest to their average".
"""

import base64
import binascii

import numpy as np

from . import config, search as search_mod, settings, textnorm

AUTHORS_KEY = "followed_authors"
INTERESTS_KEY = "interests"
BLEND_KEY = "interests_blend"

# Bounds on what the UI may submit. None is a limit anyone will reach by using
# the thing as intended; they exist so a runaway paste cannot put an unbounded
# blob in the settings file. Each embedded interest costs DIM float32s in the
# cache, so the count is what actually matters: 50 x 2560 x 4 is ~500 KB.
MAX_AUTHORS = 500
MAX_INTERESTS = 50
MAX_INTEREST_CHARS = 2_000

DEFAULT_WEIGHT = 1.0
MAX_WEIGHT = 2.0

# How much a second, third, ... matching interest adds; see `blend_decay`.
DEFAULT_BLEND = 0.35

# Little-endian float32, stated rather than left to the platform so that an
# index copied between machines reads back the vectors it stored. float32 and
# not the corpus's float16: this is the query side, which the search path keeps
# in float32 throughout.
VECTOR_DTYPE = "<f4"
VECTOR_BYTES = config.DIM * np.dtype(VECTOR_DTYPE).itemsize


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


def clean_weight(raw) -> float:
    """A submitted weight as a number in [0, MAX_WEIGHT].

    Anything unreadable becomes the default rather than an error: a weight is a
    dial on an entry that is otherwise perfectly good, and refusing the whole
    save over a mistyped one would lose the text with it.
    """
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_WEIGHT
    if value != value:  # NaN, which would poison every comparison downstream
        return DEFAULT_WEIGHT
    return round(max(0.0, min(MAX_WEIGHT, value)), 3)


def clean_interests(raw) -> list:
    """Normalise a submitted interests list to [{"text", "weight"}, ...].

    Accepts bare strings as well as records, so a caller with nothing to say
    about weights can post a plain list. Text is whitespace-collapsed, which
    matters beyond tidiness: unchanged text is what lets `save` reuse an
    existing embedding, and a description that differs only by a line wrap is
    not a different description.
    """
    if not isinstance(raw, (list, tuple)):
        return []
    seen, out = set(), []
    for entry in raw:
        if isinstance(entry, dict):
            text, weight = entry.get("text", ""), entry.get(
                "weight", DEFAULT_WEIGHT)
        else:
            text, weight = entry, DEFAULT_WEIGHT
        text = " ".join(str(text).split())[:MAX_INTEREST_CHARS]
        if not text:
            continue
        key = textnorm.fold(text)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append({"text": text, "weight": clean_weight(weight)})
        if len(out) >= MAX_INTERESTS:
            break
    return out


def clean_blend(raw) -> float:
    """The blend knob, clamped to [0, 1]. See `blend_decay`."""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_BLEND
    if value != value:
        return DEFAULT_BLEND
    return round(max(0.0, min(1.0, value)), 3)


def blend_decay(blend: float, count: int) -> np.ndarray:
    """Weights applied to a paper's matches, best first.

    A paper's interests are scored individually, sorted best-first, and summed
    under these coefficients: 1, b, b^2, ... So the best match always counts in
    full and each further one counts less, which is the whole point -- a paper
    squarely on one project should not be overtaken by one that is vaguely near
    several, but genuinely matching two projects should still beat matching
    one.

    The two ends are exactly the two rules this replaces. At b = 0 only the
    best match survives and the score is a weighted maximum. At b = 1 every
    match counts in full and the score is a weighted sum, which is the same
    ranking as averaging the interest vectors into a single point. The default
    sits nearer the max end.

    Geometric rather than, say, counting the top two: it decays smoothly, needs
    one number, and cannot be gamed by padding the list, since twenty weak
    matches sum to less than 1/(1-b) times the weakest of them.
    """
    return np.power(float(blend), np.arange(max(count, 0), dtype=np.float32),
                    dtype=np.float32)


def _decode(raw):
    """One stored base64 vector as a unit float32 array, or None.

    Returns None rather than raising on anything unexpected -- absent, damaged,
    or the wrong length because config.DIM changed. Every caller treats "no
    vector" as "this entry cannot be ranked by yet", which is the right answer
    in all three cases, and saving the profile again rebuilds it.
    """
    if not raw or not isinstance(raw, str):
        return None
    try:
        buf = base64.b64decode(raw, validate=True)
    except (ValueError, binascii.Error):
        return None
    if len(buf) != VECTOR_BYTES:
        return None
    return np.frombuffer(buf, dtype=VECTOR_DTYPE)


def _encode(unit) -> str:
    return base64.b64encode(
        np.asarray(unit, dtype=VECTOR_DTYPE).tobytes()).decode("ascii")


def _interests() -> list:
    """The interests as written in the settings file, cleaned."""
    return clean_interests(settings.get(INTERESTS_KEY, []))


def _cache() -> dict:
    return settings.read_vector_cache(config.QUERY_EMBEDDER)


def load() -> dict:
    """The stored profile, without the vectors themselves.

    Each interest reports `embedded` rather than its vector: the caller is a
    JSON response to a browser, which has no use for 10 KB of float32 per entry
    and every reason not to be sent it.
    """
    cache = _cache()
    interests = [e | {"embedded": _decode(cache.get(e["text"])) is not None}
                 for e in _interests()]
    return {
        "authors": clean_authors(settings.get(AUTHORS_KEY, [])),
        "interests": interests,
        "blend": clean_blend(settings.get(BLEND_KEY, DEFAULT_BLEND)),
        # Kept so the UI can say "ranking is unavailable" without walking the
        # list itself, and so a caller can distinguish "nothing written yet"
        # from "written, but Ollama was down".
        "embedded": sum(1 for e in interests if e["embedded"]),
    }


def vectors():
    """(Q, weights) for ranking: unit rows and the weight beside each.

    Only entries that have both a usable vector and a non-zero weight appear. A
    zero weight is the reader switching an interest off, and it has to be
    dropped rather than scored as zero: the ranking rule sorts a paper's
    matches and decays them by rank, so a dead entry left in place would push
    real matches into weaker slots.

    Returns (None, None) when nothing is rankable, which every caller reads as
    "ranking is not available yet".
    """
    cache = _cache()
    rows, weights = [], []
    for entry in _interests():
        unit = _decode(cache.get(entry["text"]))
        if unit is None or entry["weight"] <= 0:
            continue
        rows.append(unit)
        weights.append(entry["weight"])
    if not rows:
        return None, None
    return (np.asarray(rows, dtype=np.float32),
            np.asarray(weights, dtype=np.float32))


def embed_missing(prune: bool = False) -> list:
    """Embed every interest the cache has no vector for. Returns the errors.

    Called on save, and again before ranking so that an interest added by
    editing the settings file is picked up without a trip through the UI.
    With `prune`, vectors for texts no longer in the list are dropped.

    Failures are reported, not raised, and leave those entries without a
    vector -- the honest state, since the text is kept and only ranking by it
    is unavailable. Each success is cached as it lands, so a failure half way
    through keeps the ones before it.
    """
    wanted = [e["text"] for e in _interests()]
    cache = _cache()
    failures = []
    for text in wanted:
        if _decode(cache.get(text)) is not None:
            continue
        try:
            unit = search_mod.embed_query_normalised(text)
        except Exception as exc:  # noqa: BLE001 - Ollama down, model missing
            failures.append(str(exc) or exc.__class__.__name__)
            continue
        settings.merge_vector_cache(config.QUERY_EMBEDDER, {text: _encode(unit)})
    if prune:
        settings.merge_vector_cache(config.QUERY_EMBEDDER, {}, keep=set(wanted))
    return failures


def save(authors, interests, blend=None):
    """Store the profile. Returns (profile, error).

    An interest is embedded only when its text is new, which is what makes
    ranking cheap afterwards and what keeps editing one entry from re-billing
    the other nineteen. The text is written first, so it is kept even when
    Ollama is unreachable.
    """
    values = {AUTHORS_KEY: clean_authors(authors),
              INTERESTS_KEY: clean_interests(interests)}
    if blend is not None:
        values[BLEND_KEY] = clean_blend(blend)
    settings.update(**values)

    failures = embed_missing(prune=True)
    error = None
    if failures:
        error = (f"{len(failures)} of {len(values[INTERESTS_KEY])} interest(s) "
                 f"could not be embedded: {failures[0]}")
    return load(), error
