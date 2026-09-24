"""The reader's own settings, kept in a file of their own.

    ~/.arxiv-index/config.json      ($ARXIV_INDEX_CONFIG overrides)

The same directory holds the index (papers.db, vectors.f16) and the cache of
interest embeddings, so everything that is the reader's rather than the code's
is in one place.

`config.py` is how the index is built -- the model, the tuning, the measurements
behind them -- and is the same for everyone. This file is who is using it: which
arXiv categories they care about, who they follow, what they work on, when their
server should top itself up, and where the index lives. Several people can then
share one index, each with their own file, and copying an index to someone else
does not hand them your profile.

    {
      "categories": ["math.AC", "math.AG", "math.CO"],
      "index_dir": "~/.arxiv-index",
      "snapshot": "~/Downloads/arxiv-metadata-oai-snapshot.json",
      "embedding": {"model": "nomic-embed-text", "dim": 768,
                    "query_prefix": "search_query: ",
                    "document_prefix": "search_document: "},
      "followed_authors": ["Emmy Noether"],
      "interests": [{"text": "invariant theory of finite groups", "weight": 1}],
      "interests_blend": 0.35,
      "auto_update": {"mode": "daily", "hours": 6, "at": "07:00"}
    }

Every key is optional. Relative paths are read from the file's own directory.

JSON rather than TOML because the web UI writes the profile back, and the
standard library can read TOML but not write it. The file is re-read on every
use, so a hand edit is picked up without a restart -- except `categories`,
`embedding` and the two paths, which a running server fixes at start.

`embedding` is the odd one out: it describes the index rather than the reader.
Everyone sharing an index has to name the model it was built with, and
`store.check_model` refuses anything else.

The embeddings of the interests are *not* kept here: 10 KB of base64 per entry
would bury the text someone may want to edit by hand. They go in a cache keyed
by their text (see `profile`), interest-vectors.json beside this file, which
can be deleted at any time.
"""

import json
import os
import re
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Settings, index and cache all live here unless the settings say otherwise.
HOME = Path.home() / ".arxiv-index"

DEFAULT_CATEGORIES = ("math.AC", "math.AG", "math.CO")

# arXiv category names: an archive, optionally a dot and a subject class --
# math.AG, hep-th, cs.LG, physics.acc-ph, q-bio.NC. Checked so that a typo is
# reported when the file is read, rather than by an update that quietly finds
# nothing.
CATEGORY = re.compile(r"[a-z][a-z-]*(\.[A-Za-z][A-Za-z-]*)?")

# Keys that used to live in the index's `meta` table. See migrate_legacy.
LEGACY_KEYS = ("followed_authors", "interests", "interests_vector",
               "interests_blend", "auto_update")

# Read-modify-write of the file is serialised within a process. Across
# processes only the web server writes it, so nothing more is needed.
_lock = threading.RLock()


class SettingsError(SystemExit):
    """The settings file exists but cannot be used as it stands.

    A SystemExit, like the index's other "fix this and run again" errors, so
    the CLI prints the message rather than a traceback -- including when the
    file is first read, at import.
    """


def path() -> Path:
    override = os.environ.get("ARXIV_INDEX_CONFIG")
    if override:
        return Path(override).expanduser()
    return HOME / "config.json"


def cache_dir() -> Path:
    return HOME


def load() -> dict:
    """The whole file as a dict; empty if there is no file yet."""
    target = path()
    try:
        text = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise SettingsError(f"Cannot read {target}: {exc}") from None
    if not text.strip():
        return {}
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise SettingsError(f"{target} is not valid JSON: {exc}") from None
    if not isinstance(data, dict):
        raise SettingsError(f"{target} must hold a JSON object.")
    return data


def get(key: str, default=None):
    return load().get(key, default)


def update(**values) -> dict:
    """Set some keys, keeping every other one -- including keys this version
    does not know about. Written to a temporary file and renamed into place,
    so a crash mid-write cannot leave half a file."""
    with _lock:
        data = load()
        data.update(values)
        target = path()
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                       encoding="utf-8")
        os.replace(tmp, target)
        return data


def write_default() -> bool:
    """Create the file with the default categories if it does not exist."""
    with _lock:
        if path().exists():
            return False
        update(categories=list(DEFAULT_CATEGORIES))
        return True


def clean_categories(raw) -> list:
    if not isinstance(raw, (list, tuple)):
        raise SettingsError(
            f'"categories" in {path()} must be a list, e.g. ["math.AG"].')
    out = []
    for entry in raw:
        name = str(entry).strip()
        if not CATEGORY.fullmatch(name):
            raise SettingsError(
                f"{name!r} in {path()} is not an arXiv category "
                "(expected something like math.AG or hep-th).")
        if name not in out:
            out.append(name)
    if not out:
        raise SettingsError(f'"categories" in {path()} lists nothing.')
    return out


def categories() -> list:
    """The categories this reader wants, in the order they gave them."""
    return clean_categories(get("categories", list(DEFAULT_CATEGORIES)))


def _path_setting(key: str, default: Path) -> Path:
    raw = get(key)
    if not raw:
        return default
    value = Path(str(raw)).expanduser()
    return value if value.is_absolute() else (path().parent / value).resolve()


def index_dir() -> Path:
    return _path_setting("index_dir", HOME)


def snapshot() -> Path:
    return _path_setting("snapshot", ROOT / "arxiv-metadata-oai-snapshot.json")


def embedding(default: dict) -> dict:
    """The embedding model and how to prompt it, over `default`.

    A model other than the default needs its dimension stated: it cannot be
    guessed, and asking Ollama here would make importing the package depend on
    it running. Its prefixes default to none, since the default's are Qwen3's
    and mean nothing to another model -- each model's card says what it wants.
    """
    raw = get("embedding")
    if raw is None:
        return dict(default)
    if not isinstance(raw, dict):
        raise SettingsError(
            f'"embedding" in {path()} must be an object, e.g. '
            '{"model": "nomic-embed-text", "dim": 768}.')

    model = raw.get("model", default["model"])
    if not isinstance(model, str) or not model.strip():
        raise SettingsError(f'"embedding.model" in {path()} must name a model.')
    model = model.strip()
    same = model == default["model"]
    out = dict(default) if same else {"model": model, "query_prefix": "",
                                      "document_prefix": ""}

    if "dim" in raw:
        dim = raw["dim"]
        if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
            raise SettingsError(
                f'"embedding.dim" in {path()} must be a positive whole number.')
        out["dim"] = dim
    elif not same:
        raise SettingsError(
            f'"embedding" in {path()} names {model!r} but not its "dim". '
            f'`ollama show {model}` gives it as the embedding length.')

    for key in ("query_prefix", "document_prefix"):
        if key in raw:
            if not isinstance(raw[key], str):
                raise SettingsError(f'"embedding.{key}" in {path()} must be text.')
            out[key] = raw[key]
    return out


def migrate_legacy(db, embedder: dict) -> bool:
    """Move a profile stored in the index into this file. True if it did.

    Earlier versions kept the profile and the auto-update setting in the
    index's `meta` table. They are copied here -- the interest vectors into the
    cache -- and only then deleted from the index, so an interruption leaves
    them in place for the next attempt. A key already set in this file wins:
    it is the newer of the two.
    """
    rows = {r[0]: r[1] for r in db.execute(
        f"SELECT key, value FROM meta WHERE key IN "
        f"({','.join('?' * len(LEGACY_KEYS))})", LEGACY_KEYS)}
    if not rows:
        return False

    with _lock:
        current = load()
        values, vectors = {}, {}
        if "followed_authors" in rows and "followed_authors" not in current:
            try:
                values["followed_authors"] = json.loads(rows["followed_authors"])
            except ValueError:
                pass
        if "interests" in rows and "interests" not in current:
            try:
                parsed = json.loads(rows["interests"])
            except ValueError:
                # The single-paragraph format, with its vector alongside.
                parsed = [{"text": rows["interests"],
                           "vector": rows.get("interests_vector", "")}]
            if isinstance(parsed, list):
                entries = []
                for entry in parsed:
                    if not isinstance(entry, dict):
                        continue
                    text = " ".join(str(entry.get("text", "")).split())
                    if not text:
                        continue
                    entries.append({"text": text,
                                    "weight": entry.get("weight", 1.0)})
                    if isinstance(entry.get("vector"), str) and entry["vector"]:
                        vectors[text] = entry["vector"]
                values["interests"] = entries
        if "interests_blend" in rows and "interests_blend" not in current:
            try:
                values["interests_blend"] = float(rows["interests_blend"])
            except ValueError:
                pass
        if "auto_update" in rows and "auto_update" not in current:
            try:
                values["auto_update"] = json.loads(rows["auto_update"])
            except ValueError:
                pass
        if "categories" not in current:
            # Whoever built this index with a profile in it was using the
            # categories that were then hard-coded.
            values["categories"] = list(DEFAULT_CATEGORIES)

        if vectors:
            merge_vector_cache(embedder, vectors)
        update(**values)

    db.execute(f"DELETE FROM meta WHERE key IN "
               f"({','.join('?' * len(LEGACY_KEYS))})", LEGACY_KEYS)
    db.commit()
    return True


# --- Interest vector cache ---------------------------------------------------
# {"embedding": {"model": ..., "query_prefix": ...}, "vectors": {text: base64}}.
# Keyed by the exact text, so a vector can never outlive the wording it was
# made from, and by model and query prefix, so changing either cannot serve
# vectors made the old way.


def _cache_path() -> Path:
    return cache_dir() / "interest-vectors.json"


def read_vector_cache(embedder: dict) -> dict:
    try:
        data = json.loads(_cache_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict) or data.get("embedding") != embedder:
        return {}
    vectors = data.get("vectors")
    return vectors if isinstance(vectors, dict) else {}


def merge_vector_cache(embedder: dict, vectors: dict, keep=None) -> None:
    """Add `vectors` to the cache. With `keep`, drop every text not in it, so
    the cache does not grow with every wording ever tried."""
    with _lock:
        current = read_vector_cache(embedder)
        current.update(vectors)
        if keep is not None:
            current = {t: v for t, v in current.items() if t in keep}
        target = _cache_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(json.dumps({"embedding": embedder, "vectors": current}),
                       encoding="utf-8")
        os.replace(tmp, target)
