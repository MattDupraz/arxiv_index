"""Incremental top-up from the live arXiv API.

The Kaggle snapshot is only used for the initial backfill. From then on this
walks the arXiv API back from the newest paper until it reaches the cursor.

Two API traps this works around
-------------------------------
**1. ``lastUpdatedDate:[A TO B]`` does not filter on the last-update date.**
Despite the name, the range filter matches on the paper's original submission,
while ``sortBy=lastUpdatedDate`` sorts on the actual last update. Using the
range filter to bound a window therefore drops revisions of older papers -- a
v2 posted today whose v1 predates the window is silently excluded. That is
precisely the case an incremental updater exists to catch, so no range filter
is used here; the walk is bounded by the cursor comparison alone.

**2. Stopping early must not advance the cursor.**
Results come back newest-first, so a walk cut short (page cap, repeated empty
responses) has collected the *newest* entries and never reached back to the
cursor. Advancing the cursor then would leave the papers in between permanently
invisible. Instead the cursor moves only when the walk provably reached the
cursor, and a truncated run says so loudly. Records already fetched are still
embedded -- they are re-fetched and deduplicated next run, which costs nothing.

Because the cursor only ever moves over ground that was fully covered, and the
boundary comparison is inclusive with a small overlap, the walk can lose a
paper only if arXiv itself omits it from a successful response.
"""

import datetime as dt
import json
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from . import config, ingest, settings, store

API = "https://export.arxiv.org/api/query"
NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}

PAGE_SIZE = 200
# arXiv asks for no more than one request per three seconds.
REQUEST_DELAY = 3.0
# The cursor is rewound by this much before each walk, so entries sharing a
# timestamp with the boundary -- or arriving during the previous walk -- are
# re-examined. Duplicates are free; the upsert discards unchanged papers.
OVERLAP = dt.timedelta(minutes=15)
# Consecutive empty responses tolerated before declaring the walk truncated.
EMPTY_RETRIES = 3
# ~100k entries at PAGE_SIZE 200. These categories see ~45 updates/day, so this
# covers a multi-year absence; a normal run touches one or two pages.
MAX_PAGES = 500
# {category: ISO timestamp}, one cursor per category the index holds. See
# `cursors` for why per category.
CURSORS_KEY = "arxiv_cursors"


# --- Parsing ----------------------------------------------------------------


def _text(entry, path: str, default=None):
    node = entry.find(path, NS)
    return node.text.strip() if node is not None and node.text else default


def _split_id(raw: str):
    """'http://arxiv.org/abs/math/0605123v2' -> ('math/0605123', 'v2')."""
    ident = raw.rsplit("/abs/", 1)[-1]
    head, sep, tail = ident.rpartition("v")
    if sep and tail.isdigit():
        return head, f"v{tail}"
    return ident, None


def _parse_stamp(raw: str):
    """Parse an Atom timestamp into an aware UTC datetime."""
    try:
        return dt.datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(
            dt.timezone.utc
        )
    except (ValueError, AttributeError, TypeError):
        return None


def _parse_entry(entry) -> dict:
    ident, version = _split_id(_text(entry, "atom:id", ""))

    terms = [c.get("term") for c in entry.findall("atom:category", NS)]
    primary = entry.find("arxiv:primary_category", NS)
    if primary is not None and primary.get("term") in terms:
        # Match the snapshot's convention: primary category first.
        terms.remove(primary.get("term"))
        terms.insert(0, primary.get("term"))

    authors = ", ".join(
        a.text.strip()
        for a in entry.findall("atom:author/atom:name", NS)
        if a.text
    )
    updated = _text(entry, "atom:updated", "")

    return {
        "id": ident,
        "version": version,
        "title": _text(entry, "atom:title", ""),
        "abstract": _text(entry, "atom:summary", ""),
        "authors": authors,
        "categories": " ".join(terms),
        # Snapshot stores a bare date; keep the same shape so they sort together.
        "update_date": updated[:10],
        "doi": _text(entry, "arxiv:doi"),
        "journal_ref": _text(entry, "arxiv:journal_ref"),
        "_updated": _parse_stamp(updated),
    }


# --- Fetching ---------------------------------------------------------------


_OPENER = None


def _opener():
    """A urllib opener whose TLS handshake advertises ALPN "http/1.1".

    Python's default SSL context offers no ALPN at all, which makes its
    ClientHello distinctive enough that arXiv's CDN rejects the request with
    406 before it ever reaches the API. Only cached responses get through, so
    the failure looks intermittent: repeat a URL and it may succeed, while the
    fresh page offsets a walk actually needs always fail. Advertising
    "http/1.1" -- and only that, since urllib cannot speak HTTP/2 -- makes the
    handshake ordinary and the 406s stop.
    """
    global _OPENER
    if _OPENER is None:
        context = ssl.create_default_context()
        context.set_alpn_protocols(["http/1.1"])
        _OPENER = urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=context)
        )
    return _OPENER


def _query(categories) -> str:
    return " OR ".join(f"cat:{c}" for c in categories)


def _fetch(categories, start: int, page_size: int = PAGE_SIZE,
           retries: int = 4):
    """One page, newest-first. Returns (entries, total_results).

    Transport errors are retried with backoff. An empty *successful* response is
    returned as-is; the caller decides whether it means end-of-stream.
    """
    url = API + "?" + urllib.parse.urlencode({
        "search_query": _query(categories),
        "sortBy": "lastUpdatedDate",
        "sortOrder": "descending",
        "start": start,
        "max_results": page_size,
    })
    request = urllib.request.Request(url, headers={"User-Agent": "arxiv-index/1.0"})

    last = None
    for attempt in range(retries):
        try:
            with _opener().open(request, timeout=60) as response:
                root = ET.fromstring(response.read())
            node = root.find("opensearch:totalResults", NS)
            total = int(node.text) if node is not None and node.text else None
            return root.findall("atom:entry", NS), total
        except (urllib.error.URLError, ET.ParseError, ValueError, OSError) as exc:
            last = exc
            if attempt < retries - 1:
                time.sleep(REQUEST_DELAY * (2 ** attempt))

    raise RuntimeError(f"arXiv API request failed after {retries} attempts: {last}")


def fetch_since(cursor: dt.datetime, categories, max_pages: int = MAX_PAGES,
                log=print):
    """Walk `categories` newest-first until reaching `cursor`.

    Returns (records, newest_seen, complete). `complete` is True only if the
    walk actually reached the cursor; the caller must not advance the cursor
    otherwise.
    """
    floor = cursor - OVERLAP
    log(f"Querying arXiv for {', '.join(categories)} updated since "
        f"{floor:%Y-%m-%d %H:%M} UTC")

    records, newest = [], None
    consumed = pages = empty_streak = 0
    total = None
    complete = False

    while pages < max_pages:
        entries, page_total = _fetch(categories, consumed)
        if total is None:
            total = page_total

        if not entries:
            # End of the corpus, or a transient blank page? The advertised
            # total distinguishes them.
            if total is not None and consumed >= total:
                complete = True
                break
            empty_streak += 1
            if empty_streak > EMPTY_RETRIES:
                log(f"  arXiv returned {empty_streak} empty pages at offset "
                    f"{consumed}; stopping short.")
                break
            time.sleep(REQUEST_DELAY * empty_streak)
            continue  # retry the same offset; not a new page

        empty_streak = 0
        for entry in entries:
            record = _parse_entry(entry)
            stamp = record.pop("_updated")
            if stamp and (newest is None or stamp > newest):
                newest = stamp
            if stamp and stamp < floor:
                complete = True
                break
            if record["id"] and config.in_scope(record["categories"],
                                                categories):
                records.append(record)

        consumed += len(entries)
        pages += 1
        log(f"  page {pages}: {consumed:,} scanned, {len(records):,} in window")

        if complete:
            break
        if total is not None and consumed >= total:
            complete = True
            break
        time.sleep(REQUEST_DELAY)

    if not complete:
        log(f"  WALK INCOMPLETE after {pages} pages ({consumed:,} entries) -- "
            f"never reached {floor:%Y-%m-%d %H:%M}.\n"
            f"  Cursor will NOT advance, so nothing is skipped. Re-run with "
            f"--max-pages above {max_pages} to finish catching up.")
    return records, newest, complete


# --- Cursor -----------------------------------------------------------------


def day_cursor(date: str) -> dt.datetime:
    """Midnight UTC of a bare YYYY-MM-DD, which is all the snapshot records.
    Starting there re-examines part of that day; the upsert discards what is
    already held."""
    return dt.datetime.strptime(date, "%Y-%m-%d").replace(
        tzinfo=dt.timezone.utc)


def cursors(db) -> dict:
    """{category: how far the index is known complete}, one per category held.

    Per category because categories join the index at different times. One
    added today is backfilled from a snapshot that may be months old, so its
    history runs up to the snapshot's date while the others are current; a
    single cursor would either skip the months in between for the newcomer or
    re-walk them for everyone on every run.

    A category is only given a cursor once its backfill has finished, so the
    keys are also the answer to "which categories does this index hold".
    """
    try:
        parsed = json.loads(store.get_meta(db, CURSORS_KEY) or "{}")
    except ValueError:
        return {}
    out = {}
    for cat, stamp in parsed.items() if isinstance(parsed, dict) else ():
        stamp = _parse_stamp(stamp)
        if stamp:
            out[cat] = stamp
    return out


def set_cursors(db, values: dict) -> None:
    """Set the cursor of each category in `values`, keeping the rest."""
    merged = cursors(db) | values
    store.set_meta(db, CURSORS_KEY, json.dumps(
        {cat: stamp.isoformat(timespec="seconds")
         for cat, stamp in sorted(merged.items())}))


def missing(db) -> list:
    """The reader's categories that this index does not hold yet."""
    held = cursors(db)
    return [c for c in settings.categories() if c not in held]


def update(db, max_pages: int = MAX_PAGES, log=print, progress=None) -> int:
    """Fetch, upsert and embed everything new since the last run.

    `log` takes the narration and `progress` the (done, total) embedding count.
    Both default to the CLI's behaviour; the web UI passes its own so the run
    can be watched from the page that started it.

    Every category the index holds is kept current, including any since
    dropped from the settings or brought in by an import.

    One walk covers them all, back to the oldest cursor, and on success every
    cursor moves to the same point. A newly backfilled category thus makes one
    run walk further than usual, after which it is in step with the rest --
    rather than costing a separate walk, three seconds a page, on every run.
    """
    store.check_model(db)
    for cat in missing(db):
        log(f"{cat} is in your settings but not in this index yet; run "
            f"`build` to backfill it from the snapshot.")
    held = cursors(db)
    if not held:
        log("This index holds no categories yet; run `build` first.")
        return 0
    categories = sorted(held)
    records, newest, complete = fetch_since(min(held.values()), categories,
                                            max_pages, log=log)

    embedded = 0
    if records:
        before = store.count_papers(db)
        pending = store.upsert_papers(db, records)
        added = store.count_papers(db) - before
        # Papers needing no work are those the upsert left with a vector, i.e.
        # fetched minus pending. `added` must NOT be subtracted as well: new
        # papers are themselves part of `pending`, so doing so double-counts
        # them and the figure goes negative.
        log(f"{len(records):,} fetched -> {added:,} new, "
            f"{pending - added:,} revised, "
            f"{len(records) - pending:,} already current "
            f"({pending:,} to embed).")
        embedded = ingest.embed_pending(db, log=log, progress=progress)
    else:
        log("No new papers.")

    # Advance only after the work lands, and only over ground fully covered.
    if complete and newest:
        set_cursors(db, {cat: newest for cat in categories})
    return embedded
