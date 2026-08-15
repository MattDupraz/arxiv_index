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
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from . import config, store

API = "http://export.arxiv.org/api/query"
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
CURSOR_KEY = "arxiv_cursor"


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


def _query() -> str:
    return " OR ".join(f"cat:{c}" for c in config.CATEGORIES)


def _fetch(start: int, page_size: int = PAGE_SIZE, retries: int = 4):
    """One page, newest-first. Returns (entries, total_results).

    Transport errors are retried with backoff. An empty *successful* response is
    returned as-is; the caller decides whether it means end-of-stream.
    """
    url = API + "?" + urllib.parse.urlencode({
        "search_query": _query(),
        "sortBy": "lastUpdatedDate",
        "sortOrder": "descending",
        "start": start,
        "max_results": page_size,
    })
    request = urllib.request.Request(url, headers={"User-Agent": "arxiv-index/1.0"})

    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                root = ET.fromstring(response.read())
            node = root.find("opensearch:totalResults", NS)
            total = int(node.text) if node is not None and node.text else None
            return root.findall("atom:entry", NS), total
        except (urllib.error.URLError, ET.ParseError, ValueError, OSError) as exc:
            last = exc
            if attempt < retries - 1:
                time.sleep(REQUEST_DELAY * (2 ** attempt))

    raise RuntimeError(f"arXiv API request failed after {retries} attempts: {last}")


def fetch_since(cursor: dt.datetime, max_pages: int = MAX_PAGES):
    """Walk newest-first until reaching `cursor`.

    Returns (records, newest_seen, complete). `complete` is True only if the
    walk actually reached the cursor; the caller must not advance the cursor
    otherwise.
    """
    floor = cursor - OVERLAP
    print(f"Querying arXiv for {', '.join(config.CATEGORIES)} updated since "
          f"{floor:%Y-%m-%d %H:%M} UTC")

    records, newest = [], None
    consumed = pages = empty_streak = 0
    total = None
    complete = False

    while pages < max_pages:
        entries, page_total = _fetch(consumed)
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
                print(f"  arXiv returned {empty_streak} empty pages at offset "
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
            if record["id"] and config.in_scope(record["categories"]):
                records.append(record)

        consumed += len(entries)
        pages += 1
        print(f"  page {pages}: {consumed:,} scanned, {len(records):,} in window",
              flush=True)

        if complete:
            break
        if total is not None and consumed >= total:
            complete = True
            break
        time.sleep(REQUEST_DELAY)

    if not complete:
        print(f"  WALK INCOMPLETE after {pages} pages ({consumed:,} entries) -- "
              f"never reached {floor:%Y-%m-%d %H:%M}.\n"
              f"  Cursor will NOT advance, so nothing is skipped. Re-run with "
              f"--max-pages above {max_pages} to finish catching up.")
    return records, newest, complete


# --- Cursor -----------------------------------------------------------------


def default_cursor(db) -> dt.datetime:
    """Where to resume from: the stored cursor, else the newest paper we hold."""
    stored = store.get_meta(db, CURSOR_KEY)
    if stored:
        parsed = _parse_stamp(stored)
        if parsed:
            return parsed

    row = db.execute("SELECT MAX(update_date) AS d FROM papers").fetchone()
    if row and row["d"]:
        # Snapshot dates are bare days; start at midnight UTC of that day and
        # let the upsert discard whatever we already have.
        return dt.datetime.strptime(row["d"], "%Y-%m-%d").replace(
            tzinfo=dt.timezone.utc
        )
    return dt.datetime(1991, 1, 1, tzinfo=dt.timezone.utc)


def update(db, max_pages: int = MAX_PAGES) -> int:
    """Fetch, upsert and embed everything new since the last run."""
    store.check_model(db)
    cursor = default_cursor(db)
    records, newest, complete = fetch_since(cursor, max_pages)

    embedded = 0
    if records:
        before = store.count_papers(db)
        pending = store.upsert_papers(db, records)
        added = store.count_papers(db) - before
        # Papers needing no work are those the upsert left with a vector, i.e.
        # fetched minus pending. `added` must NOT be subtracted as well: new
        # papers are themselves part of `pending`, so doing so double-counts
        # them and the figure goes negative.
        print(f"{len(records):,} fetched -> {added:,} new, "
              f"{pending - added:,} revised, "
              f"{len(records) - pending:,} already current "
              f"({pending:,} to embed).")

        from . import ingest

        embedded = ingest.embed_pending(db)
    else:
        print("No new papers.")

    # Advance only after the work lands, and only over ground fully covered.
    if complete and newest:
        store.set_meta(db, CURSOR_KEY, newest.isoformat(timespec="seconds"))
    return embedded
