"""Normalising arXiv's LaTeX author strings for search.

13.8% of records write author names as LaTeX: "J\\'er\\'emy Blanc",
"Daniela K\\\"uhn", "Mikkel {\\O}bro". Matching a typed name against that raw
text is hopeless -- LIKE '%Kollar%' finds a single paper out of the many by
Kollár -- so names are reduced to a plain lowercase ASCII-ish form first.

Two steps, deliberately separate:

  latex_to_unicode   "Koll\\'ar"  -> "Kollár"     (what a human would write)
  fold               "Kollár"     -> "kollar"     (what we compare)

Folding both sides means a search for "kollar", "Kollar" or "Kollár" all hit
the same papers, which is what someone typing a name actually expects.

The web UI carries an equivalent routine in JavaScript for *display*; this one
exists for *matching*, and the two are intentionally allowed to differ (display
keeps the accents, matching removes them).
"""

import re
import unicodedata

# TeX accent command -> Unicode combining mark, applied to the following letter
# and then normalised, which composes far more characters than a table would.
_COMBINING = {
    '"': "\u0308", "'": "\u0301", "`": "\u0300", "^": "\u0302", "~": "\u0303",
    "=": "\u0304", ".": "\u0307", "u": "\u0306", "v": "\u030C", "H": "\u030B",
    "c": "\u0327", "k": "\u0328", "r": "\u030A", "d": "\u0323", "b": "\u0331",
}
_LETTERS = {
    "ss": "ß", "ae": "æ", "AE": "Æ", "oe": "œ", "OE": "Œ", "aa": "å", "AA": "Å",
    "o": "ø", "O": "Ø", "l": "ł", "L": "Ł", "i": "ı", "j": "ȷ",
}

# Special letters first: TeX writes an accented i as \'\i, so \i must already
# be a letter by the time accents are applied.
_LETTER_RE = re.compile(r"\\(ss|ae|AE|oe|OE|aa|AA|[oOlLij])(\{\}|[ \t]+|\b)")
_ACCENT_RE = re.compile(
    r"\\([\"'`^~=.]|[uvHckrdb])\s*\{([A-Za-zıȷ])\}"
    r"|\\([\"'`^~=.])\s*([A-Za-zıȷ])"
)
# Affiliations ride along in some records: "Krattenthaler (Universit\"at Wien)".
# Dropping them keeps a search for a place name from matching every author there.
_PARENS = re.compile(r"\([^()]*\)")

# Characters with no decomposition, so NFKD alone will not reduce them.
_FOLD = str.maketrans({
    "ø": "o", "Ø": "O", "ł": "l", "Ł": "L", "đ": "d", "Đ": "D", "ħ": "h",
    "ı": "i", "ȷ": "j", "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE",
    "ß": "ss", "þ": "th", "Þ": "Th", "ð": "d", "Ð": "D",
})


def _accent(match):
    symbol, braced, bare_symbol, bare = match.groups()
    mark = _COMBINING[symbol if symbol is not None else bare_symbol]
    base = braced if braced is not None else bare
    # An accented dotless i/j is simply the accented i/j.
    base = {"ı": "i", "ȷ": "j"}.get(base, base)
    return unicodedata.normalize("NFC", base + mark)


def latex_to_unicode(text: str) -> str:
    """Turn LaTeX-escaped names into ordinary Unicode."""
    if not text:
        return ""
    text = _LETTER_RE.sub(lambda m: _LETTERS.get(m.group(1), m.group(0)), text)
    text = _ACCENT_RE.sub(_accent, text)
    # Braces are grouping, never content: "{\O}bro" has already become "{Ø}bro".
    text = text.replace("{", "").replace("}", "")
    return " ".join(text.split())


_TERM_SPLIT = re.compile(r"[,;]|\s+and\s+", re.IGNORECASE)


def fold_terms(query: str):
    """Split an author query into folded terms that must ALL match.

    "Larson, Payne" means the papers those two wrote together, so the terms are
    combined with AND -- the reading that makes a multi-name query useful, and
    the one that narrows rather than widens.

    Writing one name the other way round, "Larson, Hannah", also works, and in
    fact does better than "Hannah Larson": the terms are matched independently,
    so it still finds "Hannah K. Larson", whom the contiguous form misses.

    The trade-off is that terms need not belong to the same person, so
    "Larson, Hannah" would also admit a paper by "Hannah Smith and Bob Larson".
    That is the price of letting one comma mean both "and this other author"
    and "surname first", and the multi-author reading is the useful one.
    """
    if not query:
        return []
    return [term for term in (fold(part) for part in _TERM_SPLIT.split(query))
            if term]


def matches_terms(folded_authors: str, terms) -> bool:
    return all(term in folded_authors for term in terms)


def fold(text: str) -> str:
    """Reduce a name to the lowercase, accent-free form used for comparison."""
    if not text:
        return ""
    text = latex_to_unicode(text)
    text = _PARENS.sub(" ", text)
    text = text.translate(_FOLD)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(text.casefold().split())
