"""Generate biblatex entries for indexed papers.

Kept separate from the web layer so the CLI can reuse it, and so the fiddly
parts -- year derivation and author splitting -- are testable on their own.
"""

import re

# arXiv identifiers encode the submission month, which is the only reliable
# year available here. `update_date` is when the *metadata* last changed, so
# alg-geom/9202001 (a 1992 paper) carries update_date 2008-02-03; using it
# would misdate a large part of the corpus by well over a decade.
_YYMM = re.compile(r"^(\d{2})(\d{2})")


def arxiv_year(paper_id: str, fallback: str = None):
    """Submission year from the identifier.

    Handles both schemes, which share a leading YYMM once the archive prefix is
    dropped: `math/0605123` -> 2006, `0704.0002` -> 2007, `2401.12345` -> 2024.
    """
    tail = (paper_id or "").split("/")[-1]
    match = _YYMM.match(tail)
    if match:
        yy, mm = int(match.group(1)), int(match.group(2))
        if 1 <= mm <= 12:
            # arXiv opened in August 1991 and the old scheme ran to March 2007,
            # so a two-digit year of 91+ can only be the 1990s.
            return 1900 + yy if yy >= 91 else 2000 + yy
    if fallback and len(fallback) >= 4 and fallback[:4].isdigit():
        return int(fallback[:4])
    return None


def split_authors(raw: str):
    """arXiv stores authors as free text, in two different shapes.

    The snapshot writes "C. Balazs, E. L. Berger, P. M. Nadolsky" while the API
    (and some snapshot rows) write "Ileana Streinu and Louis Theran". Both
    separators appear, sometimes in the same string, so normalise then split.
    """
    if not raw:
        return []
    text = " ".join(raw.split())
    # Case-insensitive: "Kenny, And Colum Watt" occurs in the corpus, and
    # treating that "And" as part of a name yields "... and And Colum Watt",
    # which biber rejects as a malformed name list.
    text = re.sub(r"\s+and\s+", ", ", text, flags=re.IGNORECASE)
    parts = []
    for part in text.split(","):
        # Doubled separators occur verbatim in the corpus ("Xavier Goaoc and
        # and Isaac Mabillard"). The first match consumes the whitespace the
        # second one needed, so a stray "and" survives into the name and biber
        # then reads an empty author. Strip any that are left over.
        part = re.sub(r"^(?:and\s+)+", "", part.strip(), flags=re.IGNORECASE)
        part = re.sub(r"(?:\s+and)+$", "", part, flags=re.IGNORECASE).strip()
        if part and part.lower() not in ("and", "&"):
            parts.append(part)
    return parts


def cite_key(paper_id: str) -> str:
    """Stable, collision-free key. Slashes are not legal in biblatex keys."""
    return "arxiv:" + (paper_id or "").replace("/", "_")


def _balance_braces(text: str) -> str:
    """Drop braces that have no partner.

    arXiv titles are author-supplied LaTeX and a handful are genuinely
    unbalanced -- "${mathrm{Sym}^{d}(X)$" (a mistyped \\mathrm), or a
    "\\title[short]{long}" whose bracket leaked into the metadata. Wrapping such
    a value in {...} yields an entry biber cannot parse, and because the parser
    then skips to the next "@", one bad entry can swallow its neighbour.

    Escaping the stray as \\{ does NOT help: BibTeX's lexer counts braces
    literally while finding the end of a field, so a backslash-escaped brace
    still opens a group as far as the parser is concerned. Hence deletion.
    Braces are counted the same literal way here, to match that behaviour.

    The affected titles are already invalid LaTeX and would not compile as
    written, so removing the stray costs nothing a reader would miss.
    """
    drop, stack = set(), []
    for i, ch in enumerate(text):
        if ch == "{":
            stack.append(i)
        elif ch == "}":
            if stack:
                stack.pop()
            else:
                drop.add(i)
    drop.update(stack)
    if not drop:
        return text
    return "".join(ch for i, ch in enumerate(text) if i not in drop)


def biblatex(paper: dict) -> str:
    """A biblatex @online entry for one paper.

    Title and abstract are stored exactly as arXiv holds them, which is already
    LaTeX source -- so they are emitted verbatim rather than escaped. Escaping
    would turn a title's `$K_4$` into literal dollar signs.
    """
    paper_id = paper.get("id", "")
    categories = (paper.get("categories") or "").split()
    year = arxiv_year(paper_id, paper.get("update_date"))
    authors = split_authors(paper.get("authors"))

    fields = [("title", " ".join((paper.get("title") or "").split()))]
    if authors:
        fields.append(("author", " and ".join(authors)))
    if year:
        fields.append(("year", str(year)))
    fields += [
        ("eprint", paper_id),
        ("eprinttype", "arxiv"),
    ]
    if categories:
        fields.append(("eprintclass", categories[0]))
    if paper.get("doi"):
        fields.append(("doi", paper["doi"]))
    if paper.get("journal_ref"):
        # arXiv's journal-ref is a free-text blob ("J. Alg. Geom. 1 (1992)
        # 449--530"); it does not decompose reliably into journaltitle/volume/
        # pages, so it goes in note rather than being guessed at.
        fields.append(("note", " ".join(paper["journal_ref"].split())))
    fields.append(("url", f"https://arxiv.org/abs/{paper_id}"))

    width = max(len(name) for name, _ in fields)
    body = ",\n".join(
        f"  {name.ljust(width)} = {{{_balance_braces(value)}}}"
        for name, value in fields
    )
    return f"@online{{{cite_key(paper_id)},\n{body},\n}}"
