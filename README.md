# arXiv index — math.AC / math.AG / math.CO

Semantic search over arXiv abstracts in commutative algebra, algebraic geometry
and combinatorics. Runs entirely on your own machine.

A paper is in scope if **any** of its categories is one of the three, so
cross-listed work (e.g. a `cs.CG math.CO` paper) counts. That is ~145,000 papers.

Once the index exists, day-to-day use is two commands:

```bash
python3 -m arxiv_index serve      # http://127.0.0.1:8000/
python3 -m arxiv_index update     # weekly top-up, about a minute
```

| | |
|---|---|
| vector search | ~210 ms |
| + reranking the top 50 | ~800 ms |
| index | 145,853 papers, 747 MB of vectors |

## Where everything lives

This repository is **code only**. No data is committed, so a fresh clone can
search nothing until you build the index — one 5.5 GB download and a few hours
of embedding, both described in [First-time setup](#first-time-setup).

```
arXiv_index/
├── arxiv_index/                     the package — everything that is in git
│   └── static/                      vendored KaTeX (no CDN; the UI works offline)
├── arxiv-metadata-oai-snapshot.json Kaggle dump, 5.5 GB — initial backfill only
└── index/                           the database; created by the first build
    ├── papers.db                    SQLite, ~160 MB, one row per paper
    ├── vectors.f16                  flat float16 vectors, ~750 MB
    └── embed.lock                   empty file, advisory lock
```

Both paths come from `arxiv_index/config.py` — `INDEX_DIR` and `SNAPSHOT`,
resolved relative to the repo root. Nothing else hardcodes a location, so
pointing `INDEX_DIR` at an external disk moves the whole index. `python3 -m
arxiv_index status` prints the directory in use, along with what is in it.

Model weights are not here either. Ollama keeps the embedding model in its own
store (`~/.ollama`, ~2.5 GB), and the reranker is downloaded to
`~/.cache/huggingface` (~570 MB) the first time reranking is switched on.

### The two index files are a matched set

`papers.db` is the source of truth: each paper's `row` column names its slot in
`vectors.f16`, and the vector file has no identity of its own. **Back them up
together, and copy them together.** If they do get separated, the vectors can be
rebuilt from the metadata — one statement plus the embedding time:

```bash
sqlite3 index/papers.db "UPDATE papers SET row = NULL"
rm index/vectors.f16
python3 -m arxiv_index build --embed-only
```

The reverse does not work: `vectors.f16` on its own is anonymous numbers.

`embed.lock` is created on demand and holds no state — deleting it while nothing
is embedding is harmless.

## First-time setup

**1. Ollama, with the embedding model.**

```bash
ollama pull qwen3-embedding:4b
```

**2. Python 3.11+ with `numpy` and `ollama`.** The arXiv API client, the web
server and the citation generator use only the standard library.

**3. Optionally torch + transformers**, for reranking and GPU search — a CUDA or
ROCm build, matching your hardware. Everything else works without them; searches
simply run on the CPU, unreranked.

**4. The Kaggle snapshot**, for the initial backfill only:
[kaggle.com/datasets/Cornell-University/arxiv](https://www.kaggle.com/datasets/Cornell-University/arxiv).
Unzip `arxiv-metadata-oai-snapshot.json` into the repo root (or set
`config.SNAPSHOT`). It is 5.5 GB and can be deleted once the build finishes;
after that the index keeps itself current from the arXiv API. There is no
API-only backfill path — arXiv caps how deep a result set can be paged, so the
snapshot is how the history gets in.

**5. Build.**

```bash
python3 -m arxiv_index build     # scan the snapshot, then embed
python3 -m arxiv_index status    # where the index is and what is in it
python3 -m arxiv_index serve
```

The scan takes a couple of minutes; embedding 145k papers takes about three
hours on a consumer GPU. It is interruptible — `build --embed-only` picks up
exactly where it stopped, skipping the scan. Budget ~910 MB for the finished
index, plus the 5.5 GB snapshot while it exists.

## Using it

The web UI has a search box, an author filter, category checkboxes, a since-date
filter, expandable abstracts, links to the abstract and PDF, a **BibLaTeX**
button, and **Similar papers** on every result. LaTeX in titles and abstracts is
rendered with a vendored copy of KaTeX, so the whole thing works offline.

```bash
python3 -m arxiv_index search "toric degenerations of flag varieties"
python3 -m arxiv_index search "chromatic polynomial" -k 20 --category math.CO
python3 -m arxiv_index search "singularities of pairs" --author Kollar
python3 -m arxiv_index search --author "Larson, Payne"     # no query needed
python3 -m arxiv_index similar 0704.0002
```

`--rerank` enables reranking from the CLI, `--scores` shows relevance numbers,
and `--json` prints full records.

**Reranking** rescores the top 50 hits with a cross-encoder that reads the query
and abstract together. It is markedly better ordering — known-item recall@1 goes
from 0.50 to 0.86 — at about a second per search, and it needs torch and a GPU.
The **Rerank** checkbox controls it in the web UI. It is skipped automatically
for author-only listings, where there is no query to be relevant to, and if the
model is missing or fails to load the search falls back to vector order and says
so rather than failing.

**Scores are hidden by default** in both interfaces — they are diagnostics, not
reading material. The **Scores** checkbox reveals them in the web UI and
`--scores` does the same on the CLI; a reranked hit then shows both its relevance
logit and the cosine it started from, since the two are on unrelated scales.

**Similar papers** works the same way, with the source paper's own text standing
in for the query. Reranking it costs about a second against 8 ms for the plain
vector lookup, so it is worth leaving off when skimming.

### Searching by author

Works alone — leave the query empty for that author's papers newest-first — or
alongside a query, which then ranks their work by relevance to it.

**Several names, comma-separated, mean papers written *together*.** `Larson,
Payne` returns the 8 Larson–Payne collaborations, not the 197 papers either
wrote. `;` and `and` also separate. A single name written surname-first,
`Larson, Hannah`, works too and does better than `Hannah Larson`: the terms match
independently, so it also finds "Hannah K. Larson". The trade-off is that the
terms need not belong to one person, so it would admit "Hannah Smith and Bob
Larson" — the price of one comma meaning both things.

**Names match regardless of case and accents**, since arXiv stores many author
fields as LaTeX (`J\'anos Koll\'ar`, `Mikkel {\O}bro`). Searching `Kollar`
returns 166 papers; a literal substring match would return the 1 that happens to
be spelled without the accents. Affiliations riding along in the field are
stripped, so a place name does not match everyone who works there.

One asymmetry worth knowing: **with** a query, only embedded papers can come
back, because ranking needs a vector. **Without** one, the listing is pure
metadata and covers every paper in the database, embedded or not.

## Keeping it current

```bash
python3 -m arxiv_index update
```

Walks the arXiv API back from the newest paper until it reaches the stored
cursor, embeds what is new, advances the cursor. Papers whose title or abstract
changed are re-embedded; papers that merely gained a DOI are not. The Kaggle
snapshot is not involved, and can be deleted after the initial build.

```cron
0 7 * * 1  cd /path/to/arXiv_index && python3 -m arxiv_index update >> update.log 2>&1
```

**Can it miss a paper?** The cursor advances *only* when a walk provably reached
it. A run cut short says `WALK INCOMPLETE`, leaves the cursor alone, and keeps
what it fetched, so the failure mode is wasted work rather than a gap. Re-run
with a larger `--max-pages`.

Embedding runs take an exclusive lock (`index/embed.lock`), so a cron `update`
firing during a long `build` exits cleanly instead of double-embedding.
Searching during a build is fine.

## Commands

`python3 -m arxiv_index <command>`; every command takes `--help`.

| | |
|---|---|
| `build` | backfill from the snapshot, then embed. `--embed-only` skips the scan |
| `update` | fetch and embed what is new from the arXiv API |
| `search` | semantic search; `--author`, `--category`, `--since`, `--rerank`, `--scores`, `--full`, `--json` |
| `similar` | neighbours of a given arXiv id |
| `serve` | the web UI; `--port`, `--host`, `--no-browser` |
| `status` | index location, model, counts, cursor |
| `compact` | reclaim vector slots left behind by re-embedded papers |

`serve` binds to `127.0.0.1` by default. The server exposes the index and,
indirectly, Ollama, so think before changing `--host`.

## Tuning

Everything adjustable is in `arxiv_index/config.py`, with the measurements
behind each choice in the comments — the models, the shortlist size, whether
search runs on the GPU. Changing `CATEGORIES` and re-running `build` adds
categories without re-embedding what you already have, and changing the
embedding model makes the index refuse to load rather than silently mixing
incomparable vectors.

Deeper background — why search is brute-force, how the reranker was chosen,
what the server holds in memory, and what was tried and abandoned — is in
[NOTES.md](NOTES.md).

## Source layout

| | |
|---|---|
| `config.py` | scope, models, paths, tuning |
| `store.py` | SQLite schema + append-only vector file |
| `embedder.py` | Ollama embedding calls with retry |
| `ingest.py` | snapshot scan + the resumable embedding loop |
| `update.py` | incremental fetch from the arXiv API |
| `search.py` | exact cosine search, author filtering |
| `rerank.py` | cross-encoder reranking of the shortlist |
| `textnorm.py` | LaTeX author names, folded for matching |
| `cite.py` | biblatex entries |
| `web.py` | local web UI (stdlib `http.server`) |
| `__main__.py` | CLI |
