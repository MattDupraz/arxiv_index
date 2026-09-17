# arXiv index — math.AC / math.AG / math.CO

Semantic search over arXiv abstracts in commutative algebra, algebraic geometry
and combinatorics, running entirely on your own machine. A paper is in scope if
**any** of its categories is one of the three, so cross-listed work counts —
~145,000 papers, 747 MB of vectors. Vector search takes ~210 ms, ~800 ms with
reranking.

Day-to-day use is two commands:

```bash
python3 -m arxiv_index serve      # http://127.0.0.1:8000/
python3 -m arxiv_index update     # weekly top-up, about a minute
```

## First-time setup

The repository is **code only**. A fresh clone can search nothing until you
build the index: one 5.5 GB download and a few hours of embedding.

1. **Ollama with the embedding model:** `ollama pull qwen3-embedding:4b`.
2. **Python 3.11+ with `numpy` and `ollama`.** Everything else is stdlib.
3. **Optionally torch + transformers**, for reranking and GPU search. Without
   them searches run on the CPU, unreranked.
4. **The Kaggle snapshot**, for the initial backfill only:
   [kaggle.com/datasets/Cornell-University/arxiv](https://www.kaggle.com/datasets/Cornell-University/arxiv).
   Unzip `arxiv-metadata-oai-snapshot.json` into the repo root. arXiv caps how
   deep a result set can be paged, so the snapshot is the only way to get the
   history in; it can be deleted once the build finishes.
5. **Build:**

```bash
python3 -m arxiv_index build     # scan the snapshot, then embed
python3 -m arxiv_index status    # where the index is and what is in it
python3 -m arxiv_index serve
```

The scan takes a couple of minutes; embedding 145k papers takes about three
hours on a consumer GPU. It is interruptible — `build --embed-only` picks up
where it stopped. Budget ~910 MB for the finished index.

Paths come from `arxiv_index/config.py` (`INDEX_DIR`, `SNAPSHOT`), so pointing
`INDEX_DIR` at an external disk moves the whole index. Model weights live in
`~/.ollama` (~2.5 GB) and `~/.cache/huggingface` (~570 MB).

### The two index files are a matched set

`index/papers.db` is the source of truth: each paper's `row` column names its
slot in `index/vectors.f16`, which has no identity of its own. **Back them up
together.** If they are separated, the vectors can be rebuilt:

```bash
sqlite3 index/papers.db "UPDATE papers SET row = NULL"
rm index/vectors.f16
python3 -m arxiv_index build --embed-only
```

The reverse does not work: `vectors.f16` alone is anonymous numbers.

## Using it

The web UI has a search box, an author filter, category checkboxes, a date
range, expandable abstracts, a **BibLaTeX** button and **Similar papers** on
every result. LaTeX is rendered with a vendored KaTeX, so it works offline. The
cog opens settings: your profile, a **Fetch new papers** button, and when to run
that automatically.

```bash
python3 -m arxiv_index search "toric degenerations of flag varieties"
python3 -m arxiv_index search "chromatic polynomial" -k 20 --category math.CO
python3 -m arxiv_index search "invariant theory of finite groups" --author Noether
python3 -m arxiv_index search --author "Hardy, Littlewood"  # no query needed
python3 -m arxiv_index similar 0704.0002
```

**Reranking** rescores the top 50 hits with a cross-encoder, which is markedly
better ordering — known-item recall@1 goes from 0.50 to 0.86 — at about a second
per search. It needs torch and a GPU; the **Rerank** checkbox appears only when
they are installed. Author-only listings skip it, having no query to be relevant
to.

**Scores are hidden by default** in both interfaces. The **Scores** checkbox and
`--scores` reveal them.

### Your profile

Two fields describing *you* rather than a search, stored in the index's `meta`
table so they travel with `papers.db`:

| | |
|---|---|
| **Followed authors** | one name per line — people worth reading whatever they write |
| **Research interests** | one short description per thing you work on, each with a weight |

They drive two buttons, both over the **Since**/**Until** range, which is used
exactly as the form has it. Leave both empty to ask the whole index.

**Followed authors** lists everything those people posted in the range,
newest-first. This is a *union* — the opposite of the author box, where several
names mean papers written **together**.

**Rank by my interests** orders the same range by closeness to what you work on.
Embedding only, no reranking, and only embedded papers can be ranked.

Write one interest per project rather than one paragraph: each is embedded
**separately**, so they stay distinct instead of averaging into a point that is
squarely none of them. Each carries a weight from `0` to `2` (default `1`);
`0` parks an entry without deleting it.

A paper is scored against every interest, the scores are multiplied by their
weights and sorted, and each after the first counts less — the first in full,
the second times *b*, the third times *b²*. The **Reward for matching several**
slider is *b*:

| *b* | what it does |
|---|---|
| `0` | only the best match counts. A paper squarely on one project wins outright |
| `0.35` | the default. The best match dominates, but a genuine second match still lifts a paper |
| `1` | every match counts in full, which ranks identically to averaging your interests into one vector |

Descriptions are embedded on save, so changing a weight or the slider embeds
nothing. If Ollama is unreachable the text is stored anyway and flagged as
unrankable until you save again.

### Searching by author

Works alone — newest-first for that author — or alongside a query, which ranks
their work by relevance to it.

**Several names, comma-separated, mean papers written *together*.** `Hardy,
Littlewood` returns their 8 joint papers rather than the 197 written by one or
the other; `;` and `and` also separate. A single name written surname-first
works too, and does better than `Godfrey Hardy`, since the terms match
independently and so also find "Godfrey H. Hardy".

**Names match regardless of case and accents**, since arXiv stores many author
fields as LaTeX (`Poincar\'e`, `Erd\H{o}s`). Affiliations riding along in the
field are stripped.

One asymmetry: **with** a query only embedded papers come back, since ranking
needs a vector. **Without** one, the listing is pure metadata and covers every
paper in the database.

## Keeping it current

```bash
python3 -m arxiv_index update
```

or the **Fetch new papers** button, which runs exactly this. It walks the arXiv
API back from the newest paper to the stored cursor, embeds what is new, and
advances the cursor. Papers whose title or abstract changed are re-embedded. The
cursor advances *only* when a walk provably reached it: a run cut short says
`WALK INCOMPLETE`, keeps what it fetched and leaves the cursor alone, so the
failure mode is wasted work rather than a gap.

Embedding takes an exclusive lock (`index/embed.lock`), so an `update` firing
during a long `build` exits cleanly. Searching during a build is fine.

From the web UI the run belongs to the server rather than the tab, so closing
the page does not stop it and reopening picks it back up. One runs at a time.

### Automatic updates

Under the cog, **Automatic updates** presses the button on a clock for as long
as `serve` is up, so an index does not go stale behind a server left running:

| | |
|---|---|
| **Off** | the default |
| **Every N hours** | measured from the end of the last run, 1 to 168 |
| **Daily at HH:MM** | a wall-clock time, in the server machine's **local** time |

Local rather than UTC on purpose: 07:00 means 07:00 where you are, and arXiv's
announcements go out at a fixed New York time.

**A missed run is caught up, not skipped.** A daily 07:00 run on a machine
asleep until 09:00 fires at 09:00. Switching the setting on works the same way:
if a slot has passed and nothing has run since, the first run happens within the
minute. Manual runs count, so **Fetch new papers** resets the clock.

Changes take effect within 30 seconds, no restart needed. For updates without a
server running, use cron:

```cron
0 7 * * 1  cd /path/to/arXiv_index && python3 -m arxiv_index update >> update.log 2>&1
```

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

## Source layout

Everything adjustable is in `config.py`, with the measurements behind each
choice in the comments. Deeper background — why search is brute-force, how the
reranker was chosen, what was tried and abandoned — is in [NOTES.md](NOTES.md).

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
| `profile.py` | followed authors + weighted interests, each with its own embedding |
| `schedule.py` | when the server tops itself up |
| `web.py` | local web UI (stdlib `http.server`) |
| `__main__.py` | CLI |
