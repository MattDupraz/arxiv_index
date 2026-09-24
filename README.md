# arXiv index

Semantic search over arXiv abstracts in the categories you choose, running
entirely on your own machine. A paper is in scope if **any** of its categories
is one of yours, so cross-listed work counts. With the default math.AC, math.AG
and math.CO that is ~145,000 papers and 747 MB of vectors; vector search takes
~210 ms, ~800 ms with reranking.

Day-to-day use is two commands:

```bash
python3 -m arxiv_index serve      # http://127.0.0.1:8000/
python3 -m arxiv_index update     # weekly top-up, about a minute
```

## First-time setup

The repository is **code only**. A fresh clone can search nothing until you
build the index: one 5.5 GB download and a few hours of embedding.

1. **Ollama with the embedding model:** `ollama pull qwen3-embedding:4b`, or
   [another](#another-embedding-model).
2. **Python 3.11+ with `numpy` and `ollama`.** Everything else is stdlib.
3. **Optionally torch + transformers**, for reranking and GPU search. Without
   them searches run on the CPU, unreranked.
4. **The Kaggle snapshot**, for backfilling only:
   [kaggle.com/datasets/Cornell-University/arxiv](https://www.kaggle.com/datasets/Cornell-University/arxiv).
   Unzip `arxiv-metadata-oai-snapshot.json` into the repo root. arXiv caps how
   deep a result set can be paged, so the snapshot is the only way to get the
   history in; it can be deleted once the build finishes, and fetched again
   if you later add a category.
5. **Choose your categories** (see [Settings](#settings)):

```bash
python3 -m arxiv_index config    # creates the settings file and shows it
```

6. **Build:**

```bash
python3 -m arxiv_index build     # scan the snapshot, then embed
python3 -m arxiv_index status    # where the index is and what is in it
python3 -m arxiv_index serve
```

The scan takes a couple of minutes. Embedding the default three categories'
145k papers takes about three hours on a consumer GPU, and more categories take
proportionally longer. It is interruptible: `build --embed-only` picks up where
it stopped. Budget ~6.3 KB of disk per paper.

Model weights live in `~/.ollama` (~2.5 GB) and `~/.cache/huggingface`
(~570 MB).

## Settings

The index and everything personal live in `~/.arxiv_index/`, or wherever
`$ARXIV_INDEX_DIR` points. The personal part is one file, `config.json`, so
handing someone `papers.db` and `vectors.f16` does not hand them your profile.

```json
{
  "categories": ["math.AC", "math.AG", "math.CO", "math.NT"],
  "snapshot": "~/Downloads/arxiv-metadata-oai-snapshot.json"
}
```

| | |
|---|---|
| `categories` | the arXiv categories you want, by their full names: `math.AG`, `hep-th`, `cs.LG` ([list](https://arxiv.org/category_taxonomy)). Default: math.AC, math.AG, math.CO |
| `snapshot` | where the Kaggle snapshot is. Default: the repo root |
| `embedding` | the Ollama embedding model. Default: `qwen3-embedding:4b`; see [below](#another-embedding-model) |

Relative paths are read from the settings file's own directory. The profile
and the automatic-update setting are stored here too, written by the web UI.
Hand edits are picked up without a restart, except the three keys above, which
`serve` reads when it starts.

### Another embedding model

```json
"embedding": {
  "model": "nomic-embed-text",
  "dim": 768,
  "query_prefix": "search_query: ",
  "document_prefix": "search_document: "
}
```

`dim` is the model's embedding length (`ollama show <model>` gives it). The
prefixes are whatever the model's card says it was trained with, and default
to none. All the measurements in `config.py` were taken with the default model.

An index is tied to the model that built it. `model`, `dim` and
`document_prefix` are recorded in it, and anything else is refused, since those
vectors would not be comparable. To switch models, point `$ARXIV_INDEX_DIR` at a new
directory and `build` there. `query_prefix` is free to change at any time.

An index from before this file existed moves its profile here the first time
it is opened.

### Adding a category

Add it to `categories` and run `build`. That scans the snapshot **for the new
categories only**, leaving papers already held alone, and the next `update`
fills in everything since the snapshot was taken. Until then, `status` and the
web UI list it as not yet in the index. A name that matches no papers in the
snapshot is reported, since it is most likely a typo.

### Removing one, or sharing an index

Removing a category from your settings hides it rather than deleting it. With
no category ticked, searches cover **your** categories. An index holding others
(someone else's, or ones you dropped) keeps them out of your results, but
`update` still keeps every category in the index current, so that whoever runs
it does not leave the others' categories to go stale.

### The two index files are a matched set

`papers.db` is the source of truth: each paper's `row` column names its
slot in `vectors.f16`, which has no identity of its own. **Back them up
together.** If they are separated, the vectors can be rebuilt:

```bash
sqlite3 ~/.arxiv_index/papers.db "UPDATE papers SET row = NULL"
rm ~/.arxiv_index/vectors.f16
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

Two fields describing *you* rather than a search, stored in your
[settings file](#settings):

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
unrankable until you save again. An interest added by editing the settings
file is embedded the first time you rank. The embeddings are cached in
`interest-vectors.json` in the same directory, which is safe to delete.

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

Embedding takes an exclusive lock (`embed.lock` in the index directory), so an `update` firing
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
| `status` | settings file, index location, model, counts, and per category how far it is complete |
| `config` | show your settings file, creating it with the defaults if absent |
| `compact` | reclaim vector slots left behind by re-embedded papers |

`serve` binds to `127.0.0.1` by default. The server exposes the index and,
indirectly, Ollama, so think before changing `--host`.

## Source layout

How the index is built is in `config.py`, with the measurements behind each
choice in the comments; what differs between people is in the settings file. Deeper background — why search is brute-force, how the
reranker was chosen, what was tried and abandoned — is in [NOTES.md](NOTES.md).

| | |
|---|---|
| `config.py` | models, tuning |
| `settings.py` | the per-person settings file: categories, paths, profile |
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
