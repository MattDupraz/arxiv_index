# arXiv index

A personal semantic search engine for arXiv, running entirely on your machine.

- **Search by meaning:** describe what you want in your own words and get the
  closest papers, whether or not they use the same terms.
- **Rank new papers by your interests**, so you see what matters to your work.
- **Follow authors** and never miss one of their papers.

![The web UI: a search for "toric degenerations of flag varieties", with the first result's abstract open](docs/screenshot.png)

*Thank you to arXiv for use of its open access interoperability. `arxiv_index` was not reviewed or approved by, nor does it necessarily express or reflect the policies or opinions of, arXiv.*

## Quick start

You need Python 3.11+ and [Ollama](https://ollama.com). The repository is code
only: the first setup takes one large download and a few hours of embedding
(or a few minutes with a [ready-made index](#2-get-the-papers)). After that,
staying current takes about a minute a week.

### 1. Install

```bash
ollama pull qwen3-embedding:4b                # the embedding model, ~2.5 GB
git clone https://github.com/MattDupraz/arxiv_index.git
pip install ./arxiv_index                     # gives the `arxiv_index` command
```

Ollama must be **running** whenever you search or fetch papers. On Linux it
runs as a service (`sudo systemctl start ollama`); on macOS and Windows, start
the Ollama app; elsewhere, run `ollama serve` in its own terminal. `ollama list`
tells you whether it is up. Embedding uses the GPU if Ollama can; nothing else
needs one.

<details>
<summary>Running without installing</summary>

```bash
pip install numpy ollama
cd arxiv_index
python3 -m arxiv_index --help
```

Every `arxiv_index …` command in this README then becomes
`python3 -m arxiv_index …`, run from the repository directory. For development,
`pip install -e ./arxiv_index` keeps edits live.
</details>

### 2. Get the papers

Pick one:

- **A ready-made index** (skips the hours of embedding). Mine covers `math.AC`,
  `math.AG` and `math.CO`:
  [download](https://app.filen.io/#/d/27a9dfe5-28fe-469f-80f6-d0e8cf6e69a8%232d66344579636d4a74585a356c4e6e44626956464474746f4f5a6d6b6943534c)
  *(last updated 2026-09-24)*. Any [export](#exporting-and-importing-an-index)
  from another instance works too.
- **arXiv's metadata snapshot**, to build your own index for any categories:
  [kaggle.com/datasets/Cornell-University/arxiv](https://www.kaggle.com/datasets/Cornell-University/arxiv)
  (free account). Unzip it to get `arxiv-metadata-oai-snapshot.json` (~5.5 GB).
  It is only needed for setup and for adding categories later.

### 3. Set up and run

```bash
arxiv_index serve     # opens http://127.0.0.1:8000/
```

With an empty index, the page is a setup wizard: either import an export, or
build from the snapshot by choosing [categories](https://arxiv.org/category_taxonomy)
(default: math.AC, math.AG, math.CO) and an embedding model. Keep the tab open
while the file is read (a minute or two). Embedding then runs on the server,
about **three hours** for the default categories on a consumer GPU; you can
search while it works.

When done, open the **settings (cog, top right)** and press **Fetch new
papers** to catch up to today. That panel also holds your
[profile](#your-profile) and [automatic updates](#automatic-updates).

<details>
<summary>The same in a terminal</summary>

```bash
# from the snapshot (asks for categories and model):
arxiv_index build ~/Downloads/arxiv-metadata-oai-snapshot.json
# or from an export (--settings also takes its profile):
arxiv_index import arxiv_index.tar

arxiv_index update    # catch up to today
arxiv_index status    # what is in the index, and up to when
arxiv_index serve
```

`build` can be interrupted at any time; `build --embed-only` resumes it.
`build --scan-only` imports the papers without embedding them, to embed later
(`build --embed-only`, the next `update`, or **Embed them now** in the web UI).
Until embedded, papers can be listed by author but not found by search.
</details>

## Using it

In the web UI: a search box, author filter, category checkboxes, date range,
expandable abstracts, **BibLaTeX** export and **Similar papers** on every
result. LaTeX renders offline. Scores are hidden unless you tick **Scores**.

The same from the terminal:

```bash
arxiv_index search "toric degenerations of flag varieties"
arxiv_index search "chromatic polynomial" -k 20 --category math.CO
arxiv_index search "invariant theory of finite groups" --author Noether
arxiv_index search --author "Hardy, Littlewood"   # no query needed
arxiv_index similar 0704.0002
```

**Author search:** several comma-separated names mean papers written
*together* (`Hardy, Littlewood` gives their 8 joint papers, not all 197). Case
and accents are ignored. See [details](#author-matching).

### Your profile

Under the cog you can fill in:

- **Followed authors**, one per line. The **Followed authors** button lists
  everything they posted in the chosen date range, newest first.
- **Research interests**, one short description per project, each with a
  weight from 0 to 2. **Rank by my interests** orders the date range by
  closeness to your work.

Write one interest per topic rather than one paragraph, since each is matched
separately. The **Reward for matching several** slider controls how much a
paper benefits from matching more than one interest (see
[how ranking works](#how-interest-ranking-works)).

### Keeping it current

Press **Fetch new papers** in the web UI, or run `arxiv_index update`. To do it
automatically while `serve` runs, set **Automatic updates** under the cog
(every N hours, or daily at a set local time; missed runs are caught up). Without
a server, use cron:

```cron
0 7 * * 1  arxiv_index update >> ~/.arxiv_index/update.log 2>&1
```

## Commands

Every command takes `--help`.

| | |
|---|---|
| `build [SNAPSHOT]` | backfill from the snapshot, then embed. `--scan-only` stops before embedding, `--embed-only` skips the scan |
| `update` | fetch and embed what is new from the arXiv API |
| `search` | semantic search; `--author`, `--category`, `--since`, `--scores`, `--full`, `--json` |
| `similar` | neighbours of a given arXiv id |
| `serve` | the web UI; `--port`, `--host`, `--no-browser` |
| `status` | settings file, index location, model, counts, and how far each category is complete |
| `config` | show your settings file, creating it with the defaults if absent |
| `compact` | reclaim vector slots left behind by re-embedded papers |
| `export FILE` | write the index, embeddings included, to one file; `--settings` adds your settings |
| `import FILE` | install an exported index; `--merge` adds it to the one here, `--replace` overwrites it, `--settings` takes up its settings |

`serve` binds to `127.0.0.1` by default. The server exposes the index and,
indirectly, Ollama, so think before changing `--host`.

## Troubleshooting

- **A search fails with "embedding failed"**: Ollama is not running. Listing by
  author or date, followed authors, similar papers and BibLaTeX still work
  without it.
- **`update` says `WALK INCOMPLETE`**: the fetch was cut short. What was
  fetched is kept and the next run fills the gap.
- **Setup page or Import/Export missing**: they are only shown when the page is
  opened on the machine running `serve`.

---

# Reference

## Settings

Everything lives in `~/.arxiv_index/`, or wherever `$ARXIV_INDEX_DIR` points.
The index is `papers.db` plus `vectors.f16` (about 6.3 KB per paper); your
personal settings are in `config.json`, so sharing the index does not share
your profile.

```json
{
  "categories": ["math.AC", "math.AG", "math.CO", "math.NT"]
}
```

| | |
|---|---|
| `categories` | arXiv categories by full name: `math.AG`, `hep-th`, `cs.LG` ([list](https://arxiv.org/category_taxonomy)). Default: math.AC, math.AG, math.CO |
| `embedding` | the Ollama embedding model. Default: `qwen3-embedding:4b`; see [below](#another-embedding-model) |

The profile and automatic-update setting are stored here too, written by the
web UI. Hand edits are picked up without a restart, except these two keys,
which `serve` reads once (categories changed in the web UI apply at once).

### Another embedding model

The setup page offers every embedding model installed in Ollama. By hand:

```json
"embedding": {
  "model": "nomic-embed-text",
  "dim": 768,
  "query_prefix": "search_query: ",
  "document_prefix": "search_document: "
}
```

`dim` is the embedding length (`ollama show <model>`). The prefixes are
whatever the model card says it was trained with, and default to none. All
measurements in `config.py` used the default model.

An index is tied to its model: `model`, `dim` and `document_prefix` are
recorded in it and anything else is refused. To switch models, point
`$ARXIV_INDEX_DIR` at a new directory and `build` there. `query_prefix` can
change at any time.

### Adding and removing categories

**To add**, tick it under **Categories** in the settings and save, then use
**Import from the arXiv snapshot** to fill in its history. In a terminal, add
it to `categories` and run `build` with the snapshot's path. Only the new
categories are scanned; the next update fills in everything since the
snapshot. A name matching no papers is reported as a likely typo.

**Removing** a category hides it rather than deleting it. With no category
ticked, searches cover your categories only. `update` keeps every category the
index holds current.

## Exporting and importing an index

An index, embeddings included, can be written to one file and installed
elsewhere, as a backup or to skip the build on another machine:

```bash
arxiv_index export arxiv_index.tar    # about 6.3 KB per paper
arxiv_index import arxiv_index.tar    # on the other machine
```

`export --settings` also includes your settings file and cached interest
embeddings. `import --settings` takes them up, keeping the old ones as
`config.json.bak`; merging keeps your categories as well as the export's.
Settings naming a different embedding model from the index's are refused.

Into an **empty index**, the import brings its model and categories and
updates your settings to match (and tells you to pull the model if missing).
Into a **non-empty index** with a different model, it needs one of:

| | |
|---|---|
| `--merge` | add the export's papers. A paper in both keeps the more recent copy (later arXiv version, then later date) with its embedding. A shared category is complete up to the later of the two dates. Nothing is re-embedded |
| `--replace` | discard this index and install the export instead |

A running `serve` picks up the result within seconds. Afterwards, `update`
brings it up to date, `build` adds any of your categories it lacks, and
`compact` reclaims space left by replaced papers.

**From the web UI**, under the cog, **Import and export** offers the same:
importing from the snapshot (for newly added categories), importing an export
(merge or replace, optionally with settings) and exporting (optionally with
settings). Files are streamed, not held in memory; keep the tab open until an
upload has been read. This section only appears on the machine running `serve`.

### The two index files are a matched set

`papers.db` is the source of truth: each paper's `row` column names its slot
in `vectors.f16`, which has no identity of its own. **Back them up together**,
or use `export`. If they get separated, rebuild the vectors:

```bash
sqlite3 ~/.arxiv_index/papers.db "UPDATE papers SET row = NULL"
rm ~/.arxiv_index/vectors.f16
arxiv_index build --embed-only
```

`vectors.f16` alone cannot be recovered: it is anonymous numbers.

## How it works

### Author matching

Works alone (newest first) or with a query (ranked by relevance). Several
names separated by `,`, `;` or `and` mean papers written *together*; this is
the opposite of **Followed authors**, which is a union. A single name written
surname-first works too and does better than `Godfrey Hardy`, since the terms
match independently and so also find "Godfrey H. Hardy". Case and accents are
ignored, since arXiv stores many names as LaTeX (`Poincar\'e`, `Erd\H{o}s`),
and affiliations in the author field are stripped.

**With** a query only embedded papers come back, since ranking needs a vector;
**without** one, the listing covers every paper in the database.

### How interest ranking works

Each interest is embedded separately, so they stay distinct instead of
averaging into a point that is squarely none of them. A paper is scored
against every interest, the scores are multiplied by their weights and sorted,
and each after the first counts less: the first in full, the second times *b*,
the third times *b²*. The **Reward for matching several** slider is *b*:

| *b* | effect |
|---|---|
| `0` | only the best match counts. A paper squarely on one project wins outright |
| `0.35` | the default. The best match dominates, but a genuine second match still lifts a paper |
| `1` | every match counts in full, ranking identically to averaging your interests into one vector |

A weight of `0` parks an interest without deleting it. Descriptions are
embedded on save, so changing weights or the slider embeds nothing. If Ollama
is unreachable the text is stored and flagged as unrankable until saved again.
Interests added by editing the settings file are embedded on the first
ranking. The embeddings are cached in `interest-vectors.json`, safe to delete.

Both profile buttons use the **Since**/**Until** range exactly as the form has
it; leave both empty to ask the whole index. Only embedded papers can be
ranked.

### Updates

arXiv's API cannot page back far enough for the whole history, hence the
snapshot for the initial build. `update` (and **Fetch new papers**) walks the
API back from the newest paper to the stored cursor, embeds what is new, and
re-embeds papers whose title or abstract changed. The cursor advances *only*
when a walk provably reached it, so a run cut short (`WALK INCOMPLETE`) wastes
work rather than leaving a gap.

Embedding takes an exclusive lock (`embed.lock`), so an `update` firing during
a long `build` exits cleanly. A running `serve` checks the index every few
seconds and picks up changes from any `build`, `update`, `import` or `compact`,
so it never needs restarting. Runs started from the web UI belong to the
server, not the tab: closing the page does not stop them. One runs at a time.

### Automatic updates

| | |
|---|---|
| **Off** | the default |
| **Every N hours** | 1 to 168, measured from when the last run started |
| **Daily at HH:MM** | to the quarter hour, in the server machine's **local** time (arXiv announces at a fixed New York time, so 07:00 means 07:00 where you are) |

A missed run is caught up, not skipped: a daily 07:00 run on a machine asleep
until 09:00 fires at 09:00, and switching the setting on runs within the minute
if a slot has passed. Manual runs reset the clock. Changes take effect within
30 seconds of pressing **Save**.

For cron, give the full path (`which arxiv_index`) if cron's `PATH` does not
reach it. Without installing:

```cron
0 7 * * 1  cd /path/to/arxiv_index && python3 -m arxiv_index update >> ~/.arxiv_index/update.log 2>&1
```

## Source layout

How the index is built is in `config.py`, with the measurements behind each
choice in the comments. Deeper background (why search is brute-force, what was
tried and abandoned) is in [NOTES.md](NOTES.md).

| | |
|---|---|
| `config.py` | models, tuning |
| `settings.py` | the per-person settings file: categories, model, profile |
| `store.py` | SQLite schema + append-only vector file |
| `embedder.py` | Ollama embedding calls with retry |
| `ingest.py` | snapshot scan + the resumable embedding loop |
| `update.py` | incremental fetch from the arXiv API |
| `search.py` | exact cosine search, author filtering |
| `textnorm.py` | LaTeX author names, folded for matching |
| `cite.py` | biblatex entries |
| `transfer.py` | exporting and importing an index |
| `profile.py` | followed authors + weighted interests, each with its own embedding |
| `schedule.py` | when the server tops itself up |
| `web.py` | the web server (stdlib `http.server`) |
| `static/` | the web pages: `index.html` and `setup.html`, their CSS and JS, and a vendored KaTeX |
| `__main__.py` | CLI |
