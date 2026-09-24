# arXiv index

A personalized search engine for arXiv, running entirely on your own machine.
It looks up papers by meaning: describe what you are looking for in your own
words and it finds the papers closest to your description, whether or not they
use the same terms. It also helps you stay up to date on the research relevant
to you. Describe your interests and it finds the papers closest to your work.
Keep a list of researchers you follow, and never miss one of their papers.

![The web UI: a search for "toric degenerations of flag varieties", with the first result's abstract open](docs/screenshot.png)

## First-time setup

The repository is **code only**. A fresh clone can search nothing until you
build the index, which takes one large download and a few hours of embedding.
After that, keeping it current takes about a minute a week.

**1. Install Ollama and the embedding model.** Ollama runs the model that
turns text into vectors. Install it from [ollama.com](https://ollama.com), make
sure it is running (`ollama serve`, or the service its installer sets up), and
fetch the model, about 2.5 GB:

```bash
ollama pull qwen3-embedding:4b
```

To use a different model, see [Another embedding model](#another-embedding-model)
**before** building: an index cannot change model afterwards.

**2. Get the code and its Python dependencies.** Python 3.11 or newer:

```bash
git clone https://github.com/MattDupraz/arxiv_index.git
cd arxiv_index
pip install numpy ollama
pip install torch   # optional, see below
```

torch is optional and only helps with a GPU: it lets the web UI search on the
GPU, which is faster. Without it everything works, on the CPU.

Every command below is run from this directory.

If you have an index exported from another machine, skip steps 3 and 4:
`python3 -m arxiv_index import arxiv-index.tar` installs it (see
[Exporting and importing](#exporting-and-importing-an-index)).

**3. Download the arXiv snapshot.** arXiv's API cannot page back far enough to
fetch the whole history, so the index is first filled from Kaggle's copy of
arXiv's metadata:
[kaggle.com/datasets/Cornell-University/arxiv](https://www.kaggle.com/datasets/Cornell-University/arxiv)
(a free Kaggle account is needed), and unzip it anywhere. The file inside,
`arxiv-metadata-oai-snapshot.json`, is about 5.5 GB. It is needed only for the
build and can be deleted afterwards, though adding a category later needs it
again.

**4. Build the index**, giving it the snapshot:

```bash
python3 -m arxiv_index build ~/Downloads/arxiv-metadata-oai-snapshot.json
```

It first asks which arXiv categories to cover, by their full names (`math.AG`,
`hep-th`, `cs.LG`; see the [list](https://arxiv.org/category_taxonomy)).
Pressing Enter takes the default, math.AC, math.AG and math.CO. The answer is
saved in `~/.arxiv_index/config.json`, where you can change it later (see
[Adding a category](#adding-a-category)).

It then scans the snapshot for those categories, which takes a couple of
minutes, and embeds every paper. Embedding the default three categories'
145,000 papers takes about three hours on a consumer GPU, and more categories
take proportionally longer. It can be interrupted at any time, and
`python3 -m arxiv_index build --embed-only` carries on where it stopped. The
index takes about 6.3 KB of disk per paper, in `~/.arxiv_index/`.

To embed at a better time, say overnight, add `--scan-only`: it imports the
papers and stops. Embed them later with `build --embed-only`, or with
**Embed them now** in the web UI, shown beside the count of papers not yet
embedded. Until then they can be listed by author but not found by searches.
The next `update` embeds them too, and so does **Fetch new papers**, so either
of those can start the long run.

**5. Catch up to today.** The snapshot is a few days or weeks old. This fetches
everything posted since, from the arXiv API:

```bash
python3 -m arxiv_index update
python3 -m arxiv_index status    # what is in the index, and up to when
```

**6. Start the web UI.**

```bash
python3 -m arxiv_index serve     # opens http://127.0.0.1:8000/
```

Open the settings (the cog, top right) to add the authors you follow and
describe your research interests (see [Your profile](#your-profile)), and to
have the index [update itself](#automatic-updates) while the server runs.

## Settings

The index and everything personal live in `~/.arxiv_index/`, or wherever
`$ARXIV_INDEX_DIR` points. The personal part is one file, `config.json`, so
handing someone `papers.db` and `vectors.f16` does not hand them your profile.

```json
{
  "categories": ["math.AC", "math.AG", "math.CO", "math.NT"]
}
```

| | |
|---|---|
| `categories` | the arXiv categories you want, by their full names: `math.AG`, `hep-th`, `cs.LG` ([list](https://arxiv.org/category_taxonomy)). Default: math.AC, math.AG, math.CO |
| `embedding` | the Ollama embedding model. Default: `qwen3-embedding:4b`; see [below](#another-embedding-model) |

The profile and the automatic-update setting are stored here too, written by
the web UI. Hand edits are picked up without a restart, except the two keys
above, which `serve` reads when it starts.

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
vectors would not be comparable. To switch models, point `$ARXIV_INDEX_DIR` at
a new directory and `build` there. `query_prefix` is free to change at any
time.

An index from before this file existed moves its profile here the first time
it is opened.

### Adding a category

Add it to `categories` and run `build` with the snapshot's path. That scans the
snapshot **for the new categories only**, leaving papers already held alone,
and the next `update` fills in everything since the snapshot was taken. Until
then, `status` and the web UI list it as not yet in the index. A name that
matches no papers in the snapshot is reported, since it is most likely a typo.

### Removing a category

Removing a category from your settings hides it rather than deleting it. With
no category ticked, searches cover **your** categories only. The index still
holds the dropped one (as it does the categories of an index someone copied to
you), and `update` keeps every category it holds current.

### Exporting and importing an index

An index, embeddings included, can be written to one file and installed
elsewhere, to back it up or to spare another machine the build:

```bash
python3 -m arxiv_index export arxiv-index.tar    # about 6.3 KB per paper
python3 -m arxiv_index import arxiv-index.tar    # on the other machine
```

The file holds the papers and their embeddings, not your settings or profile.
Importing refuses an index built with a different embedding model from the one
your settings name. Where there is an index already, it needs one of:

| | |
|---|---|
| `--merge` | add the export's papers to this index. A paper in both keeps the more recent copy (the later arXiv version, then the later date), with its embedding. A category in both is complete up to the later of the two dates. Nothing needs embedding again |
| `--replace` | discard this index and install the export in its place |

A running `serve` picks up the result within a few seconds, as it does any
change to the index. After importing, `update` brings the index up to date,
and `build` adds any of your categories it does not hold. A merge that
replaced papers leaves their old vectors unused; `compact` reclaims the space.

### The two index files are a matched set

`papers.db` is the source of truth: each paper's `row` column names its
slot in `vectors.f16`, which has no identity of its own. **Back them up
together**, or use `export`, which does. If they are separated, the vectors
can be rebuilt:

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

**Scores are hidden by default** in both interfaces. The **Scores** checkbox and
`--scores` reveal them.

### Your profile

Two fields describing *you* rather than a search, filled in under the cog and
stored in your [settings file](#settings):

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
Only embedded papers can be ranked.

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

Embedding takes an exclusive lock (`embed.lock` in the index directory), so an
`update` firing during a long `build` exits cleanly. Searching during a build
is fine: a running `serve` checks the index every few seconds and picks up
whatever changed, whether from a `build`, `update`, `import` or `compact` run
in a terminal or by cron, so it never needs restarting for the index's sake.

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
0 7 * * 1  cd /path/to/arxiv_index && python3 -m arxiv_index update >> update.log 2>&1
```

## Commands

`python3 -m arxiv_index <command>`; every command takes `--help`.

| | |
|---|---|
| `build [SNAPSHOT]` | backfill from the snapshot, then embed. `--scan-only` stops before embedding, `--embed-only` skips the scan |
| `update` | fetch and embed what is new from the arXiv API |
| `search` | semantic search; `--author`, `--category`, `--since`, `--scores`, `--full`, `--json` |
| `similar` | neighbours of a given arXiv id |
| `serve` | the web UI; `--port`, `--host`, `--no-browser` |
| `status` | settings file, index location, model, counts, and per category how far it is complete |
| `config` | show your settings file, creating it with the defaults if absent |
| `compact` | reclaim vector slots left behind by re-embedded papers |
| `export FILE` | write the index, embeddings included, to one file |
| `import FILE` | install an exported index; `--merge` adds it to the one here, `--replace` overwrites it |

`serve` binds to `127.0.0.1` by default. The server exposes the index and,
indirectly, Ollama, so think before changing `--host`.

## Source layout

How the index is built is in `config.py`, with the measurements behind each
choice in the comments; what differs between people is in the settings file.
Deeper background — why search is brute-force, what was tried and abandoned
— is in [NOTES.md](NOTES.md).

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
| `web.py` | local web UI (stdlib `http.server`) |
| `__main__.py` | CLI |
