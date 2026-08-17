# arXiv index — math.AC / math.AG / math.CO

Semantic search over arXiv abstracts in commutative algebra, algebraic geometry
and combinatorics.

A paper is in scope if **any** of its categories is one of the three, so
cross-listed work (e.g. a `cs.CG math.CO` paper) counts. That is ~145,000 papers
out of the snapshot's 3.1M.

```bash
python3 -m arxiv_index serve      # http://127.0.0.1:8000/
python3 -m arxiv_index update     # weekly top-up, about a minute
```

| | |
|---|---|
| vector search | ~210 ms (175 ms of it embedding the query) |
| + reranking the top 50 | ~800 ms |
| index | 145,853 papers, 747 MB of vectors |

## Requirements

- **Ollama**, with the embedding model: `ollama pull qwen3-embedding:4b`
- **Python 3.11+** with `numpy` and `ollama`
- **torch + transformers** — only for reranking and GPU search; everything else
  works without them

The arXiv API client uses only the standard library.

### Installing torch

```bash
pip install --user --index-url https://download.pytorch.org/whl/rocm7.0 torch
pip install --user transformers
```

ROCm 7.0 wheels recognise this machine's RDNA4 GPU (gfx1200) natively. Fedora's
`python3-torch` will *not* do — it is a CPU-only build and there is no ROCm
variant in the repos.

Installed to `--user` rather than a virtualenv on purpose: it is one library, not
a dependency set worth isolating, and a venv would mean two interpreters —
`python3` silently giving a slower, un-reranked search and `.venv/bin/python`
giving the real thing. The cost is ~13 GB in `~/.local` visible to every
`python3`, removed with `pip uninstall` rather than deleting a directory.

## Using it

The web UI has a search box, an author filter, category checkboxes, a since-date
filter, expandable abstracts, links to the abstract and PDF, a **BibLaTeX**
button, and **Similar papers** on every result. Standard library only — no
framework, no CDN, works offline. LaTeX in titles and abstracts is rendered with
a vendored copy of KaTeX; accents outside maths mode (`Gr\"obner`) are rewritten
to Unicode separately, since KaTeX never sees them.

```bash
python3 -m arxiv_index search "toric degenerations of flag varieties"
python3 -m arxiv_index search "chromatic polynomial" -k 20 --category math.CO
python3 -m arxiv_index search "singularities of pairs" --author Kollar
python3 -m arxiv_index search --author "Larson, Payne"     # no query needed
python3 -m arxiv_index similar 0704.0002
python3 -m arxiv_index status
```

`--rerank` enables reranking from the CLI, `--scores` shows relevance
numbers, and `--json` prints full records.

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

**Names match regardless of case and accents**, because arXiv stores 13.8% of
author fields as LaTeX (`J\'anos Koll\'ar`, `Mikkel {\O}bro`). A literal
`LIKE '%Kollar%'` finds **1** paper; folding both sides to lowercase ASCII finds
**166**. Affiliations riding along in the field are stripped, so a place name
does not match everyone who works there. `textnorm.py` splits this into
`latex_to_unicode` (what a human would write) and `fold` (what is compared).

The two modes reach different sets, deliberately. **With a query**, only embedded
papers can be returned — ranking needs a vector. **Without one**, the search is
pure metadata and covers the whole corpus. That matters: restricting a name
lookup to embedded rows silently drops papers, and mid-build that was half of
them.

### Citations

The **BibLaTeX** button emits an `@online` entry with `eprint` / `eprinttype` /
`eprintclass`, plus `doi` and the journal reference when arXiv has them. Two
things are less obvious than they look:

- **The year comes from the arXiv identifier, not `update_date`.** That column
  records when the *metadata* last changed, so `alg-geom/9202001` — a 1992 paper
  — carries `update_date` 2008-02-03. Over 20,000 records its year disagrees
  with the true submission year **51%** of the time. Both identifier schemes
  begin with YYMM once the archive prefix is dropped.
- **Titles are emitted verbatim, because they are already LaTeX.** Escaping would
  turn `$K_4$` into literal dollar signs. The exception is unbalanced braces,
  which a few author-supplied titles genuinely contain: those are dropped, since
  BibTeX's lexer counts braces literally (escaping as `\{` does not help) and one
  unparseable entry makes biber skip into the next.

The journal reference goes in `note`, not `journaltitle`: arXiv stores it as free
text ("J. Alg. Geom. 1 (1992) 449--530") that does not decompose reliably.
Validated by generating 40,000 entries and parsing them with **biber** — three
defects showed up that way, each on roughly one record in ten thousand, none
visible by eye.

## Keeping it current

```bash
python3 -m arxiv_index update
```

Walks the arXiv API back from the newest paper until it reaches the stored
cursor, embeds what is new, advances the cursor. The 5.5 GB Kaggle snapshot is
only for the initial backfill. Papers whose title or abstract changed are
re-embedded; papers that merely gained a DOI are not.

```cron
0 7 * * 1  cd /home/matt/Code/arXiv_index && python3 -m arxiv_index update >> update.log 2>&1
```

**Can it miss a paper?** The cursor advances *only* when a walk provably reached
it. A run cut short says `WALK INCOMPLETE`, leaves the cursor alone, and keeps
what it fetched, so the failure mode is wasted work rather than a gap. Re-run
with a larger `--max-pages`.

Two arXiv API traps, both easy to reintroduce:

- **`lastUpdatedDate:[A TO B]` does not filter on the last-update date.** It
  matches the *original submission* date, while `sortBy=lastUpdatedDate` sorts on
  the real one. Bounding a window with it drops revisions of older papers —
  measured at 38% of an 8-day window (342 results instead of 555), which is
  exactly what an incremental updater exists to catch. No range filter is used.
- **An empty page is not the end of the stream.** arXiv returns blank pages
  transiently, so `update` retries an offset and consults
  `opensearch:totalResults` rather than trusting a short page.

Embedding runs take an exclusive lock (`index/embed.lock`), so a cron `update`
firing during a long `build` exits cleanly instead of double-embedding.
Searching during a build is fine.

## How it works

```
index/papers.db     SQLite: metadata, one row per paper
index/vectors.f16   flat float16 array, 2560 dims per paper
```

`papers.row` is a paper's slot in the vector file, or `NULL` if it still needs
embedding. **That single nullable column is the whole work queue**: every ingest
path writes metadata with `row = NULL` and `embed_pending` drains it, so an
interrupted build resumes exactly where it left off. The vector file is
append-only; re-embedding a revision appends a slot and repoints the paper, and
`compact` reclaims the stale ones.

### Search is exact, with no ANN index

145k × 2560 float16 is ~747 MB, and scoring a query against all of it is one
matrix-vector product. So search is brute-force and therefore **exact** — no
recall cliff, no tuning, and nothing to rebuild when papers are appended. Adding
a paper is appending 5 KB to a file. This is the main reason the index is cheap
to maintain, and it holds to a few million papers.

`GPU_SEARCH` mirrors the matrix into VRAM: **418 ms → 2.8 ms**, a 149× speed-up
for 747 MB and a 0.14 s upload whenever the index grows. Server only — a CLI
search is a fresh process and would pay the torch import to save 0.4 s. It falls
back to CPU silently if torch or the GPU is unavailable.

The GPU computes in float16 where the CPU promotes to float32, so scores differ
by ~2e-4 — enough to swap papers that were already tied. In one query two
abstracts 1.94e-05 apart traded places because both round to exactly 0.719727 in
float16. Same papers, arbitrary order between two of them. `GPU_SEARCH = False`
restores bit-identical agreement with the CLI.

### What the server holds

Resident set of the `serve` process, measured on the 145k-paper index:

| | RSS | anonymous |
|---|---|---|
| CPU search (`GPU_SEARCH = False`) | 840 MB | **98 MB** |
| GPU search | 1.0 GB | 566 MB |
| after one reranked search | 2.1 GB | 1.5 GB |

Only the anonymous column is memory the kernel cannot take back. On the CPU path
the other 742 MB is the vector file mapped in: clean page-cache, evicted under
pressure and re-read from disk, so the server nominally holding 840 MB does not
mean 840 MB is unavailable to anything else.

Almost everything above 100 MB is torch: ~480 MB to import it and open a HIP
context, and another ~700 MB the first time a kernel runs, which is ROCm loading
its kernel libraries and is not returned afterwards. That cost is per-process and
independent of corpus size. It buys the 149× search speed-up and the reranker; if
neither is wanted, `GPU_SEARCH = False` and leaving **Rerank** unticked keeps the
process under 100 MB of real memory.

Two things keep the rest small, both of which had to be built rather than freed —
CPython returns very little to the OS once it has grown:

- **The host copy of the matrix is dropped after the upload to VRAM.** Nothing
  reads it again while `gpu` is set, and the upload has just paged all 747 MB in.
- **Per-row metadata is streamed and pooled.** `fetchall()` on 145k rows is ~65 MB
  of `sqlite3.Row` objects that a build re-pays every few seconds, and the rows
  are mostly repetition: 5.2k distinct dates and 4.6k distinct category sets
  across 145k papers, plus every folded author string built twice. Holding one
  instance of each turns 47 MB of category sets into under one.

### Reranking

The index is a *bi-encoder*: query and document are embedded separately, so their
vectors never interact and the ranking is only as good as one dot product can
express. A cross-encoder reads the pair *together* — much better, and far too
slow for 145k papers. So the index proposes 50 candidates and
`Alibaba-NLP/gte-reranker-modernbert-base` reorders them.

Measured on 50 known-item queries against identical shortlists:

| | recall@1 | recall@5 | MRR | 50 docs | VRAM |
|---|---|---|---|---|---|
| vector only | 0.500 | 0.760 | 0.622 | — | — |
| Qwen3-Reranker-0.6B | 0.760 | 0.820 | 0.781 | 1.66 s | 1.11 G |
| Qwen3-Reranker-4B | 0.800 | 0.860 | 0.827 | 4.09 s | 7.54 G |
| bge-reranker-v2-m3 | 0.820 | 0.860 | 0.835 | 0.65 s | 1.13 G |
| **gte-reranker-modernbert-base** | **0.860** | **0.860** | **0.860** | **0.38 s** | **0.36 G** |

modernbert beat bge on 2 cases, lost 0, tied 41. Two discordant pairs is not
significance — but every earlier comparison traded wins for losses, and this one
is strictly non-worse at 1.7× the speed in a third of the VRAM.

#### A failure the harness could not see

Asked for *"K-rings of matroids"*, bge put a paper on **g-elements** above the
actual **K-rings** paper — 2.35 against 2.26, i.e. the wrong order and only 0.09
apart in a 2.9 range. It recognised "matroid", which all 50 candidates share so
the signal is useless, and knew nothing about K-theory. The embedding meanwhile
had the right paper at cosine rank 2: confident and correct. modernbert scores
the same pair 3.79 and 1.74.

Known-item retrieval measures finding *one specific paper*. It cannot detect bad
ordering among near-neighbours in a jargon-dense field, which is what browsing
actually surfaces. One real query was worth more than fifty synthetic ones.

Reciprocal rank fusion was tried as a fix — blending the reranker's order with
the index's so an indifferent reranker cannot overrule a confident index. It
repaired that query but cost 10 points of recall@1 (0.820 → 0.720), because it
damps the reranker uniformly: of four cases where it pulled a target from rank
≥10 into the top 3, fusion pushed three back out. Removed. A better model turned
out to be the right answer. If the problem recurs, the principled version is to
gate fusion on the reranker's confidence — fuse only when its top margins are
bunched — which would keep the rescues.

#### Models that did not work here

| | |
|---|---|
| zerank-2 (4 B), zerank-1-small (1.7 B) | never finished loading — CPU-bound at ~0.25 GiB/min, >5 min even on an idle GPU |
| gte-multilingual-reranker-base | crashed the GPU (`HSA_STATUS_ERROR_EXCEPTION`) on its custom kernels |
| jina-reranker-v2 | custom code imports symbols removed in transformers 5.x |
| jina-reranker-v3 | `score.weight` absent from the checkpoint — the head loads randomly initialised |

The pattern: **anything relying on custom remote code is a lottery** on RDNA4
with transformers 5.x. Plain `ForSequenceClassification` on stock transformers
works. The backend refuses other architectures rather than guessing, because
loading a causal model through `AutoModelForSequenceClassification` *succeeds*
and silently invents a classification head — worse than an error.

#### Details that carry the performance

`RERANK_CANDIDATES` is 50 rather than 100 on measurement: doubling the shortlist
moved the known-item ceiling only from 86% to 88%, because six of the seven
misses are not in the top 100 either. Those are failures of the embedding, not
of the cutoff.

- **Sort by length before batching.** Every sequence is padded to the longest in
  its batch, and on a real shortlist that padding was 40% of the compute.
- **Small batches beat large**, for the same reason.
- **Right padding is a correctness requirement**, not a preference. The model
  reads CLS at position 0; pad on the left and it still returns plausible numbers
  while scoring the wrong position.

Scores are hidden by default, in both interfaces — they are diagnostics, not
reading material. The **Scores** checkbox reveals them in the web UI (remembered
across sessions, and toggling costs no re-search since the badges are only
hidden, not removed); `--scores` does the same on the CLI. A reranked hit then
shows its relevance logit *and* the cosine it started from, since the two are on
unrelated scales and one without the other misleads. They are always present in
the API payload and in `--json`.

The checkbox turns reranking off, and it is skipped automatically for
author-only listings, where there is no query to be relevant to. If the reranker
is missing or broken, the search falls back to vector order and says so rather
than failing.

**Similar papers** honours the same checkbox. The cross-encoder scores a text
pair either way, so the source paper's own title and abstract simply stand in
for the query — nothing about the model changes, only what sits on the left. It
costs about a second, against 8 ms for the plain vector lookup, so it is worth
leaving off when you are just skimming neighbours.

### Embedding

- Documents are embedded as `"{title}\n\n{abstract}"` with whitespace collapsed
  (arXiv hard-wraps both at ~80 characters).
- Queries get the Qwen3-Embedding instruct prefix; documents deliberately do not.
- Vectors are L2-normalised before storage, so cosine is a plain dot product.
- float16 storage halves bytes read per search; round-trip error ~1e-5.

Change `MODEL` in `config.py` and the index refuses to load rather than silently
mixing incomparable vectors.

**The query embedding runs on the CPU.** A search embeds one short query while
the reranker scores fifty documents, so the GPU is better spent on the latter:
175 ms against 89 ms, in exchange for 4.1 GB of VRAM. Indexing keeps the GPU at
14.1 docs/s.

`num_gpu` must be stated explicitly on **both** paths. Ollama does not move a
model back on its own — once loaded with `num_gpu: 0` it stays on the CPU, and a
request that merely omits the option will not return it. This silently
invalidated a measurement here, where a supposed GPU-vs-CPU comparison was really
CPU against CPU and came out impossibly bit-identical. Alternating costs a ~3 s
reload each way, so searches issued *during* an update will thrash.

**Ollama's embeddings are not deterministic**: reduction order depends on
batching, so the same query can come back ~4e-3 apart, `cos 0.9996`. Query
embeddings are therefore memoised on `(query, model)` — mainly a latency win
(~104 ms per repeat, and the UI resubmits the same text whenever you change a
filter), with reproducibility as a side effect. The instability was harmless:
papers 3e-3 apart in cosine are ties.

### Benchmarking

The GPU downclocks when idle, and the first searches after a quiet spell run
**4× slower** until the clocks ramp. Warm up before timing anything, or you will
measure power management.

## Tuning

Everything adjustable is in `arxiv_index/config.py`, with the measurements behind
each choice in the comments. Changing `CATEGORIES` and re-running `build` adds
categories without re-embedding what you have.

## Files

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
