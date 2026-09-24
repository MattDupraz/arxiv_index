# Design notes

Why the index is built the way it is: the measurements behind each choice, and
the things that were tried and did not work. None of this is needed to use it —
the [README](README.md) covers that. Much of it also appears as comments at the
point of use, where it is harder to miss: the query-embedding measurements
in `config.py`, the arXiv API traps in `update.py`, the storage invariants in
`store.py`. Reranking, and how its model was chosen, is on the `reranking`
branch.

## Search is exact, with no ANN index

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

## What the server holds

Resident set of the `serve` process, measured on the 145k-paper index:

| | RSS | anonymous |
|---|---|---|
| CPU search (`GPU_SEARCH = False`) | 840 MB | **98 MB** |
| GPU search | 1.0 GB | 566 MB |

Only the anonymous column is memory the kernel cannot take back. On the CPU path
the other 742 MB is the vector file mapped in: clean page-cache, evicted under
pressure and re-read from disk, so the server nominally holding 840 MB does not
mean 840 MB is unavailable to anything else.

Almost everything above 100 MB is torch: ~480 MB to import it and open a HIP
context, and another ~700 MB the first time a kernel runs, which is ROCm loading
its kernel libraries and is not returned afterwards. That cost is per-process and
independent of corpus size. It buys the 149× search speed-up; if that is not
wanted, `GPU_SEARCH = False` keeps the process under 100 MB of real memory.

Two things keep the rest small, both of which had to be built rather than freed —
CPython returns very little to the OS once it has grown:

- **The host copy of the matrix is dropped after the upload to VRAM.** Nothing
  reads it again while `gpu` is set, and the upload has just paged all 747 MB in.
- **Per-row metadata is streamed and pooled.** `fetchall()` on 145k rows is ~65 MB
  of `sqlite3.Row` objects that a build re-pays every few seconds, and the rows
  are mostly repetition: 5.2k distinct dates and 4.6k distinct category sets
  across 145k papers, plus every folded author string built twice. Holding one
  instance of each turns 47 MB of category sets into under one.

## Embedding

- Documents are embedded as `"{title}\n\n{abstract}"` with whitespace collapsed
  (arXiv hard-wraps both at ~80 characters).
- Queries get the Qwen3-Embedding instruct prefix; documents deliberately do not.
- Vectors are L2-normalised before storage, so cosine is a plain dot product.
- float16 storage halves bytes read per search; round-trip error ~1e-5.

Change `MODEL` in `config.py` and the index refuses to load rather than silently
mixing incomparable vectors.

**The query embedding runs on the CPU**: 175 ms against 89 ms on the GPU, in
exchange for 4.1 GB of VRAM left to the vector matrix and to builds. Indexing keeps the GPU at
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

## Benchmarking

The GPU downclocks when idle, and the first searches after a quiet spell run
**4× slower** until the clocks ramp. Warm up before timing anything, or you will
measure power management.

## Two arXiv API traps

Both are easy to reintroduce, so `update` works around them deliberately.

- **`lastUpdatedDate:[A TO B]` does not filter on the last-update date.** It
  matches the *original submission* date, while `sortBy=lastUpdatedDate` sorts on
  the real one. Bounding a window with it drops revisions of older papers —
  measured at 38% of an 8-day window (342 results instead of 555), which is
  exactly what an incremental updater exists to catch. No range filter is used.
- **An empty page is not the end of the stream.** arXiv returns blank pages
  transiently, so `update` retries an offset and consults
  `opensearch:totalResults` rather than trusting a short page.

## Citations

The **BibLaTeX** button emits an `@online` entry with `eprint` / `eprinttype` /
`eprintclass`, plus `doi` and the journal reference when arXiv has them. Two
things there are less obvious than they look:

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

## Author matching

**Names match regardless of case and accents**, because arXiv stores 13.8% of
author fields as LaTeX (`Poincar\'e`, `Erd\H{o}s`, `{\O}re`). Taking one surname
in the corpus: a literal `LIKE` on its ASCII spelling finds **1** paper, while
folding both sides to lowercase ASCII finds **166**. Affiliations riding along
in the field are stripped, so a place name does not match everyone who works
there. `textnorm.py` splits this into `latex_to_unicode` (what a human would
write) and `fold` (what is compared).

The two modes reach different sets, deliberately. **With a query**, only embedded
papers can be returned — ranking needs a vector. **Without one**, the search is
pure metadata and covers the whole corpus. That matters: restricting a name
lookup to embedded rows silently drops papers, and mid-build that was half of
them.
