*This project has been created as part of the 42 curriculum by elal-hai.*

# RAG against the machine

## Description

This project implements a **Retrieval-Augmented Generation (RAG) system** that answers questions about the vLLM codebase (v0.10.1). The pipeline ingests the repository into a searchable BM25 index, retrieves the most relevant snippets for a given question, and generates a grounded natural-language answer using `Qwen/Qwen3-0.6B`. Retrieval quality is measured with a recall@k metric against reference datasets.

The system covers the full RAG lifecycle:

- **Indexing**: chunk the codebase (Python and Markdown, each with a dedicated strategy) and build a BM25 index.
- **Retrieval**: given a question, return the top-k most relevant source locations (`file_path` + character range).
- **Augmenting**: assemble retrieved sources into a context that fits the model's token budget.
- **Generating**: produce a coherent, grounded answer with `Qwen/Qwen3-0.6B`.

## Instructions

### Requirements

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) as package/project manager
- The vLLM v0.10.1 source tree placed under `data/raw/vllm-0.10.1/`
- The question datasets placed under `data/datasets/{AnsweredQuestions,UnansweredQuestions}/`

### Installation

```bash
uv sync
```

This installs all dependencies declared in `pyproject.toml` / `uv.lock`, including a CPU-only build of `torch` (no CUDA dependencies are pulled, since the mandatory model runs comfortably on CPU).

### Project layout

```
data/
  raw/vllm-0.10.1/                 # indexed source repository (not versioned)
  processed/                       # BM25 index, produced by `index` (not versioned)
  datasets/AnsweredQuestions/       # reference datasets (question + ground-truth sources + answer)
  datasets/UnansweredQuestions/     # question-only datasets
  output/search_results/            # output of `search_dataset`, scoped by dataset (not versioned)
  output/search_results_and_answer/ # output of `answer_dataset`, scoped by dataset (not versioned)
src/
  ingestion.py     # repository walking, chunking, BM25 index construction
  retrieval.py     # index loading, single/batch search
  generation.py     # answer generation with Qwen/Qwen3-0.6B
  evaluation.py     # local recall@k computation against a reference dataset
  models.py        # pydantic data models shared across stages
  cli.py           # Python Fire CLI exposing all commands
```

## Example usage

Every command is invoked as `uv run python -m src <command> [options]`.

### 1. Build the index

```bash
uv run python -m src index --max_chunk_size 2000
```

Ingests `data/raw/` and writes the BM25 index under `data/processed/`. Options: `--repo_path` (default `data/raw`), `--save_directory` (default `data/processed`).

### 2. Search a single query

```bash
uv run python -m src search "How to configure the OpenAI server?" --k 5
```

Prints the top-k source locations directly to the terminal.

### 3. Search a whole dataset

```bash
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions
```

Writes a `StudentSearchResults` JSON file (named after the input dataset) into `save_directory`.

### 4. Answer a single query

```bash
uv run python -m src answer "How to configure the OpenAI server?" --k 5
```

Runs retrieval then generation with `Qwen/Qwen3-0.6B`, and prints the question and the generated answer. Noticeably slower than `search` (tens of seconds on CPU) since it involves a full LLM forward pass.

### 5. Generate answers for a whole dataset

```bash
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

Takes an **already-produced** `StudentSearchResults` file (from `search_dataset`) and adds a generated answer to each entry — retrieval is not re-run. Writes a `StudentSearchResultsAndAnswer` JSON file. On CPU, expect roughly 30 seconds per question; budget accordingly for a 100-question dataset (~50 minutes).

### 6. Evaluate recall@k locally

```bash
uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```

Compares retrieved sources against a reference `AnsweredQuestions` dataset and prints recall@1/3/5/10. This command is for iterative development only — **the official recall@k used for grading is computed by the provided `moulinette` executable**, never by this code.

### Full pipeline example

```bash
uv run python -m src index --max_chunk_size 2000
uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json --k 10 --save_directory data/output/search_results/UnansweredQuestions
uv run python -m src evaluate --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_code_public.json --dataset_path data/datasets/AnsweredQuestions/dataset_code_public.json
uv run python -m src answer_dataset --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_code_public.json --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

### Makefile shortcuts

```bash
make install       # uv sync
make run           # show CLI help (see commands above for actual usage)
make debug ARGS="index --max_chunk_size 2000"   # run any command under pdb
make clean         # remove __pycache__, .mypy_cache, etc.
make lint          # flake8 + mypy (required flags)
make lint-strict   # flake8 + mypy --strict
```

## System architecture

The pipeline is organized into four independent stages, each in its own module:

1. **`ingestion.py`** — walks `data/raw/`, dispatches each file to a chunking strategy based on its extension, and builds a BM25 index (via `bm25s`) over all produced chunks. Chunk metadata (`file_path`, `first_character_index`, `last_character_index`) is persisted alongside the index so that retrieval results can be traced back to their exact source location.
2. **`retrieval.py`** — reloads the persisted index, tokenizes a query with the *same* tokenizer configuration used at indexing time (a single shared `tokenize_corpus` function guarantees this), and returns the top-k ranked sources. Works identically for a single query or a whole dataset.
3. **`generation.py`** — given retrieved sources, rereads the exact substrings from the original files on disk, assembles them into a token-budgeted context, and prompts `Qwen/Qwen3-0.6B` (via its chat template, with thinking mode disabled for concise answers) to produce a grounded answer.
4. **`evaluation.py`** — computes recall@k locally by comparing retrieved sources against a reference dataset, using an IoU-based overlap check on character ranges (same file + IoU ≥ 0.05 counts as a match, matching the grading rule described in the subject).

All stages communicate through `pydantic` models (`models.py`), which validate data at every hand-off and are the single source of truth for the JSON formats expected by the grading moulinette.

## Chunking strategy

Two chunking strategies are implemented, each tuned to the syntax of its file type:

- **Python (`chunk_python_file`)**: a new chunk starts at each top-level `def`/`class`, so that a chunk typically corresponds to one function or method. If a chunk would exceed `max_chunk_size`, it is force-split on the nearest newline before the limit.
- **Markdown (`chunk_md_file`)**: a new chunk starts at each header line (`#`, `##`, ...), keeping a section's content together. Oversized sections are force-split the same way as Python chunks.
- **Fallback (`chunk_other_file`)**: any other file type is chunked purely by size, without semantic boundaries (kept as a safety net; not triggered in practice since only `.py`/`.md` files are ingested by default).

All three strategies track a single running cursor (`current_char_index`) for the absolute position in the file, which is used directly to compute `first_character_index`/`last_character_index` — this avoids maintaining two separate offset counters that could silently drift apart after a forced split.

An alternative implementation using LangChain's `RecursiveCharacterTextSplitter.from_language()` was also tried and is discussed in **Design decisions** below.

## Retrieval method

Retrieval uses **BM25** (via the `bm25s` library), the classic lexical ranking algorithm. Both the corpus and incoming queries are tokenized through the same `tokenize_corpus()` function (English stopword removal, no stemming beyond what `bm25s` applies by default), which is critical: any divergence between indexing-time and query-time tokenization would silently break retrieval without raising any error.

Given a query, the system retrieves the top-k chunks by BM25 score and returns them as `MinimalSource` objects (`file_path`, `first_character_index`, `last_character_index`). A single shared index is used for both single-query search and dataset-wide batch search.

## Performance analysis

Measured on the public reference datasets (`dataset_docs_public.json`, `dataset_code_public.json`), using the project's own `evaluate` command:

| Dataset | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Required threshold |
|---|---|---|---|---|---|
| docs | 58.0% | 73.0% | **82.0%** | 86.0% | ≥ 80% recall@5 |
| code | 31.0% | 45.0% | **53.0%** | 59.0% | ≥ 50% recall@5 |

Both thresholds required by the subject are met, with a small margin (2 and 3 points respectively). The lower recall on code questions is expected: a natural-language question rarely shares vocabulary with source code identifiers, while questions about documentation tend to paraphrase the prose they are answered by.

**Indexing performance**: chunking + BM25 indexing of the full corpus (1937 files, ~27k chunks) completes in under 2 seconds, far below the 5-minute limit imposed by the subject. **Retrieval throughput**: searching 100 questions completes in a few seconds, far below the 90-second/200-question limit.

**Generation performance**: answer generation with `Qwen/Qwen3-0.6B` on CPU takes roughly 30 seconds per question (no time limit is imposed by the subject on this stage, unlike retrieval).

## Design decisions

- **BM25 over TF-IDF**: BM25 was chosen as the lexical retrieval method for its length-normalization properties, which suit a corpus with highly variable chunk sizes (short functions vs. long documentation sections).
- **Manual chunking over LangChain**: a version of the chunkers based on LangChain's `RecursiveCharacterTextSplitter.from_language()` was implemented and benchmarked. It produced roughly half as many, larger chunks (grouping multiple semantic units together up to `chunk_size`), which resulted in a *lower* recall@5 on the docs dataset (79% vs. 82% with the manual chunker) — likely because larger, more heterogeneous chunks dilute the BM25 signal for narrowly-targeted questions. The manual, boundary-driven chunker (splitting strictly on `def`/`class`/headers) was kept as the final implementation.
- **Single-cursor offset tracking**: chunking functions track one absolute character cursor rather than two independent counters (content length vs. read position), eliminating a class of silent offset-desynchronization bugs that would otherwise only surface as an unexplained recall drop.
- **`enable_thinking=False` for generation**: `Qwen3-0.6B` supports a "thinking" mode that emits visible reasoning traces. It was disabled to keep answers concise and directly usable, avoiding the need to parse and strip `<think>` blocks from the output.
- **Token-budgeted context construction**: rather than concatenating all k retrieved sources unconditionally, `generation.py` counts tokens as it assembles context and stops adding sources once a budget is reached, trading a small amount of potential context for a guarantee against exceeding the model's usable window.

## Challenges faced

- **Offset desynchronization bug**: an early version of the chunker maintained two separate counters (`current_char_index` for content length, `chunk_start_index` for the reported offset), which could drift apart after a forced split on oversized chunks. This was caught by writing a dedicated diagnostic script that re-reads each source file and verifies that `content[first:last+1]` matches the stored chunk text exactly — a check now considered essential before trusting any recall@k measurement.
- **Path-prefix mismatch**: `file_path` values must match the corpus path exactly as expected by the grader (e.g. `data/raw/vllm-0.10.1/...`). Running the indexer from a different working directory silently produced a different (and invalid) prefix; this is only caught by testing the CLI from the exact expected project root.
- **Disk quota constraints**: the default `torch` install pulls several gigabytes of CUDA dependencies (`nvidia-cudnn`, `nvidia-cublas`, etc.) that exceeded the small campus home-directory quota. Resolved by pinning `torch` to the CPU-only wheel index and relocating the `uv` cache and virtual environment to a larger partition.
- **`moulinette` schema mismatch**: the official grading executable expects a `question_str` field in `MinimalSearchResults`, while the subject's PDF documents (and shows in its JSON example) the field as `question`. This was resolved by adding a `question_str` computed field that mirrors `question` at serialization time, satisfying the moulinette without breaking compliance with the documented schema or requiring any change elsewhere in the codebase.

## Resources

- [BM25 (Wikipedia)](https://en.wikipedia.org/wiki/Okapi_BM25) — background on the ranking function.
- [`bm25s` documentation](https://github.com/xhluca/bm25s) — the BM25 implementation used for indexing and retrieval.
- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B) — model documentation, including chat template and thinking-mode usage.
- [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers) — `AutoModelForCausalLM`/`AutoTokenizer` API used for generation.
- [Pydantic v2 documentation](https://docs.pydantic.dev/latest/) — data validation, including `computed_field` usage.
- [Python Fire documentation](https://github.com/google/python-fire) — CLI framework used in `cli.py`.

**AI usage**: an AI assistant (Claude) was used throughout this project in a pedagogical, review-oriented capacity rather than as a code generator: reviewing hand-written chunking logic to identify a silent offset-desynchronization bug, explaining IoU-based recall@k computation before it was implemented, diagnosing environment issues (disk quota, CUDA dependency bloat, moulinette schema mismatch), and discussing trade-offs between chunking strategies (manual vs. LangChain) backed by measured recall@k comparisons. All code in this repository was written and is fully understood by the author; AI suggestions were verified empirically (via diagnostic scripts and the official moulinette) before being trusted.