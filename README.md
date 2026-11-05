*This project has been created as part of the 42 curriculum by elal-hai.*

# RAG Against the Machine

## Description

A Retrieval-Augmented Generation (RAG) system that answers questions about a
codebase (the vLLM 0.10.1 repository). It indexes the sources, retrieves the
most relevant snippets for a question with BM25, and asks a small local model
(`Qwen/Qwen3-0.6B`) to write an answer grounded in those snippets. Retrieval
quality is measured with recall@k.

Pipeline: **index -> retrieve -> augment -> generate**, plus a local
`evaluate` command.

## Instructions

Requirements: Python 3.12+, [`uv`](https://docs.astral.sh/uv/), and a few GB
of free disk space (PyTorch CPU + model weights).

```bash
make install                  # uv sync
```

Put the vLLM repository in `data/raw/` (so that files live under
`data/raw/vllm-0.10.1/...`). The datasets are already in `data/datasets/`.

```bash
uv run python -m src --help
```

Other make rules: `make run`, `make debug ARGS="index"`, `make clean`,
`make lint` (flake8 + mypy with the mandatory flags), `make lint-strict`.

### Example usage

```bash
# 1. Index the corpus (writes data/processed/)
uv run python -m src index --max_chunk_size 2000

# 2. Single query
uv run python -m src search "How to configure the OpenAI server?" --k 5
uv run python -m src answer "How to configure the OpenAI server?" --k 5

# 3. Search a whole dataset
uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    --k 10 \
    --save_directory data/output/search_results/UnansweredQuestions

# 4. Local recall@k
uv run python -m src evaluate \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json

# 5. Generate answers for a whole dataset
uv run python -m src answer_dataset \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

All input/output paths are CLI arguments. Invalid input (empty query, `k <= 0`,
non-integer `k`, missing file, malformed JSON, missing index) produces a clear
error message and exit code 1, never a traceback.

## System architecture

```text
data/raw/ ──> ingestion.py ──> data/processed/ (BM25 index + chunk metadata)
                                      │
 question ──> retrieval.py <──────────┘   -> list of MinimalSource
                   │
                   ▼
            generation.py  (re-reads the exact text of each source,
                            builds a token-limited context, Qwen3-0.6B)
                   │
                   ▼
        JSON (StudentSearchResults / StudentSearchResultsAndAnswer)
                   │
            evaluation.py  (local recall@k, IoU >= 0.05)
```

| File | Role |
| --- | --- |
| `src/models.py` | Pydantic models exchanged between the stages |
| `src/ingestion.py` | Corpus walk, chunking, tokenization, BM25 index |
| `src/retrieval.py` | Index loading, search, batch search over a dataset |
| `src/generation.py` | Context building and answer generation |
| `src/evaluation.py` | Local recall@k |
| `src/cli.py` | Python Fire CLI, argument validation, error handling |

## Chunking strategy

Two strategies, selected by file extension (`--max_chunk_size`, default 2000):

- **Python**: a new chunk starts at each `def`, `async def`, `class` or
  decorator (a decorator stays attached to its function).
- **Markdown / text**: a new chunk starts at each heading (`#`...). Headings
  inside fenced code blocks are ignored. `.txt` files are split by size only.

In both cases a chunk that exceeds `max_chunk_size` is cut at the last newline
(hard cut if there is none). A chunk is **never longer than `max_chunk_size`**,
which the grader requires. Chunk content is the exact slice of the file, and
files are read with `newline=""` so that character offsets are identical on
Windows (CRLF) and Linux.

Stored `file_path` values always use `/` (e.g.
`data/raw/vllm-0.10.1/docs/features/lora.md`), because the grader compares
paths verbatim.

## Retrieval method

- **BM25** (`bm25s`) over the chunks.
- **Tokenizer** shared by indexing and queries: identifiers are kept whole
  (`load_lora_adapter`) *and* split into parts (`load`, `lora`, `adapter`;
  snake_case and camelCase), lowercased, English stopwords removed, plus a
  light suffix normalization (`supported` -> `support`). This allows matching
  both a verbatim identifier and a paraphrase.
- The file path is prepended to the indexed text so that folder and module
  names take part in matching.

## Performance analysis

Measured on the public datasets, `k = 10`, CPU only, default chunk size 2000:

| Dataset | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Required @5 |
| --- | --- | --- | --- | --- | --- |
| docs (100 q) | 0.640 | 0.830 | **0.840** | 0.860 | >= 0.80 |
| code (99 q) | 0.404 | 0.646 | **0.687** | 0.737 | >= 0.50 |

Timings: indexing about 15-50 s for about 28,000 chunks (limit 5 min); batch
search about 8 s per 100 questions (limit 90 s per 200).

Effect of `--max_chunk_size` (recall@5, docs / code):

| max_chunk_size | chunks | docs | code |
| --- | --- | --- | --- |
| 500 | 55,813 | 0.82 | 0.697 |
| 1000 | 36,237 | 0.83 | 0.667 |
| 1500 | 30,714 | 0.84 | 0.697 |
| 2000 | 28,362 | 0.84 | 0.687 |

Differences between sizes are within one or two points, so the default of 2000
(the maximum allowed) is kept: it gives the fewest chunks and the best docs
score.

Answers: on a sample of questions, Qwen3-0.6B gives correct and grounded
answers (e.g. `/v1/load_lora_adapter` for the LoRA question). Generation takes
a few seconds per question on CPU.

## Design decisions

- **BM25 only**: simple, fast to index, and very strong on code/doc questions
  that quote identifiers.
- **Same tokenizer on both sides**, otherwise vocabularies would not match.
- **Chunks as exact file slices**: offsets, stored text and the text given to
  the model can never drift apart.
- **Context built from the retrieved sources** in relevance order, within a
  token budget (`MAX_CONTEXT_TOKENS = 3000`); the prompt forces the model to
  answer only from the context or say it does not know. Thinking mode of
  Qwen3 is disabled and decoding is greedy (deterministic).
- **Model cached with `lru_cache`** so it is loaded once per run.
- **Pydantic** for every model exchanged between stages and for validating
  input files.

## Challenges faced

- **0 % recall on Windows**: paths were stored with `\`, but the grader compares
  `/` paths verbatim. Fixed with `Path.as_posix()`.
- **Offsets drifting**: stripping chunk text and reading files in text mode
  (CRLF conversion) made character ranges inaccurate. Fixed by slicing the raw
  text and reading with `newline=""`.
- **Chunks over the limit**: very long lines could produce chunks larger than
  2000 characters. The splitting loop now cuts until every chunk fits.
- **Infinite loop on `--max_chunk_size 0`**: now rejected cleanly.
- **`mypy .` crashing on the corpus**: it scanned `data/raw`; `data/` is now
  excluded from mypy and flake8.

## Resources

- BM25: Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and
  Beyond*; [bm25s](https://github.com/xhluca/bm25s)
- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP
  Tasks* (2020)
- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B),
  [Hugging Face Transformers docs](https://huggingface.co/docs/transformers)
- [Pydantic](https://docs.pydantic.dev/),
  [Python Fire](https://github.com/google/python-fire),
  [uv](https://docs.astral.sh/uv/), [tqdm](https://tqdm.github.io/)
- [vLLM](https://github.com/vllm-project/vllm) (the indexed corpus)

**Use of AI**: AI assistants were used to review the code against the subject,
to help diagnose the Windows path/offset bugs, to write unit tests and to
draft this README. Every change was run and checked (recall measured on the
datasets, flake8, mypy, pytest) and is understood by the author.
