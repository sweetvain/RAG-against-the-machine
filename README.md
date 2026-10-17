# RAG Against the Machine

42 project implementing a Retrieval-Augmented Generation pipeline over the vLLM v0.10.1 repository.

The goal is straightforward: index the repository, retrieve the most relevant excerpts for a question, and generate an answer grounded in those excerpts. Official grading is done by the moulinette located in `exams/`.

## Overview

The pipeline is split into four stages:

- BM25 ingestion and indexing,
- retrieval of relevant sources,
- answer generation from retrieved context,
- local `recall@k` evaluation.

## Requirements

- Python 3.12+
- `uv`
- the vLLM v0.10.1 repository under `data/raw/vllm-0.10.1/`
- the datasets under `data/datasets/`

## Installation

```bash
uv sync
```

## Useful Layout

```text
data/
  raw/vllm-0.10.1/                 indexed source repository
  processed/                       BM25 index produced by `index`
  datasets/AnsweredQuestions/       reference datasets
  datasets/UnansweredQuestions/     question-only datasets
  output/                          outputs of `search_dataset` and `answer_dataset`
src/
  ingestion.py                     repository walking, chunking, indexing
  retrieval.py                     index loading and search
  generation.py                    answer generation from retrieved sources
  evaluation.py                    local recall@k computation
  models.py                        shared Pydantic models
  cli.py                           CLI commands
```

Two summary files are kept for grading notes:

- [RESUME_CHANGEMENTS.md](RESUME_CHANGEMENTS.md)
- [RESUME_PROJET.md](RESUME_PROJET.md)

## Main Commands

All commands go through the `src` module:

```bash
uv run python -m src index --max_chunk_size 2000
uv run python -m src search "How to configure the OpenAI server?" --k 5
uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --k 10 --save_directory data/output/search_results/UnansweredQuestions
uv run python -m src answer "How to configure the OpenAI server?" --k 5
uv run python -m src answer_dataset --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json --save_directory data/output/search_results_and_answer/UnansweredQuestions
uv run python -m src evaluate --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```

## How It Works

### Indexing

`src/ingestion.py` walks the source repository, splits files into chunks, and builds a BM25 index.

- `.py` files are split by `def` / `class`,
- `.md` files are split by sections,
- other files use a size-based fallback.

Each chunk keeps its metadata: `file_path`, `first_character_index`, and `last_character_index`.

### Retrieval

`src/retrieval.py` reloads the index, tokenizes the query with the same logic used for the corpus, and fetches the best sources.

A small reranking pass favors documentation files, which improves docs performance without breaking the output contract.

### Generation

`src/generation.py` rereads the exact text of the retrieved sources, builds a token-limited context, and asks `Qwen/Qwen3-0.6B` to produce a concise answer.

### Evaluation

`src/evaluation.py` compares retrieved sources against the reference dataset and computes local `recall@k`.

## Output Format

Search results are serialized as `StudentSearchResults`.

The format expected by the moulinette is preserved as-is to remain compatible with grading.

## Validation

The project was validated with the retrieval moulinette provided in `exams/scripts/`.

- indexing: OK
- search throughput: OK
- docs recall@5: OK
- code recall@5: OK

## Notes

- [RESUME_CHANGEMENTS.md](RESUME_CHANGEMENTS.md) summarizes all modifications.
- [RESUME_PROJET.md](RESUME_PROJET.md) describes the full project in detail.
