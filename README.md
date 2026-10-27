*This project has been created as part of the 42 curriculum by elal-hai.*

# RAG Against the Machine

This project builds a small retrieval-augmented generation pipeline for a local code and documentation corpus.

The workflow is simple:

- index a corpus of source files,
- retrieve the most relevant excerpts for a question,
- generate an answer grounded in those excerpts,
- evaluate the retrieved sources against a reference dataset.

## Overview

The project is organized in four main stages:

- ingestion and BM25 indexing,
- retrieval of relevant sources,
- answer generation from the retrieved context,
- local recall@k evaluation.

## Requirements

- Python 3.12+
- `uv`
- a corpus directory to index (default: `data/raw`)
- the public dataset files under `data/datasets/`

## Installation

```bash
uv sync
```

## Project Layout

```text
data/
  raw/                              optional source corpus to index
  processed/                        BM25 index written by the ingestion step
  datasets/AnsweredQuestions/      reference answers for evaluation
  datasets/UnansweredQuestions/    question-only dataset for retrieval/search
  output/                           generated search and answer outputs
src/
  ingestion.py                      repository walking, chunking, index generation
  retrieval.py                      BM25 loading and retrieval logic
  generation.py                     answer generation from retrieved context
  evaluation.py                     local recall@k evaluation
  models.py                         shared Pydantic data models
  cli.py                            CLI commands exposed via Fire
```

Two project summaries are included in the repository:

- [RESUME_CHANGEMENTS.md](RESUME_CHANGEMENTS.md)
- [RESUME_PROJET.md](RESUME_PROJET.md)

## Main Commands

All commands are available through the `src` module:

```bash
uv run python -m src --help
uv run python -m src index --max_chunk_size 2000 --repo_path data/raw --save_directory data/processed
uv run python -m src search "How to configure the OpenAI server?" --k 5 --index_directory data/processed
uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --k 10 --save_directory data/output/search_results/UnansweredQuestions
uv run python -m src answer "How to configure the OpenAI server?" --k 5 --index_directory data/processed
uv run python -m src answer_dataset --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json --save_directory data/output/search_results_and_answer/UnansweredQuestions
uv run python -m src evaluate --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```

## How It Works

### Indexing

`src/ingestion.py` walks the source corpus, splits files into chunks, and builds a BM25 index.

- `.py` files are split by `def` and `class` boundaries,
- `.md` files are split by their sections,
- other files fall back to a size-based split.

Each chunk keeps metadata such as `file_path`, `first_character_index`, and `last_character_index`.

### Retrieval

`src/retrieval.py` loads the index, tokenizes the query with the same logic as the corpus, and retrieves the best matches.

The ranking step includes a small reranking adjustment to favor documentation-like sources without changing the required output contract.

### Generation

`src/generation.py` reloads the exact retrieved text, builds a token-limited context, and asks the model to answer from that context.

### Evaluation

`src/evaluation.py` compares retrieved sources against the expected dataset and computes a local recall@k score.

## Output Format

The search results are serialized as `StudentSearchResults`, and answer results as `StudentSearchResultsAndAnswer`.

These schemas are kept aligned with the project’s evaluation flow and with the expected grading contract.

## Validation

The project is validated with the standard local checks:

```bash
make lint
```

This runs both `flake8` and `mypy` on the project sources.

## Notes

- [RESUME_CHANGEMENTS.md](RESUME_CHANGEMENTS.md) summarizes the project changes and cleanup work.
- [RESUME_PROJET.md](RESUME_PROJET.md) gives a broader project overview.
