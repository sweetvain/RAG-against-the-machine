import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import bm25s
import pydantic
from tqdm import tqdm

from src.ingestion import tokenize_corpus
from src.models import (
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)

logger = logging.getLogger(__name__)


def load_index(
    index_directory: str,
) -> Tuple[bm25s.BM25, List[Dict[str, Any]]]:
    """Loads the BM25 index and chunk metadata from disk.

    Args:
        index_directory: Directory containing bm25_index/ and
            chunks_metadata.pkl produced by build_and_save_index.

    Returns:
        A tuple (retriever, chunks_metadata) where retriever is the loaded
        BM25 index and chunks_metadata is the list of chunks in the same order
        used during indexing.

    Raises:
        FileNotFoundError: If the index or metadata files are missing.
    """
    bm25_path = Path(index_directory) / "bm25_index"
    chunks_metadata_path = Path(index_directory) / "chunks_metadata.pkl"

    if not bm25_path.exists() or not chunks_metadata_path.exists():
        raise FileNotFoundError(
            f"Index not found in {index_directory} "
            "(missing bm25_index/ or chunks_metadata.pkl)"
        )

    try:
        retriever = bm25s.BM25.load(str(bm25_path), load_corpus=False)
    except Exception as exc:
        logger.error("Failed to load index from %s: %s", index_directory, exc)
        raise

    try:
        with chunks_metadata_path.open("rb") as metadata_file:
            chunks_metadata = pickle.load(metadata_file)
    except Exception as exc:
        logger.error(
            "Failed to load metadata from %s: %s",
            index_directory,
            exc,
        )
        raise

    return retriever, chunks_metadata


def search(
    query: str,
    retriever: bm25s.BM25,
    chunks_metadata: List[Dict[str, Any]],
    k: int,
) -> List[MinimalSource]:
    """Returns the k most relevant sources for a query.

    The query is tokenized with the same logic as the corpus so the code can
    match either a quoted identifier or a reformulated version.

    Args:
        query: Question or search text.
        retriever: BM25 index loaded via load_index.
        chunks_metadata: Chunk metadata in the same order used during indexing.
        k: Number of results to return.

    Returns:
        The k most relevant sources, ordered by relevance. Empty list if the
        query is empty, contains no useful terms, k is 0, or the corpus is
        empty.
    """
    effective_k = min(k, len(chunks_metadata))
    if not query.strip() or effective_k <= 0:
        return []

    query_tokens = tokenize_corpus([query])
    if not query_tokens[0]:
        return []

    retrieval_window = min(
        max(effective_k * 15, effective_k),
        min(len(chunks_metadata), 300),
    )
    results, scores = retriever.retrieve(query_tokens, k=retrieval_window)

    query_terms = set(query_tokens[0])
    reranked_candidates = []
    for idx, score in zip(results[0], scores[0]):
        candidate = chunks_metadata[int(idx)]["metadata"]
        adjusted_score = float(score) + _candidate_bonus(
            candidate, query_terms
        )
        reranked_candidates.append(
            (adjusted_score, float(score), int(idx))
        )

    reranked_candidates.sort(reverse=True)

    sources: List[MinimalSource] = []
    for _adjusted_score, _base_score, idx in reranked_candidates[:effective_k]:
        meta = chunks_metadata[idx]["metadata"]
        sources.append(MinimalSource.model_validate(meta))
    return sources


def _candidate_bonus(
    source: MinimalSource,
    query_terms: set[str],
) -> float:
    """Applies a small heuristic bonus to documentation files.

    The first BM25 pass keeps the main lexical signal. This helper only
    reorders the best candidates to favor Markdown files and docs paths, which
    often match the reference sources in docs datasets.
    """
    path = Path(source.file_path)
    path_text = path.as_posix().lower()

    bonus = 0.0
    if "/docs/" in path_text:
        bonus += 0.80
    if path.suffix == ".md":
        bonus += 0.30
    if any(
        part in {"tests", "examples", "benchmarks", ".buildkite"}
        for part in path.parts
    ):
        bonus -= 0.50

    path_terms = set(tokenize_corpus([source.file_path])[0])
    overlap = len(query_terms & path_terms)
    bonus += min(overlap, 5) * 0.12

    return bonus


def load_dataset_questions(dataset_path: str) -> List[Dict[str, Any]]:
    """Loads and validates questions from a JSON dataset using pydantic.

    Args:
        dataset_path: Path to a JSON file matching the RagDataset schema with a
            "rag_questions" field.

    Returns:
        A list of dictionaries with at least "question_id" and "question".

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the JSON is malformed or does not match the
            RagDataset schema.
    """
    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"The file {dataset_path} does not exist")

    try:
        with open(dataset_path, "r", encoding="utf-8") as file:
            raw_data = json.load(file)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"The file {dataset_path} is malformed: {exc}"
        ) from exc

    try:
        rag_dataset = RagDataset.model_validate(raw_data)
    except pydantic.ValidationError as exc:
        raise ValueError(
            f"The dataset does not match the expected schema: {exc}"
        ) from exc

    return [question.model_dump() for question in rag_dataset.rag_questions]


def search_dataset(
    dataset_path: str,
    k: int,
    save_directory: str,
    index_directory: str = "data/processed",
) -> StudentSearchResults:
    """Runs BM25 search across all questions in a dataset.

    Args:
        dataset_path: Path to the JSON questions file (UnansweredQuestions or
            AnsweredQuestions).
        k: Number of sources to return per question.
        save_directory: Directory where the result JSON is written,
            matching the StudentSearchResults schema.
        index_directory: Directory containing the BM25 index created by
            the index command.

    Returns:
        The StudentSearchResults object, also written to disk in
        save_directory.

    Raises:
        FileNotFoundError: If the dataset or index is missing.
        ValueError: If the dataset is malformed.
    """
    retriever, chunks_metadata = load_index(index_directory)
    questions = load_dataset_questions(dataset_path)

    search_results: List[MinimalSearchResults] = []
    for question in tqdm(questions, desc="BM25 search by question"):
        sources = search(question["question"], retriever, chunks_metadata, k)
        search_results.append(
            MinimalSearchResults(
                question_id=question["question_id"],
                question=question["question"],
                retrieved_sources=sources,
            )
        )

    student_search_results = StudentSearchResults(
        search_results=search_results,
        k=k,
    )

    output_dir = Path(save_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / Path(dataset_path).name
    try:
        with output_path.open("w", encoding="utf-8") as output_file:
            output_file.write(student_search_results.model_dump_json(indent=2))
    except OSError as exc:
        logger.error("Failed to write results to %s: %s", output_path, exc)
        raise

    logger.info("Search results saved in %s", output_path)

    return student_search_results
