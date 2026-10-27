import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Tuple

import pydantic
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models import (
    MinimalAnswer,
    MinimalSource,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from src.retrieval import load_index, search

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "Qwen/Qwen3-0.6B"
MAX_CONTEXT_TOKENS = 3000
MAX_NEW_TOKENS = 300

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about a codebase. "
    "Use ONLY the provided context to answer. If the context does not "
    "contain enough information to answer, say so explicitly rather "
    "than guessing. Be concise and precise."
)


@lru_cache(maxsize=1)
def _load_model_and_tokenizer(
    model_name: str = DEFAULT_MODEL_NAME,
) -> Tuple[Any, Any, str]:
    """Loads the model and tokenizer once using a cache.

    The cache is essential for answer_dataset: without it, the model would be
    reloaded for each question, making batch processing very slow.

    Args:
        model_name: Model identifier on the Hugging Face Hub.

    Returns:
        A tuple of (tokenizer, model, device).

    Raises:
        RuntimeError: If the model or tokenizer cannot be loaded.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading %s on %s...", model_name, device)

    try:
        tokenizer: Any = AutoTokenizer.from_pretrained(model_name)
        model: Any = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32 if device == "cpu" else torch.bfloat16,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load model {model_name}: {exc}"
        ) from exc

    model.to(device)
    model.eval()

    logger.info("Model loaded successfully.")
    return tokenizer, model, device


def _read_source_text(source: MinimalSource) -> str:
    """Re-reads the exact source text from disk.

    Args:
        source: Source whose text content we want to retrieve.

    Returns:
        The text corresponding to the given character range, or an empty string
        if the file is unreadable.
    """
    try:
        content = Path(source.file_path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning(
            "Unable to re-read source %s: %s",
            source.file_path,
            exc,
        )
        return ""

    start = source.first_character_index
    end = source.last_character_index
    return content[start:end + 1]


def _build_context(
    sources: List[MinimalSource], tokenizer: Any, max_context_tokens: int
) -> str:
    """Builds the context block from the retrieved sources.

    Sources are added in relevance order until the allowed token budget is
    reached. It is better to have a truncated but coherent context than an
    oversized context that exceeds the model window.

    Args:
        sources: Retrieved sources, sorted by decreasing relevance.
        tokenizer: Tokenizer used to count tokens.
        max_context_tokens: Maximum token budget for the context.

    Returns:
        The context text made from the concatenated sources. Empty string if no
        source could be read.
    """
    blocks = []
    total_tokens = 0

    for source in sources:
        text = _read_source_text(source)
        if not text:
            continue

        block = (
            f"# Source: {source.file_path} "
            f"[{source.first_character_index}:{source.last_character_index}]\n"
            f"{text}\n"
        )
        block_tokens = len(tokenizer.encode(block))

        if total_tokens + block_tokens > max_context_tokens:
            break

        blocks.append(block)
        total_tokens += block_tokens

    return "\n".join(blocks)


def generate_answer_from_sources(
    question: str,
    sources: List[MinimalSource],
    model_name: str = DEFAULT_MODEL_NAME,
) -> str:
    """Generates an answer from a question and already-retrieved sources.

    Args:
        question: User question.
        sources: Sources to use as context.
        model_name: Model used for generation.

    Returns:
        The generated answer. Empty string if the question is empty.
    """
    if not question.strip():
        logger.warning("Empty question received; no generation performed.")
        return ""

    tokenizer, model, device = _load_model_and_tokenizer(model_name)

    context = _build_context(sources, tokenizer, MAX_CONTEXT_TOKENS)
    if not context:
        logger.warning(
            "No context available for the question '%s': the retrieved "
            "sources could not be read.",
            question,
        )

    user_content = (
        f"Context:\n{context}\n\nQuestion: {question}"
        if context
        else f"Question: {question}\n\n(No context could be retrieved.)"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    generated_tokens = output_ids[0][
        inputs["input_ids"].shape[1]:
    ]
    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    return str(answer)


def answer(
    query: str,
    k: int,
    index_directory: str = "data/processed",
) -> MinimalAnswer:
    """Answers a single question by searching and then generating.

    Args:
        query: Question to answer.
        k: Number of sources to retrieve before generation.
        index_directory: Directory containing the BM25 index.

    Returns:
        A MinimalAnswer containing the question, retrieved sources, and answer.

    Raises:
        FileNotFoundError: If the index is not found.
    """
    retriever, chunks_metadata = load_index(index_directory)
    sources = search(query, retriever, chunks_metadata, k)
    generated_answer = generate_answer_from_sources(query, sources)

    return MinimalAnswer(
        question_id="single-query",
        question=query,
        retrieved_sources=sources,
        answer=generated_answer,
    )


def answer_dataset(
    student_search_results_path: str,
    save_directory: str,
) -> StudentSearchResultsAndAnswer:
    """Generates an answer for each question from a StudentSearchResults file.

    This does not rerun the search; it reuses the sources already retrieved by
    a previous search_dataset run.

    Args:
        student_search_results_path: Path to the JSON file matching the
            StudentSearchResults schema produced by search_dataset.
        save_directory: Directory where the JSON result is written,
            matching the StudentSearchResultsAndAnswer schema.

    Returns:
        The generated StudentSearchResultsAndAnswer object, also written
        to disk under save_directory.

    Raises:
        FileNotFoundError: If the input file is missing.
        ValueError: If the input file is malformed.
    """
    path = Path(student_search_results_path)
    if not path.exists():
        raise FileNotFoundError(
            f"The file {student_search_results_path} does not exist"
        )

    try:
        raw_data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"The file {student_search_results_path} is malformed: {exc}"
        ) from exc

    try:
        student_results = StudentSearchResults.model_validate(raw_data)
    except pydantic.ValidationError as exc:
        raise ValueError(
            "The file does not respect the StudentSearchResults "
            f"schema: {exc}"
        ) from exc

    answers: List[MinimalAnswer] = []
    for result in tqdm(
        student_results.search_results,
        desc="Generating answers",
    ):
        generated_answer = generate_answer_from_sources(
            result.question,
            result.retrieved_sources,
        )
        answers.append(
            MinimalAnswer(
                question_id=result.question_id,
                question=result.question,
                retrieved_sources=result.retrieved_sources,
                answer=generated_answer,
            )
        )

    student_results_and_answer = StudentSearchResultsAndAnswer(
        search_results=answers,
        k=student_results.k,
    )

    output_dir = Path(save_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / Path(student_search_results_path).name

    try:
        output_path.write_text(
            student_results_and_answer.model_dump_json(indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.error(
            "Failed to write results to %s: %s",
            output_path,
            exc,
        )
        raise

    logger.info("Generated results saved in %s", output_path)

    return student_results_and_answer
