import logging
import pickle
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import bm25s
from bm25s.stopwords import STOPWORDS_EN
from tqdm import tqdm

from src.models import MinimalSource

logger = logging.getLogger(__name__)

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PARTS = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
_STOPWORDS = frozenset(STOPWORDS_EN)


def _normalize_token(token: str) -> str:
    """Normalizes a token to a more stable form for BM25 matching.

    The goal is not perfect stemming, but to merge simple variants such as
    `implement` / `implements`, `support` / `supported`, and `rerank` /
    `reranking` to improve recall without adding a heavy dependency.
    """
    if len(token) > 5 and token.endswith("ing"):
        base = token[:-3]
        if len(base) > 3 and base[-1] == base[-2]:
            base = base[:-1]
        return base

    if len(token) > 4 and token.endswith("ied"):
        return f"{token[:-3]}y"

    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]

    if len(token) > 4 and token.endswith("es"):
        return token[:-2]

    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]

    return token


def read_repository(
    repo_path: str,
    extensions: Tuple[str, ...] = (".py", ".md", ".txt"),
) -> List[Path]:
    """Walks the repository and returns the list of valid files.

    Args:
        repo_path: Root directory to scan.
        extensions: File extensions to keep.

    Returns:
        List of files whose extension is allowed.

    Raises:
        ValueError: If the provided path does not exist or is not a directory.
    """
    path = Path(repo_path)
    if not path.exists() or not path.is_dir():
        raise ValueError(f"The path {repo_path} is invalid or does not exist")

    valid_files = [
        f for f in path.rglob("*") if f.suffix in extensions and f.is_file()
    ]
    return valid_files


def _make_chunk(
    file_path: Path,
    text: str,
    start: int,
    end: int,
) -> Dict[str, Any]:
    """Builds a chunk covering text[start:end] (end exclusive).

    The content is the exact slice of the file, so that the stored character
    range always matches the content.
    """
    return {
        "content": text[start:end],
        "metadata": MinimalSource(
            file_path=file_path.as_posix(),
            first_character_index=start,
            last_character_index=end - 1,
        ).model_dump(),
    }


def _split_by_boundaries(
    file_path: Path,
    lines: List[str],
    is_boundary: Callable[[str], bool],
    max_chunk_size: int,
) -> List[Dict[str, Any]]:
    """Splits lines into chunks starting at boundary lines.

    A chunk is closed when a boundary line is met, and whenever it would exceed
    max_chunk_size (cut on the last newline when possible, hard cut otherwise).
    No chunk is ever longer than max_chunk_size.

    Args:
        file_path: Source file path.
        lines: File lines from splitlines(keepends=True).
        is_boundary: Tells whether a line starts a new logical block.
        max_chunk_size: Maximum number of characters in one chunk.

    Returns:
        A list of chunks with "content" and "metadata" entries.
    """
    if max_chunk_size < 1:
        raise ValueError("max_chunk_size must be at least 1")

    text = "".join(lines)
    chunks: List[Dict[str, Any]] = []
    start = 0
    position = 0

    def flush(end: int) -> None:
        if end > start and text[start:end].strip():
            chunks.append(_make_chunk(file_path, text, start, end))

    for line in lines:
        starts_block = is_boundary(line)
        if starts_block and position > start:
            flush(position)
            start = position

        position += len(line)

        while position - start > max_chunk_size:
            window_end = start + max_chunk_size
            cut = text.rfind("\n", start, window_end) + 1
            if cut <= start:
                cut = window_end
            flush(cut)
            start = cut

    flush(position)
    return chunks


def chunk_section(
    file_path: Path,
    max_chunk_size: int = 2000,
) -> List[Dict[str, Any]]:
    """Chooses the chunking strategy based on the file extension.

    Args:
        file_path: File to split into chunks.
        max_chunk_size: Maximum number of characters in one chunk.

    Returns:
        A list of chunks as dictionaries with "content" and "metadata". Empty
        list if the file is unreadable.
    """
    try:
        with file_path.open("r", encoding="utf-8", newline="") as handle:
            content = handle.read()
    except UnicodeDecodeError:
        logger.warning("Unable to decode %s with UTF-8. Ignoring.", file_path)
        return []
    except OSError as exc:
        logger.warning("Error reading file %s: %s. Ignoring.", file_path, exc)
        return []

    lines = content.splitlines(keepends=True)

    if file_path.suffix == ".md":
        return chunk_md_file(file_path, lines, max_chunk_size=max_chunk_size)

    if file_path.suffix == ".py":
        return chunk_python_file(
            file_path,
            lines,
            max_chunk_size=max_chunk_size,
        )

    return chunk_other_file(
        file_path,
        lines,
        max_chunk_size=max_chunk_size,
    )


def chunk_md_file(
    file_path: Path,
    lines: List[str],
    max_chunk_size: int = 2000,
) -> List[Dict[str, Any]]:
    """Splits a Markdown file into chunks by headings (#, ##, ...).

    Headings inside fenced code blocks are not treated as boundaries.

    Args:
        file_path: Source file path.
        lines: File lines from splitlines(keepends=True).
        max_chunk_size: Maximum number of characters in one chunk.

    Returns:
        A list of chunks with "content" and "metadata" entries.
    """
    in_fence = False

    def is_heading(line: str) -> bool:
        nonlocal in_fence
        stripped = line.lstrip()
        if stripped.startswith(("```", "~~~")):
            in_fence = not in_fence
            return False
        return not in_fence and stripped.startswith("#")

    return _split_by_boundaries(file_path, lines, is_heading, max_chunk_size)


def chunk_python_file(
    file_path: Path,
    lines: List[str],
    max_chunk_size: int = 2000,
) -> List[Dict[str, Any]]:
    """Splits a Python file into chunks by definitions (def/class).

    Decorators are kept with the definition they decorate.

    Args:
        file_path: Source file path.
        lines: File lines from splitlines(keepends=True).
        max_chunk_size: Maximum number of characters in one chunk.

    Returns:
        A list of chunks with "content" and "metadata" entries.
    """
    previous_is_decorator = False

    def is_definition(line: str) -> bool:
        nonlocal previous_is_decorator
        stripped = line.strip()
        is_decorator = stripped.startswith("@")
        starts_block = stripped.startswith(
            ("def ", "class ", "async def ")
        ) or is_decorator
        boundary = starts_block and not previous_is_decorator
        if stripped:
            previous_is_decorator = is_decorator
        return boundary

    return _split_by_boundaries(
        file_path, lines, is_definition, max_chunk_size
    )


def chunk_other_file(
    file_path: Path,
    lines: List[str],
    max_chunk_size: int = 2000,
) -> List[Dict[str, Any]]:
    """Generic size-based splitting for other text files (e.g. .txt).

    Args:
        file_path: Source file path.
        lines: File lines from splitlines(keepends=True).
        max_chunk_size: Maximum number of characters in one chunk.

    Returns:
        A list of chunks with "content" and "metadata" entries.
    """
    return _split_by_boundaries(
        file_path, lines, lambda _line: False, max_chunk_size
    )


def tokenize_text(text: str) -> List[str]:
    """Tokenizes a text while preserving identifiers and their parts.

    Each identifier is kept as-is (lowercased) to recover an exact citation,
    and it is also split into subwords (snake_case and camelCase) to recover a
    reformulated query. Stopwords are removed.

    Args:
        text: Text to tokenize (chunk content or query).

    Returns:
        A list of lowercase tokens.
    """
    tokens: List[str] = []
    for ident in _IDENT.findall(text):
        parts = [part.lower() for part in _PARTS.findall(ident)]
        for token in [ident.lower(), *parts]:
            if token in _STOPWORDS:
                continue

            tokens.append(token)

            normalized = _normalize_token(token)
            if normalized != token and normalized not in _STOPWORDS:
                tokens.append(normalized)
    return tokens


def tokenize_corpus(texts: List[str]) -> List[List[str]]:
    """Tokenizes a list of texts using tokenize_text.

    The same function is used for indexing and queries, which guarantees that
    both sides use the same vocabulary.

    Args:
        texts: Texts to tokenize.

    Returns:
        A list of token lists, one per input text.
    """
    return [tokenize_text(text) for text in texts]


def build_and_save_index(
    all_chunks: List[Dict[str, Any]],
    save_directory: str,
) -> None:
    """Builds and saves the BM25 index with chunk metadata.

    The file path is added to the indexed text so that path terms (module name,
    folder name) participate in retrieval. The stored metadata remains the
    original one.

    Args:
        all_chunks: Chunks in the form {"content": str, "metadata": dict}.
        save_directory: Directory where bm25_index/ and chunks_metadata.pkl are
            written.

    Raises:
        OSError: If writing to disk fails.
    """
    if not all_chunks:
        logger.warning("No chunk to index, empty index.")
        return

    logger.info("Indexing %d chunks...", len(all_chunks))

    sources: List[MinimalSource] = [
        MinimalSource.model_validate(chunk["metadata"])
        for chunk in tqdm(all_chunks, desc="Validating metadata")
    ]
    corpus_texts = [chunk["content"] for chunk in all_chunks]
    index_texts = []
    for text, source in zip(corpus_texts, sources):
        path = Path(source.file_path)

        if path.suffix == ".md" and "/docs/" in path.as_posix():
            prefix = "\n".join(
                [
                    source.file_path,
                    source.file_path,
                    path.stem.replace("_", " "),
                    path.stem.replace("_", " "),
                    path.parent.name.replace("_", " "),
                ]
            )
        else:
            prefix = source.file_path

        index_texts.append(f"{prefix}\n{text}")

    corpus_tokenized = tokenize_corpus(index_texts)

    retriever = bm25s.BM25()
    retriever.index(corpus_tokenized, show_progress=True)

    output_dir = Path(save_directory)

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        retriever.save(str(output_dir / "bm25_index"))

        metadata_path = output_dir / "chunks_metadata.pkl"
        with metadata_path.open("wb") as metadata_file:
            pickle.dump(
                [
                    {"content": text, "metadata": source}
                    for text, source in zip(corpus_texts, sources)
                ],
                metadata_file,
            )

    except OSError as exc:
        logger.error("Failed to save the index in %s: %s", output_dir, exc)
        raise

    logger.info("Indexing completed and saved in %s", output_dir)


def run_ingestion(
    repo_path: str,
    save_directory: str,
    max_chunk_size: int = 2000,
    extensions: Tuple[str, ...] = (".py", ".md", ".txt"),
) -> List[Dict[str, Any]]:
    """Reads the repository, splits files, and builds the index.

    Args:
        repo_path: Root directory to index (for example: data/raw).
        save_directory: Directory where the index is saved (for example:
            data/processed).
        max_chunk_size: Maximum number of characters in a chunk.
        extensions: File extensions to index.

    Returns:
        The complete list of produced chunks.

    Raises:
        ValueError: If repo_path is invalid.
        OSError: If writing the index fails.
    """
    files = read_repository(repo_path, extensions=extensions)
    logger.info("Ingesting %d files from %s", len(files), repo_path)

    all_chunks: List[Dict[str, Any]] = []
    skipped_files: List[Path] = []

    for file_path in tqdm(files, desc="Chunking files"):
        try:
            chunks = chunk_section(file_path, max_chunk_size=max_chunk_size)

        except Exception as exc:
            logger.warning("Chunking failed for %s: %s", file_path, exc)
            skipped_files.append(file_path)
            continue

        if not chunks:
            skipped_files.append(file_path)
            continue

        all_chunks.extend(chunks)

    logger.info(
        "Chunking complete: %d chunks produced, %d files without chunk",
        len(all_chunks),
        len(skipped_files),
    )

    build_and_save_index(all_chunks, save_directory)

    return all_chunks
