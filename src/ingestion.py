import logging
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
    extensions: Tuple[str, ...] = (".py", ".md"),
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
        content = file_path.read_text(encoding="utf-8")
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

    Args:
        file_path: Source file path.
        lines: File lines from splitlines(keepends=True).
        max_chunk_size: Maximum number of characters in one chunk.

    Returns:
        A list of chunks with "content" and "metadata" entries.
    """
    chunks: List[Dict[str, Any]] = []
    current_chunk_content = ""
    chunk_start_index = 0
    current_char_index = 0

    for line in lines:
        if line.strip().startswith("#") and current_chunk_content:
            chunks.append(
                {
                    "content": current_chunk_content.strip(),
                    "metadata": MinimalSource(
                        file_path=str(file_path),
                        first_character_index=chunk_start_index,
                        last_character_index=(
                            chunk_start_index + len(current_chunk_content) - 1
                        ),
                    ).model_dump(),
                }
            )
            current_chunk_content = ""
            chunk_start_index = current_char_index

        current_chunk_content += line

        if len(current_chunk_content) > max_chunk_size:
            split_point = current_chunk_content.rfind("\n", 0, max_chunk_size)
            if split_point == -1:
                split_point = max_chunk_size

            chunks.append(
                {
                    "content": current_chunk_content[:split_point].strip(),
                    "metadata": MinimalSource(
                        file_path=str(file_path),
                        first_character_index=chunk_start_index,
                        last_character_index=(
                            chunk_start_index + split_point - 1
                        ),
                    ).model_dump(),
                }
            )
            current_chunk_content = current_chunk_content[split_point:]
            chunk_start_index += split_point

        current_char_index += len(line)

    if current_chunk_content:
        chunks.append(
            {
                "content": current_chunk_content.strip(),
                "metadata": MinimalSource(
                    file_path=str(file_path),
                    first_character_index=chunk_start_index,
                    last_character_index=(
                        chunk_start_index + len(current_chunk_content) - 1
                    ),
                ).model_dump(),
            }
        )

    return chunks


def chunk_python_file(
    file_path: Path,
    lines: List[str],
    max_chunk_size: int = 2000,
) -> List[Dict[str, Any]]:
    """Splits a Python file into chunks by definitions (def/class).

    Args:
        file_path: Source file path.
        lines: File lines from splitlines(keepends=True).
        max_chunk_size: Maximum number of characters in one chunk.

    Returns:
        A list of chunks with "content" and "metadata" entries.
    """
    chunks: List[Dict[str, Any]] = []
    current_chunk_content = ""
    chunk_start_index = 0
    current_char_index = 0

    for line in lines:
        stripped_line = line.strip()
        is_definition = (
            stripped_line.startswith("def ")
            or stripped_line.startswith("class ")
        )
        if is_definition and current_chunk_content:
            chunks.append(
                {
                    "content": current_chunk_content.strip(),
                    "metadata": MinimalSource(
                        file_path=str(file_path),
                        first_character_index=chunk_start_index,
                        last_character_index=(
                            chunk_start_index + len(current_chunk_content) - 1
                        ),
                    ).model_dump(),
                }
            )
            current_chunk_content = ""
            chunk_start_index = current_char_index

        current_chunk_content += line

        if len(current_chunk_content) > max_chunk_size:
            split_point = current_chunk_content.rfind("\n", 0, max_chunk_size)
            if split_point == -1:
                split_point = max_chunk_size

            chunks.append(
                {
                    "content": current_chunk_content[:split_point].strip(),
                    "metadata": MinimalSource(
                        file_path=str(file_path),
                        first_character_index=chunk_start_index,
                        last_character_index=(
                            chunk_start_index + split_point - 1
                        ),
                    ).model_dump(),
                }
            )
            current_chunk_content = current_chunk_content[split_point:]
            chunk_start_index += split_point

        current_char_index += len(line)

    if current_chunk_content:
        chunks.append(
            {
                "content": current_chunk_content.strip(),
                "metadata": MinimalSource(
                    file_path=str(file_path),
                    first_character_index=chunk_start_index,
                    last_character_index=(
                        chunk_start_index + len(current_chunk_content) - 1
                    ),
                ).model_dump(),
            }
        )

    return chunks


def chunk_other_file(
    file_path: Path,
    lines: List[str],
    max_chunk_size: int = 2000,
) -> List[Dict[str, Any]]:
    """Generic size-based splitting for other extensions.

    No extension other than .py and .md is read by default, but this safety
    fallback prevents a crash if chunk_section is called on another file type.

    Args:
        file_path: Source file path.
        lines: File lines from splitlines(keepends=True).
        max_chunk_size: Maximum number of characters in one chunk.

    Returns:
        A list of chunks with "content" and "metadata" entries.
    """
    chunks: List[Dict[str, Any]] = []
    current_chunk_content = ""
    chunk_start_index = 0
    current_char_index = 0

    for line in lines:
        too_long = len(current_chunk_content) + len(line) > max_chunk_size
        if too_long and current_chunk_content:
            chunks.append(
                {
                    "content": current_chunk_content.strip(),
                    "metadata": MinimalSource(
                        file_path=str(file_path),
                        first_character_index=chunk_start_index,
                        last_character_index=(
                            chunk_start_index + len(current_chunk_content) - 1
                        ),
                    ).model_dump(),
                }
            )
            current_chunk_content = ""
            chunk_start_index = current_char_index

        current_chunk_content += line
        current_char_index += len(line)

    if current_chunk_content:
        chunks.append(
            {
                "content": current_chunk_content.strip(),
                "metadata": MinimalSource(
                    file_path=str(file_path),
                    first_character_index=chunk_start_index,
                    last_character_index=(
                        chunk_start_index + len(current_chunk_content) - 1
                    ),
                ).model_dump(),
            }
        )

    return chunks


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
    extensions: Tuple[str, ...] = (".py", ".md"),
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
