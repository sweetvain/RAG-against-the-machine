"""Lecture, Chunking et Indexation"""

"""lire les données, de les découper proprement, puis de créer un index de recherche."""

"""Logique de recherche BM25"""

import logging
import pickle
import bm25s

from pathlib import Path
from typing import List, Dict, Tuple, Any
from src.models import MinimalSource
from tqdm import tqdm

logger = logging.getLogger(__name__)


def read_repository(repo_path: str, extensions: Tuple[str, ...] = (".py", ".md")) -> List[Path]:
    """Parcourt le depot et retourne la liste des fichiers valides"""
    path = Path(repo_path)
    if not path.exists() or not path.is_dir():
        raise ValueError(f"The path {repo_path} is invalid or does not exist")

    valid_files = [f for f in path.rglob("*") if f.suffix in extensions and f.is_file()]
    return valid_files


def chunk_section(file_path: Path, max_chunk_size: int = 2000) -> List[Dict]:
    try:
        content = file_path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        print(f"Impossible de décoder le fichier {file_path} avec UTF-8. Ignoré.")
        return []
    except Exception as e:
        print(f"Erreur de lecture du fichier {file_path}: {e}. Ignoré.")
        return []

    lines = content.splitlines(keepends=True)  # Garder les sauts de ligne pour les indices

    if file_path.suffix == ".md":
        return chunk_md_file(file_path, lines, max_chunk_size=max_chunk_size)

    elif file_path.suffix == ".py":
        return chunk_python_file(file_path, lines, max_chunk_size=max_chunk_size)

    else:
        # Fallback pour les fichiers non gérés spécifiquement
        return chunk_other_file(file_path, lines, max_chunk_size=max_chunk_size)


def chunk_md_file(file_path: Path, lines, max_chunk_size: int = 2000) -> List[Dict]:
    """Découpe un fichier Markdown en chunks par en-têtes (#, ##...).

    Args:
        file_path: Chemin du fichier source.
        lines: Lignes du fichier (splitlines(keepends=True)).
        max_chunk_size: Taille maximale en caractères d'un chunk.

    Returns:
        Liste de chunks {"content": str, "metadata": dict}.
    """
    chunks = []
    current_chunk_content = ""
    chunk_start_index = 0
    current_char_index = 0

    for line in lines:
        if line.strip().startswith("#") and current_chunk_content:
            chunks.append({
                "content": current_chunk_content.strip(),
                "metadata": MinimalSource(
                    file_path=str(file_path),
                    first_character_index=chunk_start_index,
                    last_character_index=chunk_start_index + len(current_chunk_content) - 1
                ).model_dump()
            })
            current_chunk_content = ""
            chunk_start_index = current_char_index

        current_chunk_content += line

        if len(current_chunk_content) > max_chunk_size:
            split_point = current_chunk_content.rfind('\n', 0, max_chunk_size)
            if split_point == -1:
                split_point = max_chunk_size

            chunks.append({
                "content": current_chunk_content[:split_point].strip(),
                "metadata": MinimalSource(
                    file_path=str(file_path),
                    first_character_index=chunk_start_index,
                    last_character_index=chunk_start_index + split_point - 1
                ).model_dump()
            })
            current_chunk_content = current_chunk_content[split_point:]
            chunk_start_index += split_point

        current_char_index += len(line)

    if current_chunk_content:
        chunks.append({
            "content": current_chunk_content.strip(),
            "metadata": MinimalSource(
                file_path=str(file_path),
                first_character_index=chunk_start_index,
                last_character_index=chunk_start_index + len(current_chunk_content) - 1
            ).model_dump()
        })

    return chunks


def chunk_python_file(file_path: Path, lines, max_chunk_size: int = 2000) -> List[Dict]:
    """Découpe un fichier Python en chunks par définitions (def/class).

    Args:
        file_path: Chemin du fichier source.
        lines: Lignes du fichier (splitlines(keepends=True)).
        max_chunk_size: Taille maximale en caractères d'un chunk.

    Returns:
        Liste de chunks {"content": str, "metadata": dict}.
    """
    chunks = []
    current_chunk_content = ""
    chunk_start_index = 0
    current_char_index = 0

    for line in lines:
        stripped_line = line.strip()
        if (stripped_line.startswith("def ") or stripped_line.startswith("class ")) and current_chunk_content:
            chunks.append({
                "content": current_chunk_content.strip(),
                "metadata": MinimalSource(
                    file_path=str(file_path),
                    first_character_index=chunk_start_index,
                    last_character_index=chunk_start_index + len(current_chunk_content) - 1
                ).model_dump()
            })
            current_chunk_content = ""
            chunk_start_index = current_char_index

        current_chunk_content += line

        if len(current_chunk_content) > max_chunk_size:
            split_point = current_chunk_content.rfind('\n', 0, max_chunk_size)
            if split_point == -1:
                split_point = max_chunk_size

            chunks.append({
                "content": current_chunk_content[:split_point].strip(),
                "metadata": MinimalSource(
                    file_path=str(file_path),
                    first_character_index=chunk_start_index,
                    last_character_index=chunk_start_index + split_point - 1
                ).model_dump()
            })
            current_chunk_content = current_chunk_content[split_point:]
            chunk_start_index += split_point

        current_char_index += len(line)

    if current_chunk_content:
        chunks.append({
            "content": current_chunk_content.strip(),
            "metadata": MinimalSource(
                file_path=str(file_path),
                first_character_index=chunk_start_index,
                last_character_index=chunk_start_index + len(current_chunk_content) - 1
            ).model_dump()
        })

    return chunks


def chunk_other_file(file_path: Path, lines, max_chunk_size: int = 2000) -> List[Dict]:
    """Découpe générique par taille, pour les extensions sans
    stratégie dédiée (aucune n'est attendue en pratique puisque
    read_repository ne retient que .py et .md par défaut, mais ce
    filet de sécurité évite un crash si chunk_section est appelée
    directement sur un fichier d'un autre type).

    Args:
        file_path: Chemin du fichier source.
        lines: Lignes du fichier (splitlines(keepends=True)).
        max_chunk_size: Taille maximale en caractères d'un chunk.

    Returns:
        Liste de chunks {"content": str, "metadata": dict}.
    """
    chunks = []
    current_chunk_content = ""
    chunk_start_index = 0
    current_char_index = 0

    for line in lines:
        if len(current_chunk_content) + len(line) > max_chunk_size and current_chunk_content:
            chunks.append({
                "content": current_chunk_content.strip(),
                "metadata": MinimalSource(
                    file_path=str(file_path),
                    first_character_index=chunk_start_index,
                    last_character_index=chunk_start_index + len(current_chunk_content) - 1
                ).model_dump()
            })
            current_chunk_content = ""
            chunk_start_index = current_char_index

        current_chunk_content += line
        current_char_index += len(line)

    if current_chunk_content:
        chunks.append({
            "content": current_chunk_content.strip(),
            "metadata": MinimalSource(
                file_path=str(file_path),
                first_character_index=chunk_start_index,
                last_character_index=chunk_start_index + len(current_chunk_content) - 1
            ).model_dump()
        })

    return chunks


def tokenize_corpus(texts: List[str]) -> Any:
    return bm25s.tokenize(texts, stopwords="en", show_progress=False)


def build_and_save_index(
    all_chunks: List[Dict[str, Any]],
    save_directory: str,
) -> None:
    if not all_chunks:
        logger.warning("Aucun chunk a indexer, index vide !")
        return

    logger.info("Indexation de %d chunks en cours...", len(all_chunks))

    sources: List[MinimalSource] = [
        MinimalSource.model_validate(chunk["metadata"])
        for chunk in tqdm(all_chunks, desc="Validation des metadatas")
    ]
    corpus_texts = [chunk["content"] for chunk in all_chunks]

    corpus_tokenized = tokenize_corpus(corpus_texts)

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
        logger.error(
            "Echec de la save de l'index dans %s: %s", output_dir, exc
        )
        raise

    logger.info("Indexation terminee et save dans %s", output_dir)


def run_ingestion(
    repo_path: str,
    save_directory: str,
    max_chunk_size: int = 2000,
    extensions: Tuple[str, ...] = (".py", ".md"),
) -> List[Dict[str, Any]]:
    files = read_repository(repo_path, extensions=extensions)
    logger.info("Ingestion de %d fichiers depuis %s", len(files), repo_path)

    all_chunks: List[Dict[str, Any]] = []
    skipped_files: List[Path] = []

    for file_path in tqdm(files, desc="Chunking des fichiers"):
        try:
            chunks = chunk_section(file_path, max_chunk_size=max_chunk_size)

        except Exception as exc:
            logger.warning("Chunking echoue pour %s: %s", file_path, exc)
            skipped_files.append(file_path)
            continue

        if not chunks:
            skipped_files.append(file_path)
            continue

        all_chunks.extend(chunks)

    logger.info(
        "Chunking termine: %d chunks produits, %d fichiers sans chunk",
        len(all_chunks),
        len(skipped_files),
    )

    build_and_save_index(all_chunks, save_directory)

    return all_chunks