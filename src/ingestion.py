"""Lecture, chunking et indexation BM25 du depot.

Ce module lit les fichiers du depot, les decoupe en chunks, puis construit
et sauvegarde un index de recherche BM25.
"""

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

# Un identifiant: lettres, chiffres et "_" (ex: get_kv_cache_manager).
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# Sous-mots d'un identifiant: KVCacheManager -> KV, Cache, Manager.
_PARTS = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
_STOPWORDS = frozenset(STOPWORDS_EN)


def _normalize_token(token: str) -> str:
    """Ramene un mot vers une forme plus stable pour la recherche BM25.

    L'objectif n'est pas de faire un stemming linguistique parfait, mais de
    rapprocher des variantes simples comme `implement` / `implements`,
    `support` / `supported` ou `rerank` / `reranking` afin d'augmenter le
    rappel sans introduire de dépendance lourde.
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
    """Parcourt le depot et retourne la liste des fichiers valides.

    Args:
        repo_path: Racine du depot a parcourir.
        extensions: Extensions de fichiers a conserver.

    Returns:
        Liste des fichiers dont l'extension est autorisee.

    Raises:
        ValueError: Si le chemin n'existe pas ou n'est pas un dossier.
    """
    path = Path(repo_path)
    if not path.exists() or not path.is_dir():
        raise ValueError(
            f"The path {repo_path} is invalid or does not exist"
        )

    valid_files = [
        f for f in path.rglob("*")
        if f.suffix in extensions and f.is_file()
    ]
    return valid_files


def chunk_section(
    file_path: Path,
    max_chunk_size: int = 2000,
) -> List[Dict[str, Any]]:
    """Choisit la strategie de chunking selon l'extension du fichier.

    Args:
        file_path: Chemin du fichier a decouper.
        max_chunk_size: Taille maximale en caracteres d'un chunk.

    Returns:
        Liste de chunks {"content": str, "metadata": dict}. Liste vide
        si le fichier est illisible.
    """
    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning(
            "Impossible de decoder %s avec UTF-8. Ignore.", file_path
        )
        return []
    except OSError as exc:
        logger.warning(
            "Erreur de lecture du fichier %s: %s. Ignore.", file_path, exc
        )
        return []

    # On garde les sauts de ligne pour que les indices restent exacts.
    lines = content.splitlines(keepends=True)

    if file_path.suffix == ".md":
        return chunk_md_file(
            file_path, lines, max_chunk_size=max_chunk_size
        )

    if file_path.suffix == ".py":
        return chunk_python_file(
            file_path, lines, max_chunk_size=max_chunk_size
        )

    # Fallback pour les fichiers non geres specifiquement.
    return chunk_other_file(file_path, lines, max_chunk_size=max_chunk_size)


def chunk_md_file(
    file_path: Path,
    lines: List[str],
    max_chunk_size: int = 2000,
) -> List[Dict[str, Any]]:
    """Decoupe un fichier Markdown en chunks par en-tetes (#, ##...).

    Args:
        file_path: Chemin du fichier source.
        lines: Lignes du fichier (splitlines(keepends=True)).
        max_chunk_size: Taille maximale en caracteres d'un chunk.

    Returns:
        Liste de chunks {"content": str, "metadata": dict}.
    """
    chunks: List[Dict[str, Any]] = []
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
                    last_character_index=(
                        chunk_start_index + len(current_chunk_content) - 1
                    ),
                ).model_dump(),
            })
            current_chunk_content = ""
            chunk_start_index = current_char_index

        current_chunk_content += line

        if len(current_chunk_content) > max_chunk_size:
            split_point = current_chunk_content.rfind(
                "\n", 0, max_chunk_size
            )
            if split_point == -1:
                split_point = max_chunk_size

            chunks.append({
                "content": current_chunk_content[:split_point].strip(),
                "metadata": MinimalSource(
                    file_path=str(file_path),
                    first_character_index=chunk_start_index,
                    last_character_index=chunk_start_index + split_point - 1,
                ).model_dump(),
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
                last_character_index=(
                    chunk_start_index + len(current_chunk_content) - 1
                ),
            ).model_dump(),
        })

    return chunks


def chunk_python_file(
    file_path: Path,
    lines: List[str],
    max_chunk_size: int = 2000,
) -> List[Dict[str, Any]]:
    """Decoupe un fichier Python en chunks par definitions (def/class).

    Args:
        file_path: Chemin du fichier source.
        lines: Lignes du fichier (splitlines(keepends=True)).
        max_chunk_size: Taille maximale en caracteres d'un chunk.

    Returns:
        Liste de chunks {"content": str, "metadata": dict}.
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
            chunks.append({
                "content": current_chunk_content.strip(),
                "metadata": MinimalSource(
                    file_path=str(file_path),
                    first_character_index=chunk_start_index,
                    last_character_index=(
                        chunk_start_index + len(current_chunk_content) - 1
                    ),
                ).model_dump(),
            })
            current_chunk_content = ""
            chunk_start_index = current_char_index

        current_chunk_content += line

        if len(current_chunk_content) > max_chunk_size:
            split_point = current_chunk_content.rfind(
                "\n", 0, max_chunk_size
            )
            if split_point == -1:
                split_point = max_chunk_size

            chunks.append({
                "content": current_chunk_content[:split_point].strip(),
                "metadata": MinimalSource(
                    file_path=str(file_path),
                    first_character_index=chunk_start_index,
                    last_character_index=chunk_start_index + split_point - 1,
                ).model_dump(),
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
                last_character_index=(
                    chunk_start_index + len(current_chunk_content) - 1
                ),
            ).model_dump(),
        })

    return chunks


def chunk_other_file(
    file_path: Path,
    lines: List[str],
    max_chunk_size: int = 2000,
) -> List[Dict[str, Any]]:
    """Decoupe generique par taille, pour les autres extensions.

    Aucune extension autre que .py et .md n'est lue par defaut, mais ce
    filet de securite evite un crash si chunk_section est appelee sur un
    fichier d'un autre type.

    Args:
        file_path: Chemin du fichier source.
        lines: Lignes du fichier (splitlines(keepends=True)).
        max_chunk_size: Taille maximale en caracteres d'un chunk.

    Returns:
        Liste de chunks {"content": str, "metadata": dict}.
    """
    chunks: List[Dict[str, Any]] = []
    current_chunk_content = ""
    chunk_start_index = 0
    current_char_index = 0

    for line in lines:
        too_long = len(current_chunk_content) + len(line) > max_chunk_size
        if too_long and current_chunk_content:
            chunks.append({
                "content": current_chunk_content.strip(),
                "metadata": MinimalSource(
                    file_path=str(file_path),
                    first_character_index=chunk_start_index,
                    last_character_index=(
                        chunk_start_index + len(current_chunk_content) - 1
                    ),
                ).model_dump(),
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
                last_character_index=(
                    chunk_start_index + len(current_chunk_content) - 1
                ),
            ).model_dump(),
        })

    return chunks


def tokenize_text(text: str) -> List[str]:
    """Tokenise un texte en gardant les identifiants entiers et leurs parties.

    Chaque identifiant est conserve tel quel (en minuscules) pour
    retrouver une citation exacte, et il est aussi decoupe en sous-mots
    (snake_case et camelCase) pour retrouver une question reformulee.
    Les mots vides anglais sont retires.

    Args:
        text: Texte a tokeniser (contenu d'un chunk ou requete).

    Returns:
        Liste de tokens en minuscules.
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
    """Tokenise une liste de textes avec tokenize_text.

    La meme fonction sert a l'indexation et a la requete, ce qui garantit
    que les deux cotes utilisent le meme vocabulaire.

    Args:
        texts: Textes a tokeniser.

    Returns:
        Une liste de tokens par texte.
    """
    return [tokenize_text(text) for text in texts]


def build_and_save_index(
    all_chunks: List[Dict[str, Any]],
    save_directory: str,
) -> None:
    """Construit l'index BM25 et le sauvegarde avec les metadonnees.

    Le chemin du fichier est ajoute au texte indexe pour que les termes
    du chemin (nom de module, dossier) comptent dans la recherche. Le
    contenu stocke dans les metadonnees reste celui d'origine.

    Args:
        all_chunks: Chunks {"content": str, "metadata": dict}.
        save_directory: Dossier ou ecrire bm25_index/ et
            chunks_metadata.pkl.

    Raises:
        OSError: Si l'ecriture sur disque echoue.
    """
    if not all_chunks:
        logger.warning("Aucun chunk a indexer, index vide !")
        return

    logger.info("Indexation de %d chunks en cours...", len(all_chunks))

    sources: List[MinimalSource] = [
        MinimalSource.model_validate(chunk["metadata"])
        for chunk in tqdm(all_chunks, desc="Validation des metadatas")
    ]
    corpus_texts = [chunk["content"] for chunk in all_chunks]
    index_texts = []
    for text, source in zip(corpus_texts, sources):
        path = Path(source.file_path)

        # Les fichiers Markdown de documentation sont ceux qui portent le plus
        # souvent les réponses du jeu de données docs. On répète donc un peu
        # plus leur nom de fichier et leurs répertoires pour stabiliser le
        # signal lexical sans changer le contenu réel indexé.
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
        logger.error(
            "Echec de la sauvegarde de l'index dans %s: %s", output_dir, exc
        )
        raise

    logger.info("Indexation terminee et sauvegardee dans %s", output_dir)


def run_ingestion(
    repo_path: str,
    save_directory: str,
    max_chunk_size: int = 2000,
    extensions: Tuple[str, ...] = (".py", ".md"),
) -> List[Dict[str, Any]]:
    """Lit le depot, decoupe les fichiers et construit l'index.

    Args:
        repo_path: Racine du depot a indexer (ex: data/raw).
        save_directory: Dossier ou ecrire l'index (ex: data/processed).
        max_chunk_size: Taille maximale en caracteres d'un chunk.
        extensions: Extensions de fichiers a indexer.

    Returns:
        La liste de tous les chunks produits.

    Raises:
        ValueError: Si repo_path est invalide.
        OSError: Si l'ecriture de l'index echoue.
    """
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