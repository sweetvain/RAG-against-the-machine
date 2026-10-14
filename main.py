"""Script de test manuel du pipeline chunking + indexation.

Usage:
    python main.py <chemin_vers_vllm-0.10.1> [max_chunk_size]

Ce script n'est pas un test automatisé (pas de pytest ici) : c'est un
outil de diagnostic rapide pour vérifier visuellement que chaque étape
produit des résultats cohérents avant de lancer une vraie évaluation
recall@k avec la moulinette.
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import bm25s

from src.ingestion import (
    build_and_save_index,
    chunk_section,
    read_repository,
    tokenize_corpus,
    run_ingestion,
)

# Limite du sujet : indexer tout le corpus doit prendre au plus 5 minutes.
MAX_INDEXING_SECONDS = 5 * 60


def check_offsets_consistency(
    file_path: Path, chunks: List[Dict[str, Any]]
) -> List[str]:
    """Vérifie que les offsets de chaque chunk pointent vers le bon
    texte dans le fichier original sur disque.

    C'est le test le plus important : si first_character_index /
    last_character_index ne correspondent pas au texte réellement
    stocké dans le chunk, la moulinette comptera zéro recall pour ce
    fichier même si le contenu retrouvé est parfaitement pertinent.

    Args:
        file_path: Chemin du fichier source original.
        chunks: Chunks produits par chunk_section pour ce fichier.

    Returns:
        Liste de messages d'erreur (vide si tout est cohérent).
    """
    errors = []
    try:
        raw_content = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return [f"Impossible de relire {file_path} pour vérification: {exc}"]

    for i, chunk in enumerate(chunks):
        meta = chunk["metadata"]  # dict, car chunk_section fait .model_dump()
        start = meta["first_character_index"]
        end = meta["last_character_index"]

        if start < 0 or end < start:
            errors.append(
                f"[{file_path.name}] chunk {i}: indices invalides "
                f"(start={start}, end={end})"
            )
            continue

        if end > len(raw_content):
            errors.append(
                f"[{file_path.name}] chunk {i}: last_character_index "
                f"({end}) dépasse la taille du fichier ({len(raw_content)})"
            )
            continue

        extracted = raw_content[start:end + 1]
        if extracted.strip() != chunk["content"].strip():
            errors.append(
                f"[{file_path.name}] chunk {i}: le texte extrait via "
                f"content[{start}:{end + 1}] ne correspond PAS au "
                f"contenu stocké. Offsets désynchronisés."
            )
    return errors


def check_max_chunk_size(
    chunks: List[Dict[str, Any]], max_chunk_size: int
) -> List[str]:
    """Vérifie qu'aucun chunk ne dépasse la taille maximale autorisée.

    Args:
        chunks: Chunks à vérifier.
        max_chunk_size: Taille maximale autorisée en caractères.

    Returns:
        Liste de messages d'erreur (vide si tout est conforme).
    """
    errors = []
    for i, chunk in enumerate(chunks):
        length = len(chunk["content"])
        if length > max_chunk_size:
            errors.append(
                f"chunk {i}: taille {length} > max_chunk_size={max_chunk_size} "
                f"(file_path={chunk['metadata']['file_path']})"
            )
    return errors


def run_chunking_diagnostics(
    repo_path: str, max_chunk_size: int
) -> Tuple[List[Dict[str, Any]], List[Path]]:
    """Exécute le chunking sur tout le dépôt et affiche un diagnostic.

    Args:
        repo_path: Racine du dépôt à ingérer (ex: data/raw/vllm-0.10.1).
        max_chunk_size: Taille maximale des chunks.

    Returns:
        Un tuple (all_chunks, files_without_chunks) où all_chunks est
        la liste complète des chunks produits et files_without_chunks
        la liste des fichiers pour lesquels aucun chunk n'a été généré.
    """
    files = read_repository(repo_path)
    print(f"Fichiers trouvés : {len(files)}")

    all_chunks: List[Dict[str, Any]] = []
    files_without_chunks: List[Path] = []
    total_offset_errors = 0
    total_size_errors = 0
    files_checked_for_offsets = 0

    for file_path in files:
        chunks = chunk_section(file_path, max_chunk_size=max_chunk_size)

        if not chunks:
            files_without_chunks.append(file_path)
            continue

        # Vérification offsets coûteuse (relit le fichier) : on limite
        # aux 50 premiers fichiers pour garder le diagnostic rapide.
        if files_checked_for_offsets < 50:
            offset_errors = check_offsets_consistency(file_path, chunks)
            total_offset_errors += len(offset_errors)
            for err in offset_errors[:3]:
                print(f"  ERREUR OFFSET: {err}")
            files_checked_for_offsets += 1

        size_errors = check_max_chunk_size(chunks, max_chunk_size)
        total_size_errors += len(size_errors)
        for err in size_errors[:3]:
            print(f"  ERREUR TAILLE: {err}")

        all_chunks.extend(chunks)

    print("\n--- Résumé du chunking ---")
    print(f"Total chunks produits         : {len(all_chunks)}")
    print(f"Fichiers sans aucun chunk     : {len(files_without_chunks)}")
    print(f"Fichiers vérifiés (offsets)   : {files_checked_for_offsets}")
    print(f"Erreurs d'offsets détectées   : {total_offset_errors}")
    print(f"Erreurs de taille détectées   : {total_size_errors}")

    if total_offset_errors > 0:
        print(
            "\n⚠️  Des offsets sont désynchronisés : le recall@k sera "
            "faux même si le contenu retrouvé est correct."
        )
    if total_size_errors > 0:
        print(
            "\n⚠️  Des chunks dépassent max_chunk_size : la moulinette "
            "invalidera toute la sortie qui les contient."
        )

    return all_chunks, files_without_chunks


def report_files_without_chunks(
    files_without_chunks: List[Path], sample_size: int = 15
) -> None:
    """Affiche un échantillon des fichiers n'ayant produit aucun chunk,
    avec leur taille, pour distinguer les cas normaux (fichiers vides)
    des cas suspects (fichiers non vides mais ignorés à tort).

    Args:
        files_without_chunks: Fichiers pour lesquels chunk_section a
            renvoyé une liste vide.
        sample_size: Nombre maximal de fichiers à afficher en détail.
    """
    if not files_without_chunks:
        return

    print(f"\n--- Détail des {len(files_without_chunks)} fichiers sans chunk ---")

    sizes = []
    for f in files_without_chunks:
        try:
            sizes.append((f, f.stat().st_size))
        except OSError:
            sizes.append((f, -1))

    # Trie du plus gros au plus petit : les gros fichiers vides sont
    # les plus suspects et méritent d'être vérifiés en priorité.
    sizes.sort(key=lambda pair: pair[1], reverse=True)

    empty_count = sum(1 for _, size in sizes if size == 0)
    print(f"Fichiers réellement vides (0 octet) : {empty_count}")
    print(f"Échantillon des {sample_size} plus gros fichiers sans chunk :")
    for file_path, size in sizes[:sample_size]:
        flag = " ⚠️ SUSPECT (non vide mais aucun chunk)" if size > 0 else ""
        print(f"  {size:>8} octets  {file_path}{flag}")


def run_indexation_diagnostics(
    all_chunks: List[Dict[str, Any]], save_directory: str
) -> None:
    """Construit l'index puis vérifie qu'il est bien rechargeable et
    cohérent en effectuant quelques recherches de test.

    Args:
        all_chunks: Chunks à indexer.
        save_directory: Répertoire de sauvegarde de l'index.
    """
    print("\n--- Construction de l'index ---")
    indexing_start = time.perf_counter()
    build_and_save_index(all_chunks, save_directory)
    indexing_duration = time.perf_counter() - indexing_start

    print(f"\nTemps d'indexation (BM25 seul) : {indexing_duration:.1f}s")
    if indexing_duration > MAX_INDEXING_SECONDS:
        print(
            f"⚠️  Dépasse la limite de {MAX_INDEXING_SECONDS}s imposée "
            "par le sujet pour l'indexation complète du corpus."
        )

    print("\n--- Vérification du rechargement de l'index ---")
    reloaded_retriever = bm25s.BM25.load(
        str(Path(save_directory) / "bm25_index"), load_corpus=False
    )
    print("Index rechargé avec succès.")

    test_queries = [
        "how to configure the openai server",
        "class definition initialization",
        "tensor parallel attention",
    ]
    for query in test_queries:
        query_tokens = tokenize_corpus([query])
        results, scores = reloaded_retriever.retrieve(query_tokens, k=3)
        print(f"\nRequête: '{query}'")
        for idx, score in zip(results[0], scores[0]):
            meta = all_chunks[idx]["metadata"]
            print(
                f"  score={score:.3f} file={meta['file_path']} "
                f"[{meta['first_character_index']}:"
                f"{meta['last_character_index']}]"
            )


def main() -> None:
    """Point d'entrée du script de diagnostic."""
    if len(sys.argv) < 2:
        print("Usage: python main.py <chemin_vers_vllm-0.10.1> [max_chunk_size]")
        sys.exit(1)

    repo_path = sys.argv[1]
    max_chunk_size = int(sys.argv[2]) if len(sys.argv) > 2 else 2000

    pipeline_start = time.perf_counter()

    all_chunks = run_ingestion(
        repo_path=repo_path,
        save_directory="data/processed_test",
        max_chunk_size=max_chunk_size,
    )

    total_duration = time.perf_counter() - pipeline_start
    print(f"\nTemps total (chunking + indexation) : {total_duration:.1f}s")
    if total_duration > MAX_INDEXING_SECONDS:
        print(f"⚠️  Dépasse les {MAX_INDEXING_SECONDS}s imposées par le sujet.")

    if not all_chunks:
        print("Aucun chunk produit, arrêt du diagnostic.")
        sys.exit(1)

    # Vérifications qualité en aval, sur le résultat déjà indexé
    print("\n--- Vérification offsets sur un échantillon ---")
    sample_files = {chunk["metadata"]["file_path"] for chunk in all_chunks[:200]}
    total_offset_errors = 0
    for file_str in list(sample_files)[:50]:
        file_path = Path(file_str)
        file_chunks = [c for c in all_chunks if c["metadata"]["file_path"] == file_str]
        errors = check_offsets_consistency(file_path, file_chunks)
        total_offset_errors += len(errors)
        for err in errors[:3]:
            print(f"  ERREUR OFFSET: {err}")
    print(f"Erreurs d'offsets détectées : {total_offset_errors}")

    run_indexation_diagnostics(all_chunks, save_directory="data/processed_test")
    


if __name__ == "__main__":
    main()