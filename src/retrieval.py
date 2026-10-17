"""Recherche : chargement de l'index et execution des requetes BM25."""

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
    """Charge l'index BM25 et les metadonnees des chunks depuis le disque.

    Args:
        index_directory: Repertoire contenant bm25_index/ et
            chunks_metadata.pkl, produits par build_and_save_index.

    Returns:
        Un tuple (retriever, chunks_metadata) ou retriever est l'index
        BM25 recharge et chunks_metadata la liste des chunks dans le
        meme ordre que celui utilise a l'indexation (l'indice i d'un
        resultat BM25 correspond a chunks_metadata[i]).

    Raises:
        FileNotFoundError: Si l'index ou les metadonnees sont absents.
    """
    bm25_path = Path(index_directory) / "bm25_index"
    chunks_metadata_path = Path(index_directory) / "chunks_metadata.pkl"

    if not bm25_path.exists() or not chunks_metadata_path.exists():
        raise FileNotFoundError(
            f"Index introuvable dans {index_directory} "
            "(bm25_index/ ou chunks_metadata.pkl manquant)"
        )

    try:
        retriever = bm25s.BM25.load(str(bm25_path), load_corpus=False)
    except Exception as exc:
        logger.error(
            "Echec du chargement de l'index depuis %s: %s",
            index_directory,
            exc,
        )
        raise

    try:
        with chunks_metadata_path.open("rb") as metadata_file:
            chunks_metadata = pickle.load(metadata_file)
    except Exception as exc:
        logger.error(
            "Echec du chargement des metadonnees depuis %s: %s",
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
    """Retourne les k sources les plus pertinentes pour une requete.

    La requete est tokenisee avec la meme fonction que le corpus, ce qui
    permet de retrouver un identifiant cite tel quel ou reformule.

    Args:
        query: Question ou texte de recherche.
        retriever: Index BM25 charge via load_index.
        chunks_metadata: Metadonnees des chunks, dans le meme ordre
            que celui utilise lors de l'indexation.
        k: Nombre de resultats a retourner.

    Returns:
        Liste des k sources les plus pertinentes, triees par
        pertinence decroissante. Liste vide si la requete est vide ou
        ne contient aucun terme utile, si k vaut 0, ou si le corpus
        est vide.
    """
    effective_k = min(k, len(chunks_metadata))
    if not query.strip() or effective_k <= 0:
        return []

    query_tokens = tokenize_corpus([query])
    if not query_tokens[0]:
        return []

    # On récupère un peu plus de candidats que nécessaire pour laisser une
    # seconde passe légère remettre en avant les vrais fichiers de
    # documentation, souvent noyés par des tests ou des exemples proches.
    retrieval_window = min(max(effective_k * 15, effective_k), min(len(chunks_metadata), 300))
    results, scores = retriever.retrieve(query_tokens, k=retrieval_window)

    query_terms = set(query_tokens[0])
    reranked_candidates = []
    for idx, score in zip(results[0], scores[0]):
        candidate = chunks_metadata[int(idx)]["metadata"]
        adjusted_score = float(score) + _candidate_bonus(candidate, query_terms)
        reranked_candidates.append((adjusted_score, float(score), int(idx)))

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
    """Applique un petit bonus heuristique aux fichiers de documentation.

    La première passe BM25 garde le gros du signal lexical. Cette fonction ne
    fait qu'un réordonnancement local des meilleurs candidats pour favoriser
    les fichiers Markdown et les chemins `docs/`, qui correspondent très
    souvent aux sources de référence du dataset docs.
    """
    path = Path(source.file_path)
    path_text = path.as_posix().lower()

    bonus = 0.0
    if "/docs/" in path_text:
        bonus += 0.80
    if path.suffix == ".md":
        bonus += 0.30
    if any(part in {"tests", "examples", "benchmarks", ".buildkite"} for part in path.parts):
        bonus -= 0.50

    path_terms = set(tokenize_corpus([source.file_path])[0])
    overlap = len(query_terms & path_terms)
    bonus += min(overlap, 5) * 0.12

    return bonus


def load_dataset_questions(dataset_path: str) -> List[Dict[str, Any]]:
    """Charge et valide les questions d'un dataset JSON via pydantic.

    Args:
        dataset_path: Chemin vers un fichier JSON conforme au modele
            RagDataset (cle "rag_questions").

    Returns:
        Liste de dicts avec au moins "question_id" et "question".

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
        ValueError: Si le JSON est malforme ou ne respecte pas le
            schema RagDataset.
    """
    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Le fichier {dataset_path} n'existe pas")

    try:
        with open(dataset_path, "r", encoding="utf-8") as file:
            raw_data = json.load(file)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"Le fichier {dataset_path} est malforme: {exc}"
        ) from exc

    try:
        rag_dataset = RagDataset.model_validate(raw_data)
    except pydantic.ValidationError as exc:
        raise ValueError(
            f"Le dataset ne respecte pas le schema attendu: {exc}"
        ) from exc

    return [question.model_dump() for question in rag_dataset.rag_questions]


def search_dataset(
    dataset_path: str,
    k: int,
    save_directory: str,
    index_directory: str = "data/processed",
) -> StudentSearchResults:
    """Execute la recherche BM25 sur toutes les questions d'un dataset.

    Args:
        dataset_path: Chemin vers le fichier JSON de questions
            (UnansweredQuestions ou AnsweredQuestions).
        k: Nombre de sources a retourner par question.
        save_directory: Repertoire ou ecrire le resultat JSON,
            conforme au modele StudentSearchResults.
        index_directory: Repertoire contenant l'index BM25 construit
            par la commande index.

    Returns:
        L'objet StudentSearchResults produit, egalement ecrit sur
        disque dans save_directory.

    Raises:
        FileNotFoundError: Si le dataset ou l'index sont introuvables.
        ValueError: Si le dataset est malforme.
    """
    retriever, chunks_metadata = load_index(index_directory)
    questions = load_dataset_questions(dataset_path)

    search_results: List[MinimalSearchResults] = []
    for question in tqdm(questions, desc="Recherche BM25 par questions"):
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
        logger.error(
            "Echec de l'ecriture des resultats dans %s: %s", output_path, exc
        )
        raise

    logger.info("Resultats de recherche sauvegardes dans %s", output_path)

    return student_search_results