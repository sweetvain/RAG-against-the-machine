"""Recherche : chargement de l'index et exécution des requêtes BM25."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import bm25s
import pickle
from tqdm import tqdm
import pydantic

from src.ingestion import tokenize_corpus
from src.models import (
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)

logger = logging.getLogger(__name__)

def load_index(index_directory: str) -> Tuple[bm25s.BM25, List[Dict[str, Any]]]:
    """Charge l'index BM25 et les métadonnées des chunks depuis le disque.

    Args:
        index_directory: Répertoire contenant bm25_index/ et
            chunks_metadata.pkl, produits par build_and_save_index.

    Returns:
        Un tuple (retriever, chunks_metadata) où retriever est l'index
        BM25 rechargé et chunks_metadata la liste des chunks dans le
        même ordre que celui utilisé à l'indexation (l'indice i d'un
        résultat BM25 correspond à chunks_metadata[i]).

    Raises:
        FileNotFoundError: Si l'index ou les métadonnées sont absents.
    """
    bm25_path = Path(index_directory) / "bm25_index"
    chunks_metadata_path = Path(index_directory) / "chunks_metadata.pkl"

    if not bm25_path.exists() or not chunks_metadata_path.exists():
        raise FileNotFoundError(f"Index introuvable dans {index_directory} (bm25_index/ ou chunks_metadata.pkl manquant)")
    
    try:
        retriever = bm25s.BM25.load(str(bm25_path), load_corpus=False)
    except Exception as er:
        logger.error("Echec du chargement de l'index depuis %s: %s", index_directory, er)
        raise
    
    try:
        with chunks_metadata_path.open("rb") as metadata_file:
            chunks_metadata = pickle.load(metadata_file)
    except Exception as exc:
        logger.error("Echec du chargement des metadonnees depuis %s: %s", index_directory, exc)
        raise

    return retriever, chunks_metadata

def search(
    query: str,
    retriever: bm25s.BM25,
    chunks_metadata: List[Dict[str, Any]],
    k: int,
) -> List[MinimalSource]:
    """Retourne les k sources les plus pertinentes pour une requête.

    Args:
        query: Question ou texte de recherche.
        retriever: Index BM25 chargé via load_index.
        chunks_metadata: Métadonnées des chunks, dans le même ordre
            que celui utilisé lors de l'indexation.
        k: Nombre de résultats à retourner.

    Returns:
        Liste des k sources les plus pertinentes, triées par
        pertinence décroissante. Liste vide si la requête est vide,
        si k vaut 0, ou si le corpus est plus petit que k.
    """
    effective_k = min(k, len(chunks_metadata))
    if not query.strip() or effective_k <= 0:
        return []

    query_tokens = tokenize_corpus([query])
    results, scores = retriever.retrieve(query_tokens, k=effective_k)

    sources = []
    for idx in results[0]:
        meta = chunks_metadata[idx]["metadata"]
        sources.append(MinimalSource.model_validate(meta))
    return sources
    


def load_dataset_questions(dataset_path: str) -> List[Dict[str, Any]]:
    """Charge et valide les questions d'un dataset JSON via pydantic.

    Args:
        dataset_path: Chemin vers un fichier JSON conforme au modèle
            RagDataset (clé "questions").

    Returns:
        Liste de dicts avec au moins "question_id" et "question".

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
        ValueError: Si le JSON est malformé ou ne respecte pas le
            schéma RagDataset.
    """
    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Le fichier {dataset_path} n'existe pas")

    try:
        with open(dataset_path, "r", encoding="utf-8") as file:
            raw_data = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Le fichier {dataset_path} est malformé: {exc}")
    
    try:
        rag_dataset = RagDataset.model_validate(raw_data)
    except pydantic.ValidationError as exc:
        raise ValueError(f"Le dataset ne respecte pas le schema attendu: {exc}")

    return [question.model_dump() for question in rag_dataset.rag_questions]

def search_dataset(
    dataset_path: str,
    k: int,
    save_directory: str,
    index_directory: str = "data/processed",
) -> StudentSearchResults:
    """Exécute la recherche BM25 sur toutes les questions d'un dataset.

    Args:
        dataset_path: Chemin vers le fichier JSON de questions
            (UnansweredQuestions ou AnsweredQuestions).
        k: Nombre de sources à retourner par question.
        save_directory: Répertoire où écrire le résultat JSON,
            conforme au modèle StudentSearchResults.
        index_directory: Répertoire contenant l'index BM25 construit
            par la commande index.

    Returns:
        L'objet StudentSearchResults produit, également écrit sur
        disque dans save_directory.

    Raises:
        FileNotFoundError: Si le dataset ou l'index sont introuvables.
        ValueError: Si le dataset est malformé.
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
                retrieved_sources=sources
            )
        )
    
    student_search_results = StudentSearchResults(
        search_results=search_results,
        k=k
    )

    output_dir = Path(save_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / Path(dataset_path).name
    try:
        with output_path.open("w", encoding="utf-8") as output_file:
            output_file.write(student_search_results.model_dump_json(indent=2))
    except OSError as exc:
        logger.error("Echec de l'ecriture des resultats dans %s: %s", output_path, exc)
        raise

    logger.info("Resultats de recherche sauvegardees dans %s", output_path)

    return student_search_results

