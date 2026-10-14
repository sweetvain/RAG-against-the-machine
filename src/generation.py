"""Génération de réponses avec Qwen/Qwen3-0.6B à partir de sources récupérées."""

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
    """Charge le modèle et le tokenizer, une seule fois grâce au cache.

    Le cache est essentiel pour answer_dataset : sans lui, le modèle
    serait rechargé à chaque question (plusieurs secondes par
    rechargement), rendant tout traitement en lot extrêmement lent.

    Args:
        model_name: Identifiant du modèle sur le Hugging Face Hub.

    Returns:
        Un tuple (tokenizer, model, device).

    Raises:
        RuntimeError: Si le modèle ou le tokenizer ne peuvent pas être
            chargés (réseau indisponible, modèle introuvable, etc.).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Chargement de %s sur %s...", model_name, device)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32 if device == "cpu" else torch.bfloat16,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Impossible de charger le modèle {model_name} : {exc}"
        ) from exc

    model.to(device)
    model.eval()

    logger.info("Modèle chargé avec succès.")
    return tokenizer, model, device


def _read_source_text(source: MinimalSource) -> str:
    """Relit le texte exact d'une source depuis le fichier sur disque.

    Args:
        source: Source dont on veut récupérer le contenu textuel.

    Returns:
        Le texte correspondant à la plage de caractères indiquée, ou
        une chaîne vide si le fichier est illisible (fichier déplacé,
        supprimé, erreur d'encodage).
    """
    try:
        content = Path(source.file_path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning(
            "Impossible de relire la source %s : %s", source.file_path, exc
        )
        return ""

    start = source.first_character_index
    end = source.last_character_index
    return content[start:end + 1]


def _build_context(
    sources: List[MinimalSource], tokenizer: Any, max_context_tokens: int
) -> str:
    """Construit le bloc de contexte à partir des sources récupérées.

    Ajoute les sources dans l'ordre de pertinence jusqu'à atteindre le
    budget de tokens autorisé, puis s'arrête : mieux vaut un contexte
    tronqué mais cohérent qu'un contexte complet qui dépasse la
    fenêtre de tokens du modèle.

    Args:
        sources: Sources récupérées, triées par pertinence décroissante.
        tokenizer: Tokenizer utilisé pour compter les tokens.
        max_context_tokens: Budget maximal de tokens pour le contexte.

    Returns:
        Le texte de contexte, sources concaténées avec leur origine
        indiquée. Chaîne vide si aucune source n'a pu être lue.
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
    """Génère une réponse à une question à partir de sources déjà récupérées.

    Args:
        question: Question posée par l'utilisateur.
        sources: Sources récupérées à utiliser comme contexte.
        model_name: Modèle à utiliser pour la génération.

    Returns:
        La réponse générée. Chaîne vide si la question est vide (pas
        d'appel au modèle dans ce cas, cohérent avec le comportement
        de search() sur une requête vide).
    """
    if not question.strip():
        logger.warning("Question vide reçue, aucune génération effectuée.")
        return ""

    tokenizer, model, device = _load_model_and_tokenizer(model_name)

    context = _build_context(sources, tokenizer, MAX_CONTEXT_TOKENS)
    if not context:
        logger.warning(
            "Aucun contexte disponible pour la question '%s' : les sources "
            "récupérées n'ont pas pu être lues.",
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

    generated_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    return answer


def answer(
    query: str,
    k: int,
    index_directory: str = "data/processed",
) -> MinimalAnswer:
    """Répond à une question unique : recherche puis génération.

    Args:
        query: Question à laquelle répondre.
        k: Nombre de sources à récupérer avant de générer la réponse.
        index_directory: Répertoire contenant l'index BM25.

    Returns:
        Un MinimalAnswer contenant la question, les sources
        récupérées et la réponse générée.

    Raises:
        FileNotFoundError: Si l'index est introuvable.
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
    """Génère une réponse pour chaque question d'un StudentSearchResults.

    Ne relance PAS la recherche : utilise les sources déjà récupérées
    par une précédente exécution de search_dataset, conformément au
    CLI attendu par le sujet (VI.6).

    Args:
        student_search_results_path: Chemin vers un fichier JSON
            conforme au modèle StudentSearchResults, produit par
            search_dataset.
        save_directory: Répertoire où écrire le résultat JSON,
            conforme au modèle StudentSearchResultsAndAnswer.

    Returns:
        L'objet StudentSearchResultsAndAnswer produit, également
        écrit sur disque dans save_directory.

    Raises:
        FileNotFoundError: Si le fichier d'entrée est introuvable.
        ValueError: Si le fichier d'entrée est malformé.
    """
    path = Path(student_search_results_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Le fichier {student_search_results_path} n'existe pas"
        )

    try:
        raw_data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Le fichier {student_search_results_path} est malformé : {exc}"
        ) from exc

    try:
        student_results = StudentSearchResults.model_validate(raw_data)
    except pydantic.ValidationError as exc:
        raise ValueError(
            f"Le fichier ne respecte pas le schéma StudentSearchResults : {exc}"
        ) from exc

    answers: List[MinimalAnswer] = []
    for result in tqdm(
        student_results.search_results, desc="Génération des réponses"
    ):
        generated_answer = generate_answer_from_sources(
            result.question, result.retrieved_sources
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
            "Echec de l'ecriture des resultats dans %s: %s", output_path, exc
        )
        raise

    logger.info(
        "Resultats de generation sauvegardes dans %s", output_path
    )

    return student_results_and_answer