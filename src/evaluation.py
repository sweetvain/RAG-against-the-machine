import json
import logging
from pathlib import Path
from statistics import mean
from typing import Dict, List, NamedTuple, Optional , Sequence

import pydantic

from src.models import (
    AnsweredQuestion,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)

logger = logging.getLogger(__name__)

DEFAULT_IOU_THRESHOLD = 0.05
DEFAULT_K_VALUES: Sequence[int] = (1, 3, 5, 10)


def _compute_iou(source_a: MinimalSource, source_b: MinimalSource) -> float:
    if source_a.file_path != source_b.file_path:
        return 0.0

    inter_start = max(source_a.first_character_index, source_b.first_character_index)
    inter_end = min(source_a.last_character_index, source_b.last_character_index)

    if inter_end < inter_start:
        return 0.0

    intersection = inter_end - inter_start + 1
    length_a = source_a.last_character_index - source_a.first_character_index + 1
    length_b = source_b.last_character_index - source_b.first_character_index + 1
    union = length_a + length_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union

def _is_source_found(
    reference: MinimalSource,
    retrieved_sources: Sequence[MinimalSource],
    iou_threshold: float,
) -> bool:
    return any(
        _compute_iou(reference, retrieved) >= iou_threshold
        for retrieved in retrieved_sources
    )

def compute_question_recall(
    reference_sources: Sequence[MinimalSource],
    retrieved_sources: Sequence[MinimalSource],
    k: int,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> Optional[float]:
    if not reference_sources:
        return None

    top_k_retrieved = retrieved_sources[:k] if k > 0 else []

    found_count = sum(
        1
        for reference in reference_sources
        if _is_source_found(reference, top_k_retrieved, iou_threshold)
    )

    return found_count / len(reference_sources)

def load_ground_truth(dataset_path: str) -> Dict[str, List[MinimalSource]]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Le fichier {dataset_path} n'existe pas")

    try:
        raw_data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Le fichier {dataset_path} est malformé : {exc}") from exc

    try:
        rag_dataset = RagDataset.model_validate(raw_data)
    except pydantic.ValidationError as exc:
        raise ValueError(
            f"Le dataset ne respecte pas le schéma attendu : {exc}"
        ) from exc

    ground_truth: Dict[str, List[MinimalSource]] = {}
    skipped_unanswered = 0

    for question in rag_dataset.rag_questions:
        if isinstance(question, AnsweredQuestion):
            ground_truth[question.question_id] = question.sources
        else:
            skipped_unanswered += 1

    if skipped_unanswered:
        logger.warning(
            "%d question(s) sans réponse ignorées dans %s (pas de "
            "sources de référence disponibles pour l'évaluation).",
            skipped_unanswered,
            dataset_path,
        )

    if not ground_truth:
        raise ValueError(
            f"Aucune AnsweredQuestion trouvée dans {dataset_path} : "
            "impossible de calculer un recall sans sources de référence."
        )
    return ground_truth

def load_student_results(
    student_search_results_path: str,
) -> Dict[str, MinimalSearchResults]:
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

    return {
        result.question_id: result for result in student_results.search_results
    }


class EvaluationResult(NamedTuple):
    """Résultat complet d'une évaluation recall@k.

    Attributes:
        recall_per_k: Recall moyen pour chaque valeur de k demandée.
        questions_evaluated: Nombre réel de questions communes entre
            le dataset de référence et les résultats étudiant, ayant
            contribué au calcul de la moyenne.
    """

    recall_per_k: Dict[int, float]
    questions_evaluated: int

def evaluate(
    student_search_results_path: str,
    dataset_path: str,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> EvaluationResult:
    ground_truth = load_ground_truth(dataset_path)
    student_results = load_student_results(student_search_results_path)

    common_ids = set(ground_truth) & set(student_results)
    missing_in_student = set(ground_truth) - set(student_results)
    missing_in_ground_truth = set(student_results) - set(ground_truth)

    if missing_in_student:
        logger.warning(
            "%d question(s) du dataset de référence absentes des "
            "résultats étudiant, ignorées dans le calcul.",
            len(missing_in_student),
        )
    if missing_in_ground_truth:
        logger.warning(
            "%d question(s) des résultats étudiant absentes du "
            "dataset de référence, ignorées dans le calcul.",
            len(missing_in_ground_truth),
        )
    if not common_ids:
        raise ValueError(
            "Aucune question commune entre le dataset de référence et "
            "les résultats étudiant : vérifiez que les deux fichiers "
            "correspondent bien au même dataset."
        )

    recall_per_k: Dict[int, float] = {}

    for k in k_values:
        per_question_recalls = []
        for question_id in common_ids:
            reference_sources = ground_truth[question_id]
            retrieved_sources = student_results[question_id].retrieved_sources

            recall = compute_question_recall(
                reference_sources, retrieved_sources, k, iou_threshold
            )
            if recall is not None:
                per_question_recalls.append(recall)

        recall_per_k[k] = mean(per_question_recalls) if per_question_recalls else 0.0

    return EvaluationResult(
        recall_per_k=recall_per_k,
        questions_evaluated=len(common_ids),
    )

def print_evaluation_report(result: EvaluationResult) -> None:
    """Affiche le rapport d'évaluation dans le même esprit que la
    moulinette, pour une lecture rapide en ligne de commande.

    Args:
        result: Résultat produit par evaluate(), contenant à la fois
            le recall par k et le nombre réel de questions évaluées.
    """
    print("\nEvaluation Results")
    print("=" * 40)
    print(f"Questions évaluées : {result.questions_evaluated}")
    for k, recall in sorted(result.recall_per_k.items()):
        print(f"Recall@{k}: {recall:.3f} ({recall * 100:.1f}%)")