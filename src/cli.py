import logging
import sys
 
import fire
 
from src.evaluation import evaluate as run_evaluate
from src.evaluation import print_evaluation_report
from src.generation import answer as run_answer
from src.generation import answer_dataset as run_answer_dataset
from src.ingestion import run_ingestion
from src.retrieval import load_index
from src.retrieval import search as run_search
from src.retrieval import search_dataset as run_search_dataset
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

class RagCLI:
    """Commandes disponibles pour le pipeline
    RAG against the machine."""

    def index(
        self,
        max_chunk_size: int = 2000,
        repo_path: str = "data/raw",
        save_directory: str = "data/processed",
    ) -> None:
        """Ingère le corpus et construit l'index BM25.
 
        Args:
            max_chunk_size: Taille maximale en caractères d'un chunk.
            repo_path: Racine du corpus à ingérer.
            save_directory: Répertoire où sauvegarder l'index.
        """

        try:
            chunks = run_ingestion(
                repo_path=repo_path,
                save_directory=save_directory,
                max_chunk_size=max_chunk_size,
            )
        except (ValueError, OSError) as error:
            logger.error("Echec de l'indexation: %s", error)

            sys.exit(1)
        print(
            f"Ingestion complete! Indexed {len(chunks)} chunks "
            f"under {save_directory}/"
        )
    
    def search(
        self,
        query: str,
        k: int = 5,
        index_directory: str = "data/processed",
    ) -> None:
        """Retourne les k sources les plus pertinentes pour une requête.
 
        Args:
            query: Texte de la requête.
            k: Nombre de résultats à retourner.
            index_directory: Répertoire contenant l'index BM25.
        """

        try:
            retriever, chunks_metadata = load_index(index_directory)
        except FileNotFoundError as error:
            logger.error("%s", error)
            sys.exit(1)

        sources = run_search(query, retriever, chunks_metadata, k)

        if not sources:
            print("Aucun résultat trouvé pour cette requête")

            return
        
        for source in sources:
            print(
                f"{source.file_path} "
                f"[{source.first_character_index}:{source.last_character_index}]"
            )
    
    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = "data/output/search_results",
    ) -> None:
        """Exécute la recherche sur toutes les questions d'un dataset.
 
        Args:
            dataset_path: Chemin vers le dataset JSON de questions.
            k: Nombre de sources à retourner par question.
            save_directory: Répertoire où écrire le résultat JSON.
        """

        try:
            result = run_search_dataset(
                dataset_path=dataset_path, k=k, save_directory=save_directory
            )
        
        except (FileNotFoundError, ValueError) as error:
            logger.error("Echec de la recherche : %s", error)
            sys.exit(1)
        
        print(
            f"Saved student_search_results to {save_directory}/ "
            f"({len(result.search_results)} questions traitées)"
        )
    
    def answer(
        self,
        query: str,
        k: int = 5,
        index_directory: str = "data/processed",
    ) -> None:
        """Répond à une question unique via recherche puis génération.
 
        Args:
            query: Question à laquelle répondre.
            k: Nombre de sources à récupérer avant génération.
            index_directory: Répertoire contenant l'index BM25.
        """
        try:
            result = run_answer(query, k=k, index_directory=index_directory)
        except FileNotFoundError as error:
            logger.error("%s", error)
            sys.exit(1)
 
        print(f"Question: {result.question}")
        print(f"Answer: {result.answer}")
 
    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer",
    ) -> None:
        """Génère une réponse pour chaque question d'un StudentSearchResults.
 
        Args:
            student_search_results_path: Chemin vers le fichier JSON
                produit par search_dataset.
            save_directory: Répertoire où écrire le résultat JSON.
        """
        try:
            result = run_answer_dataset(
                student_search_results_path=student_search_results_path,
                save_directory=save_directory,
            )
        except (FileNotFoundError, ValueError) as error:
            logger.error("Echec de la generation : %s", error)
            sys.exit(1)
 
        print(
            f"Saved student_search_results_and_answer to {save_directory}/ "
            f"({len(result.search_results)} questions traitées)"
        )
 
    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
    ) -> None:
        """Calcule le recall@k contre un dataset de référence.
 
        Cette commande est destinée à l'itération personnelle : le
        recall@k officiel est calculé par la moulinette fournie
        séparément, jamais par cette commande.
 
        Args:
            student_search_results_path: Résultats produits par
                search_dataset.
            dataset_path: Dataset AnsweredQuestions de référence.
        """
        try:
            result = run_evaluate(
                student_search_results_path=student_search_results_path,
                dataset_path=dataset_path,
            )
        except (FileNotFoundError, ValueError) as error:
            logger.error("Echec de l'evaluation : %s", error)
            sys.exit(1)
 
        print_evaluation_report(result)
 
 
def main() -> None:
    """Point d'entrée pour `uv run python -m src <command> [options]`.
 
    Englobe fire.Fire dans un filet de sécurité générique : les
    erreurs métier attendues (FileNotFoundError, ValueError) sont
    déjà gérées commande par commande avec des messages clairs, mais
    toute exception imprévue est ici interceptée en dernier recours
    pour ne jamais laisser remonter une trace Python brute, comme
    l'exige le sujet.
    """
    try:
        fire.Fire(RagCLI())
    except SystemExit:
        raise
    except (Exception, KeyboardInterrupt) as error:
        logger.error("Erreur inattendue : %s", error)
        sys.exit(1)
 
 
if __name__ == "__main__":
    main()