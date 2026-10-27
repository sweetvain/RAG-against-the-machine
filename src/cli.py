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
    """Available commands for the RAG against the machine pipeline."""

    def index(
        self,
        max_chunk_size: int = 2000,
        repo_path: str = "data/raw",
        save_directory: str = "data/processed",
    ) -> None:
        """Indexes the corpus and builds the BM25 index.

        Args:
            max_chunk_size: Maximum size in characters for a chunk.
            repo_path: Root directory of the corpus to ingest.
            save_directory: Directory where the index is written.
        """
        try:
            chunks = run_ingestion(
                repo_path=repo_path,
                save_directory=save_directory,
                max_chunk_size=max_chunk_size,
            )
        except (ValueError, OSError) as error:
            logger.error("Indexing failed: %s", error)
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
        """Returns the k most relevant sources for a query.

        Args:
            query: Search text.
            k: Number of results to return.
            index_directory: Directory containing the BM25 index.
        """
        try:
            retriever, chunks_metadata = load_index(index_directory)
        except FileNotFoundError as error:
            logger.error("%s", error)
            sys.exit(1)

        sources = run_search(query, retriever, chunks_metadata, k)

        if not sources:
            print("No result found for this query")
            return

        for source in sources:
            print(
                f"{source.file_path} "
                f"[{source.first_character_index}:"
                f"{source.last_character_index}]"
            )

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = "data/output/search_results",
    ) -> None:
        """Runs the BM25 search on all questions in a dataset.

        Args:
            dataset_path: Path to the JSON dataset file.
            k: Number of sources to return per question.
            save_directory: Directory where the result JSON is written.
        """
        try:
            result = run_search_dataset(
                dataset_path=dataset_path,
                k=k,
                save_directory=save_directory,
            )
        except (FileNotFoundError, ValueError) as error:
            logger.error("Search failed: %s", error)
            sys.exit(1)

        print(
            f"Saved student_search_results to {save_directory}/ "
            f"({len(result.search_results)} questions "
            "processed)"
        )

    def answer(
        self,
        query: str,
        k: int = 5,
        index_directory: str = "data/processed",
    ) -> None:
        """Answers a single question using retrieval and generation.

        Args:
            query: Question to answer.
            k: Number of sources to retrieve before generation.
            index_directory: Directory containing the BM25 index.
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
        """Generates an answer for each question in a StudentSearchResults file

        Args:
            student_search_results_path: Path to the JSON file produced by
                search_dataset.
            save_directory: Directory where the result JSON is written.
        """
        try:
            result = run_answer_dataset(
                student_search_results_path=student_search_results_path,
                save_directory=save_directory,
            )
        except (FileNotFoundError, ValueError) as error:
            logger.error("Generation failed: %s", error)
            sys.exit(1)

        print(
            f"Saved student_search_results_and_answer to {save_directory}/ "
            f"({len(result.search_results)} questions "
            "processed)"
        )

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
    ) -> None:
        """Computes recall@k against a reference dataset.

        This command is intended for local iteration only; the official
        recall@k is computed by the provided grading script.

        Args:
            student_search_results_path: Results produced by search_dataset.
            dataset_path: Reference AnsweredQuestions dataset.
        """
        try:
            result = run_evaluate(
                student_search_results_path=student_search_results_path,
                dataset_path=dataset_path,
            )
        except (FileNotFoundError, ValueError) as error:
            logger.error("Evaluation failed: %s", error)
            sys.exit(1)

        print_evaluation_report(result)


def main() -> None:
    """Entry point for `uv run python -m src <command> [options]`.

    Wraps fire.Fire in a generic safety layer. Expected business errors are
    handled per command, while any unexpected exception is caught here to avoid
    leaking a raw Python traceback.
    """
    try:
        fire.Fire(RagCLI())
    except SystemExit:
        raise
    except (Exception, KeyboardInterrupt) as error:
        logger.error("Unexpected error: %s", error)
        sys.exit(1)


if __name__ == "__main__":
    main()
