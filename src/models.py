import uuid
from typing import List

from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """Represents a single source: file + character range."""

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """Question without an answer or associated sources."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Question with its reference sources and expected answer."""

    sources: List[MinimalSource]
    answer: str


class MinimalSearchResults(BaseModel):
    """Search results for a single question."""

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Search results enriched with a generated answer."""

    answer: str


class RagDataset(BaseModel):
    """Dataset of RAG questions, answered or unanswered."""

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class StudentSearchResults(BaseModel):
    """Expected output for the search_dataset command."""

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Expected output for the answer_dataset command."""

    search_results: List[MinimalAnswer]
    k: int
