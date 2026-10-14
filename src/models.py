"""Modèles Pydantic (data validation)"""

from pydantic import BaseModel, Field
from typing import List
import uuid


class MinimalSource(BaseModel):
    """ Represente une source unique: fichier + plage de caracteres """
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """ Question sans reponse ni sources associees """
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """ Question avec ses sources de reference et sa reponse attendue """
    sources: List[MinimalSource]
    answer: str


class MinimalSearchResults(BaseModel):
    """ Resultats de recherche pour une question donnee """
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """ Resultats de recherche enrichis d'une reponse generee """
    answer: str


class RagDataset(BaseModel):
    """ Datset de question RAG, repondues ou non"""
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class StudentSearchResults(BaseModel):
    """Sortie attendue pour la commande search_dataset """
    search_results: List[MinimalSearchResults]
    k: int

class StudentSearchResultsAndAnswer(BaseModel):
    """Sortie attendue pour la commande answer_dataset """
    search_results: List[MinimalAnswer]
    k: int