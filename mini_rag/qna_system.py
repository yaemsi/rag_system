from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GroundedAnswer:
    text: str
    doc_ids: list[str]


class QnASystem(ABC):
    """
    Interface for QnA systems to work with the evaluation framework.

    Implementations answer multiple questions from a corpus of documents and can return None for individual
    answers when choosing not to answer. Use `from_corpus()` to instantiate from a list of documents.
    """

    @abstractmethod
    def get_answers(self, questions: list[str]) -> list[GroundedAnswer | None]:
        """
        Answer multiple questions. Returns a list of answers, with None for questions the system decides not
        to answer.
        """
        pass
