from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from knowai_technical_challenge.qna_system import GroundedAnswer, QnASystem


@dataclass
class QnAChallenge:
    """Container for a single question and its target answer."""

    question: str
    target_answer: str
    target_document_id: str


class Metric(ABC):
    """Base class for evaluation metrics. Implement compute() to create custom metrics."""

    @abstractmethod
    def compute(self, answers: list[GroundedAnswer | None], target_answers: list[str]) -> list[float]:
        """Compute scores comparing answers to target answers."""
        pass

    @abstractmethod
    def __str__(self) -> str:
        """Return metric name (used as key in results)."""
        pass

    def __repr__(self) -> str:
        return self.__str__()


class EvaluationLoop:
    """
    Evaluates a QnA system against challenges with known answers.
    Returns dict mapping metric names to score lists.
    """

    def __init__(self, challenges: list[QnAChallenge], metrics: list[Metric], batch_size: int | None = None):
        self._challenges = challenges
        self._num_questions = len(challenges)
        self._metrics = metrics
        self._batch_size = self._num_questions if batch_size is None else batch_size

    def run(self, qna_system: QnASystem) -> dict[str, list[Any]]:
        """Run evaluation and return dict of metric names to score lists."""
        answers = self._get_answers(qna_system)
        results: dict[str, list[Any]] = {}
        for metric in self._metrics:
            results[str(metric)] = self._get_metric_results(answers, metric)
        return results

    def _get_answers(self, qna_system: QnASystem) -> list[str | None]:
        answers = []
        for start in range(0, self._num_questions, self._batch_size):
            end = start + self._batch_size
            questions = [c.question for c in self._challenges[start:end]]
            answers += qna_system.get_answers(questions)
        return answers

    def _get_metric_results(self, answers: list[GroundedAnswer | None], metric: Metric) -> list[float]:
        results = []
        for start in range(0, self._num_questions, self._batch_size):
            end = start + self._batch_size
            target_answers = [c.target_answer for c in self._challenges[start:end]]
            results += metric.compute(answers, target_answers)
        return results
