"""
evaluation.py — Evaluation loop + metrics.
 
Metrics implemented:
  1. ExactMatchMetric     — strict string equality after normalisation
  2. TokenF1Metric        — token-level precision/recall F1 (standard QA metric)
  3. AnswerCoverageMetric — fraction of gold answer tokens present in predicted answer
  4. RefusalRateMetric    — fraction of questions the system refused to answer (None)
"""
 
from __future__ import annotations
 
import re
import string
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
 
from mini_rag.qna_system import GroundedAnswer, QnASystem
 
 
# ── Data container ────────────────────────────────────────────────────────────
 
@dataclass
class QnAChallenge:
    """Container for a single question and its target answer."""
    question: str
    target_answer: str
    target_document_id: str
 
 
# ── Base metric ───────────────────────────────────────────────────────────────
 
class Metric(ABC):
    """Base class for evaluation metrics."""
 
    @abstractmethod
    def compute(
        self,
        answers: list[GroundedAnswer | None],
        target_answers: list[str],
    ) -> list[float]:
        pass
 
    @abstractmethod
    def __str__(self) -> str:
        pass
 
    def __repr__(self) -> str:
        return self.__str__()
 
 
# ── Normalisation helper ──────────────────────────────────────────────────────
 
def _normalise(text: str) -> str:
    """Lowercase, strip punctuation and extra whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text
 
 
def _tokens(text: str) -> list[str]:
    return _normalise(text).split()
 
 
# ── Metrics ───────────────────────────────────────────────────────────────────
 
class ExactMatchMetric(Metric):
    """
    1.0 if the normalised predicted answer exactly matches the gold answer,
    0.0 otherwise. None predictions score 0.0.
 
    Why: a strict upper-bound signal; useful for short factual answers.
    """
 
    def compute(
        self,
        answers: list[GroundedAnswer | None],
        target_answers: list[str],
    ) -> list[float]:
        scores = []
        for pred, gold in zip(answers, target_answers):
            if pred is None:
                scores.append(0.0)
            else:
                scores.append(float(_normalise(pred.text) == _normalise(gold)))
        return scores
 
    def __str__(self) -> str:
        return "exact_match"
 
 
class TokenF1Metric(Metric):
    """
    Token-level F1: harmonic mean of precision and recall over bag-of-words.
    Standard metric for extractive / abstractive QA (SQuAD-style).
    None predictions score 0.0.
 
    Why: captures partial matches — important when gold answers are long
    summaries where exact match is too strict.
    """
 
    def compute(
        self,
        answers: list[GroundedAnswer | None],
        target_answers: list[str],
    ) -> list[float]:
        scores = []
        for pred, gold in zip(answers, target_answers):
            if pred is None:
                scores.append(0.0)
                continue
            pred_tokens = _tokens(pred.text)
            gold_tokens = _tokens(gold)
            if not pred_tokens or not gold_tokens:
                scores.append(0.0)
                continue
            pred_set = set(pred_tokens)
            gold_set = set(gold_tokens)
            common = pred_set & gold_set
            if not common:
                scores.append(0.0)
                continue
            precision = len(common) / len(pred_set)
            recall = len(common) / len(gold_set)
            f1 = 2 * precision * recall / (precision + recall)
            scores.append(f1)
        return scores
 
    def __str__(self) -> str:
        return "token_f1"
 
 
class AnswerCoverageMetric(Metric):
    """
    Fraction of gold answer tokens that appear in the predicted answer.
    Equivalent to token recall — penalises missing information.
    None predictions score 0.0.
 
    Why: grounding quality; we want to ensure the predicted answer covers
    the key facts in the gold answer, even if it adds extra text.
    """
 
    def compute(
        self,
        answers: list[GroundedAnswer | None],
        target_answers: list[str],
    ) -> list[float]:
        scores = []
        for pred, gold in zip(answers, target_answers):
            if pred is None:
                scores.append(0.0)
                continue
            pred_tokens = set(_tokens(pred.text))
            gold_tokens = _tokens(gold)
            if not gold_tokens:
                scores.append(1.0)
                continue
            covered = sum(1 for t in gold_tokens if t in pred_tokens)
            scores.append(covered / len(gold_tokens))
        return scores
 
    def __str__(self) -> str:
        return "answer_coverage"
 
 
class RefusalRateMetric(Metric):
    """
    Fraction of questions the system refused to answer (returned None).
    Not a quality metric per se, but a diagnostic: high refusal rate may
    indicate an overly conservative retriever or reader threshold.
 
    Why: helps detect silent failures vs. calibrated abstentions.
    """
 
    def compute(
        self,
        answers: list[GroundedAnswer | None],
        target_answers: list[str],
    ) -> list[float]:
        return [1.0 if a is None else 0.0 for a in answers]
 
    def __str__(self) -> str:
        return "refusal_rate"
 
 
# ── Evaluation loop ───────────────────────────────────────────────────────────
 
class EvaluationLoop:
    """
    Evaluates a QnA system against challenges with known answers.
    Returns dict mapping metric names to score lists.
    """
 
    def __init__(
        self,
        challenges: list[QnAChallenge],
        metrics: list[Metric],
        batch_size: int | None = None,
    ):
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
 
    def _get_answers(self, qna_system: QnASystem) -> list[GroundedAnswer | None]:
        answers = []
        for start in range(0, self._num_questions, self._batch_size):
            end = start + self._batch_size
            questions = [c.question for c in self._challenges[start:end]]
            answers += qna_system.get_answers(questions)
        return answers
 
    def _get_metric_results(
        self,
        answers: list[GroundedAnswer | None],
        metric: Metric,
    ) -> list[float]:
        results = []
        for start in range(0, self._num_questions, self._batch_size):
            end = start + self._batch_size
            target_answers = [c.target_answer for c in self._challenges[start:end]]
            results += metric.compute(answers[start:end], target_answers)
        return results
 
 
# ── Summary helper ────────────────────────────────────────────────────────────
 
def summarise(results: dict[str, list[float]]) -> dict[str, float]:
    """Compute mean score per metric."""
    return {name: sum(scores) / len(scores) for name, scores in results.items()}