"""
Metrics implemented:
  1. ExactMatchMetric     — strict string equality after normalisation
  2. TokenF1Metric        — token-level precision/recall F1 (standard QA metric)
  3. AnswerCoverageMetric — fraction of gold answer tokens present in predicted answer
  4. RefusalRateMetric    — fraction of questions the system refused to answer (None)
  5. RecallAtK            — recall at k
  6. PrecisionAtK         — precision at k
  7. MeanReciprocalRank   — mean reciprocal rank
"""
from __future__ import annotations

import re
import string
from abc import ABC, abstractmethod
from mini_rag.qna_system import GroundedAnswer

 
# ── Normalisation helper ──────────────────────────────────────────────────────
 
def _normalise(text: str) -> str:
    """Lowercase, strip punctuation and extra whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text
 
 
def _tokens(text: str) -> list[str]:
    return _normalise(text).split()


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

# ── Retrieval metrics ─────────────────────────────────────────────────────────
class RetrievalMetric(ABC):
    """
    Base class for retrieval-only metrics.
    Operates on retrieved doc_id lists vs. a single gold doc_id per query.
    These run before the LLM reader, allowing retrieval to be debugged
    independently of answer generation.
    """

    @abstractmethod
    def compute(
        self,
        retrieved_ids_list: list[list[str]],
        gold_ids: list[str],
    ) -> list[float]:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass


class RecallAtK(RetrievalMetric):
    """
    Recall@k: 1.0 if the gold document appears in the top-k retrieved docs,
    0.0 otherwise.

    With one gold doc per query this equals Hit@k. It is the primary metric
    for retrieval quality — a miss here means the LLM never had a chance.
    """

    def __init__(self, k: int = 5) -> None:
        self.k = k

    def compute(
        self,
        retrieved_ids_list: list[list[str]],
        gold_ids: list[str],
    ) -> list[float]:
        return [
            1.0 if gold in retrieved[:self.k] else 0.0
            for retrieved, gold in zip(retrieved_ids_list, gold_ids)
        ]

    def __str__(self) -> str:
        return f"recall@{self.k}"


class PrecisionAtK(RetrievalMetric):
    """
    Precision@k: fraction of top-k results that are relevant.
    With one gold doc, this is (1/k) if found in top-k, else 0.

    Penalises retrieving many irrelevant docs alongside the right one,
    which wastes LLM context and can confuse the reader.
    """

    def __init__(self, k: int = 5) -> None:
        self.k = k

    def compute(
        self,
        retrieved_ids_list: list[list[str]],
        gold_ids: list[str],
    ) -> list[float]:
        return [
            (1.0 / self.k) if gold in retrieved[:self.k] else 0.0
            for retrieved, gold in zip(retrieved_ids_list, gold_ids)
        ]

    def __str__(self) -> str:
        return f"precision@{self.k}"


class MeanReciprocalRank(RetrievalMetric):
    """
    MRR: 1/rank of the first relevant result, averaged over queries.
    Rewards finding the gold document at a higher rank.
    """

    def compute(
        self,
        retrieved_ids_list: list[list[str]],
        gold_ids: list[str],
    ) -> list[float]:
        scores = []
        for retrieved, gold in zip(retrieved_ids_list, gold_ids):
            rr = 0.0
            for rank, doc_id in enumerate(retrieved, start=1):
                if doc_id == gold:
                    rr = 1.0 / rank
                    break
            scores.append(rr)
        return scores

    def __str__(self) -> str:
        return "mrr"