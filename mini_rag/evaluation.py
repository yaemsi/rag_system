"""
evaluation.py — Evaluation loop.
Defines the EvaluationLoop and RetrievalEvaluationLoop classes, which run evaluation of a QnA system against a set of challenges and compute metrics. 
Also defines the QnAChallenge data class for representing individual question-answer pairs, and a summarise() helper to compute mean scores per metric.
"""
 
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
 
from mini_rag.qna_system import GroundedAnswer, QnASystem
from mini_rag.metrics import Metric, RetrievalMetric
 
 
# ── Data container ────────────────────────────────────────────────────────────

@dataclass
class QnAChallenge:
    """Container for a single question and its target answer."""
    question: str
    target_answer: str
    target_document_id: str
 
 
 
 
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




@dataclass
class RetrievalQueryResult:
    """Per-query retrieval details produced by RetrievalEvaluationLoop."""
    question: str
    gold_id: str
    gold_type: str                        # "synthetic" | "coveo"
    retrieved_ids: list[str]
    retrieved_scores: dict[str, float]    # doc_id -> RRF score
    found_at: int | None                  # 1-based rank, or None if not found
    filter_applied: bool
    filter_size: int
    filter_contains_gold: bool | None
    failure_mode: str                     # "OK" | "LATE" | "MISS" | "FILTER_KILL"
 
def _doc_type(doc_id: str) -> str:
    try:
        int(doc_id)
        return "synthetic"
    except ValueError:
        return "coveo"

class RetrievalEvaluationLoop:
    """
    Evaluates a Retriever independently of the LLM reader.
    Runs retrieval for all challenges and computes retrieval metrics.

    Usage
    -----
    from mini_rag.evaluation import (
        RetrievalEvaluationLoop, RecallAtK, PrecisionAtK, MeanReciprocalRank
    )
    loop = RetrievalEvaluationLoop(
        challenges=challenges,
        metrics=[RecallAtK(1), RecallAtK(5), RecallAtK(10), PrecisionAtK(5), MeanReciprocalRank()],
    )
    results = loop.run(retriever)
    print(summarise(results))
    """

    def __init__(
        self,
        challenges: list[QnAChallenge],
        metrics: list[RetrievalMetric],
        top_k: int = 10,
    ) -> None:
        self._challenges = challenges
        self._metrics = metrics
        self._top_k = top_k

    def run(
        self, retriever
    ) -> tuple[dict[str, list[float]], list[RetrievalQueryResult]]:
        """
        Retrieve for all challenges, compute metrics, and collect per-query details.
 
        Returns
        -------
        results : dict[str, list[float]]
            Metric name -> per-query scores (same structure as EvaluationLoop).
        query_results : list[RetrievalQueryResult]
            Per-query retrieval details for failure analysis and verbose logging.
        """
        retrieved_ids_list: list[list[str]] = []
        query_results: list[RetrievalQueryResult] = []
 
        for ch in self._challenges:
            gold_id = ch.target_document_id
            candidate_ids = retriever.filter_by_metadata(ch.question)
            chunks = retriever.retrieve(
                query=ch.question,
                candidate_doc_ids=candidate_ids,
            )
 
            retrieved_ids    = [c.doc_id for c in chunks]
            retrieved_scores = {c.doc_id: round(c.score, 5) for c in chunks}
            retrieved_ids_list.append(retrieved_ids)
 
            filter_applied       = candidate_ids is not None
            filter_contains_gold = (gold_id in candidate_ids) if filter_applied else None
            found_at             = next(
                (rank for rank, did in enumerate(retrieved_ids, 1) if did == gold_id),
                None,
            )
 
            if found_at == 1:
                failure_mode = "OK"
            elif found_at is not None:
                failure_mode = "LATE"
            elif filter_applied and not filter_contains_gold:
                failure_mode = "FILTER_KILL"
            else:
                failure_mode = "MISS"
 
            query_results.append(RetrievalQueryResult(
                question=ch.question,
                gold_id=gold_id,
                gold_type=_doc_type(gold_id),
                retrieved_ids=retrieved_ids,
                retrieved_scores=retrieved_scores,
                found_at=found_at,
                filter_applied=filter_applied,
                filter_size=len(candidate_ids) if filter_applied else 0,
                filter_contains_gold=filter_contains_gold,
                failure_mode=failure_mode,
            ))
 
        gold_ids = [ch.target_document_id for ch in self._challenges]
        metric_results: dict[str, list[float]] = {
            str(metric): metric.compute(retrieved_ids_list, gold_ids)
            for metric in self._metrics
        }
        return metric_results, query_results
 
