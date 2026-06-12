
"""
main.py — Entry point for the Coveo QA challenge.
 
Usage
-----
# First run (builds + saves index, takes ~10–20 min on 3630 docs):
python main.py
 
# Subsequent runs (loads cached index, fast):
python main.py
 
# Force rebuild even if index exists:
python main.py --rebuild
 
# Evaluate only on validation set:
python main.py --split valid
 
# Run retrieval evaluation with per-query diagnosis:
python main.py --ret_eval --verbose --split valid
 
# Save results to JSON:
python main.py --split valid --output_dir ./output
 
# Run on bonus queries (no gold answers — prints answers only):
python main.py --split bonus
"""
 
from __future__ import annotations
 
import json
import os
import sys
 
from argparse import Namespace
from loguru import logger
from pathlib import Path
from transformers import HfArgumentParser
 
from mini_rag import (
    GeneralArguments,
    ChunkerArguments,
    RetrieverArguments,
    ReaderArguments,
)
from mini_rag import (
    load_queries,
    summarise,
    build_system,
)
from mini_rag import (
    RetrievalEvaluationLoop,
    RetrievalQueryResult,
    RecallAtK,
    PrecisionAtK,
    MeanReciprocalRank,
    AnswerCoverageMetric,
    EvaluationLoop,
    ExactMatchMetric,
    QnAChallenge,
    RefusalRateMetric,
    TokenF1Metric,
)
 
 
def make_challenges(rows: list[dict]) -> list[QnAChallenge]:
    return [
        QnAChallenge(
            question=r["query"],
            target_answer=r.get("answer", ""),
            target_document_id=str(r["document_id"]),
        )
        for r in rows
    ]
 
 
def _save_json(data: dict, output_dir: str, split: str) -> None:
    """Serialize results dict to output_dir/results-{split}.json."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    file_path = out_path / f"results-{split}.json"
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"==>> Results saved to {file_path}")
 
 
def run_ret_eval(params: Namespace, system) -> None:
    split = params.split
    path = Path(params.data_dir) / f"{split}.jsonl.gz"
    rows = load_queries(path)
 
    if split == "bonus":
        logger.info("BONUS QUERIES — retrieval evaluation skipped (no gold answers)")
        return
 
    challenges = make_challenges(rows)
    n = len(challenges)
    top_k = params.top_k_eval
 
    metrics = [
        RecallAtK(k=1),
        RecallAtK(k=5),
        RecallAtK(k=10),
        PrecisionAtK(k=5),
        MeanReciprocalRank(),
    ]
    loop = RetrievalEvaluationLoop(challenges, metrics, top_k=top_k)
    results, query_results = loop.run(system._retriever)
    summary = summarise(results)
 
    # ── Summary ──────────────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"***** Retrieval evaluation — {split.upper()} ({n} questions)")
    logger.info(f"{'='*60}")
    for name, score in summary.items():
        logger.info(f"  {name:<22} {score:.4f}")
 
    # ── Failure mode breakdown ────────────────────────────────────────────────
    modes = {"OK": 0, "LATE": 0, "MISS": 0, "FILTER_KILL": 0}
    for r in query_results:
        modes[r.failure_mode] += 1
 
    logger.info(f"\n{'='*60}")
    logger.info(f"  Failure mode breakdown")
    logger.info(f"{'='*60}")
    logger.info(f"  OK           (rank 1):               {modes['OK']:>3}  ({modes['OK']/n*100:.1f}%)")
    logger.info(f"  LATE         (rank 2–{top_k}):           {modes['LATE']:>3}  ({modes['LATE']/n*100:.1f}%)")
    logger.info(f"  MISS         (not in top-{top_k}):       {modes['MISS']:>3}  ({modes['MISS']/n*100:.1f}%)")
    logger.info(f"  FILTER_KILL  (filter excluded gold):  {modes['FILTER_KILL']:>3}  ({modes['FILTER_KILL']/n*100:.1f}%)")
 
    for mode in ("MISS", "FILTER_KILL", "LATE"):
        subset = [r for r in query_results if r.failure_mode == mode]
        if not subset:
            continue
        synthetic = sum(1 for r in subset if r.gold_type == "synthetic")
        coveo     = sum(1 for r in subset if r.gold_type == "coveo")
        logger.info(f"  {mode} by doc type: synthetic={synthetic}  coveo={coveo}")
 
    # ── Verbose: per-query diagnosis ─────────────────────────────────────────
    if params.verbose:
        failed = [r for r in query_results if r.failure_mode != "OK"]
        logger.info(f"\n{'='*60}")
        logger.info(f"  Per-query diagnosis — {len(failed)} failed queries")
        logger.info(f"{'='*60}")
 
        for i, r in enumerate(query_results):
            if r.failure_mode == "OK":
                continue
 
            icon     = {"LATE": "~", "MISS": "x", "FILTER_KILL": "o"}[r.failure_mode]
            rank_str = f"rank={r.found_at}" if r.found_at else f"not in top-{top_k}"
 
            logger.info(
                f"\n  {icon} [{i:03d}] {r.failure_mode:<12} {rank_str:<15} "
                f"gold={r.gold_id} [{r.gold_type}]"
            )
            logger.info(f"    Q: {r.question[:90]}")
 
            if r.filter_applied:
                logger.info(
                    f"    filter: {r.filter_size} candidate docs  |  "
                    f"contains_gold={r.filter_contains_gold}"
                )
            else:
                logger.info("    filter: none (Coveo doc or no metadata match)")
 
            logger.info(f"    top-5 retrieved:")
            for rank, doc_id in enumerate(r.retrieved_ids[:5], 1):
                marker = " <- GOLD" if doc_id == r.gold_id else ""
                score  = r.retrieved_scores.get(doc_id, 0)
                logger.info(f"      rank {rank}: {doc_id:<30} score={score}{marker}")
 
    # ── Save JSON ─────────────────────────────────────────────────────────────
    output = {
        "split": split,
        "evaluation_type": "retrieval",
        "n_questions": n,
        "summary": summary,
        "failure_modes": modes,
        "per_query": [
            {
                "index": i,
                "question": r.question,
                "gold_id": r.gold_id,
                "gold_type": r.gold_type,
                "failure_mode": r.failure_mode,
                "found_at_rank": r.found_at,
                "filter_applied": r.filter_applied,
                "filter_size": r.filter_size,
                "filter_contains_gold": r.filter_contains_gold,
                "retrieved_ids": r.retrieved_ids,
                "retrieved_scores": r.retrieved_scores,
            }
            for i, r in enumerate(query_results)
        ],
    }
    _save_json(output, params.output_dir, f"{split}-retrieval")
 
 
def run_eval(params: Namespace, system) -> None:
    split = params.split
    path = Path(params.data_dir) / f"{split}.jsonl.gz"
    rows = load_queries(path)
 
    if split == "bonus":
        questions = [r["query"] for r in rows]
        answers = system.get_answers(questions)
        logger.info(f"\n{'='*60}")
        logger.info(f"BONUS QUERIES — system responses")
        logger.info(f"{'='*60}")
 
        bonus_output = []
        for q, a in zip(questions, answers):
            logger.info(f"\nQ: {q}")
            if a is None:
                logger.info("A: [REFUSED — not answerable from corpus]")
                bonus_output.append({"question": q, "answer": None, "doc_ids": []})
            else:
                logger.info(f"A: {a.text}")
                logger.info(f"   [source doc_ids: {a.doc_ids}]")
                bonus_output.append({"question": q, "answer": a.text, "doc_ids": a.doc_ids})
 
        _save_json(
            {"split": split, "evaluation_type": "bonus", "responses": bonus_output},
            params.output_dir, split,
        )
        return
 
    challenges = make_challenges(rows)
    metrics = [
        ExactMatchMetric(),
        TokenF1Metric(),
        AnswerCoverageMetric(),
        RefusalRateMetric(),
    ]
    loop = EvaluationLoop(challenges, metrics, batch_size=params.eval_batch_size)
    results = loop.run(system)
    summary = summarise(results)
 
    logger.info(f"\n{'='*60}")
    logger.info(f"***** Evaluation results — {split.upper()} ({len(challenges)} questions)")
    logger.info(f"{'='*60}")
    for name, score in summary.items():
        logger.info(f"  {name:<22} {score:.4f}")
 
    if params.verbose:
        logger.info(f"\nPer-question scores (token_f1):")
        for i, (ch, score) in enumerate(zip(challenges, results["token_f1"])):
            logger.info(f"  [{i:03d}] f1={score:.3f}  Q: {ch.question[:80]}")
 
    # ── Save JSON ─────────────────────────────────────────────────────────────
    output = {
        "split": split,
        "evaluation_type": "qa",
        "n_questions": len(challenges),
        "summary": summary,
        "per_question": [
            {
                "index": i,
                "question": ch.question,
                "target_answer": ch.target_answer,
                "target_document_id": ch.target_document_id,
                "scores": {name: results[name][i] for name in results},
            }
            for i, ch in enumerate(challenges)
        ],
    }
    _save_json(output, params.output_dir, split)
 
 
def main() -> None:
    parser = HfArgumentParser(
        (GeneralArguments, ChunkerArguments, RetrieverArguments, ReaderArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        gen_args, chunk_args, ret_args, read_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    elif len(sys.argv) == 2 and (sys.argv[1].endswith(".yaml") or sys.argv[1].endswith(".yml")):
        gen_args, chunk_args, ret_args, read_args = parser.parse_yaml_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        gen_args, chunk_args, ret_args, read_args = parser.parse_args_into_dataclasses()
 
    logger.info(f"***** Main program *****")
    logger.info(f"==>> Building the retrieval-augmented generation system...")
    system = build_system(
        chunk_params=chunk_args,
        ret_params=ret_args,
        read_params=read_args,
        corpus_path=Path(gen_args.data_dir) / "corpus.jsonl.gz",
        index_dir=Path(gen_args.index_dir),
        force_rebuild=gen_args.rebuild,
    )
 
    if gen_args.ret_eval:
        logger.info(f"==>> Running the retrieval evaluation...")
        run_ret_eval(gen_args, system)
 
    logger.info(f"==>> Scoring the answers...")
    run_eval(gen_args, system)
 
 
if __name__ == "__main__":
    main()
 