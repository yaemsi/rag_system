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
 
# Run on bonus queries (no gold answers — prints answers only):
python main.py --split bonus
"""
 
from __future__ import annotations
 
import argparse
import os
import sys

from argparse import Namespace
from pathlib import Path
from transformers import HfArgumentParser
from mini_rag.arguments import (
    GeneralArguments,
    ChunkerArguments,
    RetrieverArguments,
    ReaderArguments,
)
from mini_rag.corpus import load_queries
from mini_rag.evaluation import (
    AnswerCoverageMetric,
    EvaluationLoop,
    ExactMatchMetric,
    QnAChallenge,
    RefusalRateMetric,
    TokenF1Metric,
    summarise,
)
from mini_rag.rag_system import build_system
 

def make_challenges(rows: list[dict]) -> list[QnAChallenge]:
    return [
        QnAChallenge(
            question=r["query"],
            target_answer=r.get("answer", ""),
            target_document_id=str(r["document_id"]),
        )
        for r in rows
    ]
 
 
def run_eval(params: Namespace, system) -> None:
    split = params.split
    path = Path(params.data_dir) / f"{split}.jsonl.gz"
    rows = load_queries(path)
 
    if split == "bonus":
        # No gold answers — just print the system's responses
        questions = [r["query"] for r in rows]
        answers = system.get_answers(questions)
        print(f"\n{'='*60}")
        print(f"BONUS QUERIES — system responses")
        print(f"{'='*60}")
        for q, a in zip(questions, answers):
            print(f"\nQ: {q}")
            if a is None:
                print("A: [REFUSED — not answerable from corpus]")
            else:
                print(f"A: {a.text}")
                print(f"   [source doc_ids: {a.doc_ids}]")
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
 
    print(f"\n{'='*60}")
    print(f"***** Evaluation results — {split.upper()} ({len(challenges)} questions)")
    print(f"{'='*60}")
    for name, score in summary.items():
        print(f"  {name:<22} {score:.4f}")
 
    if params.verbose:
        print(f"\nPer-question scores (token_f1):")
        for i, (ch, score) in enumerate(zip(challenges, results["token_f1"])):
            print(f"  [{i:03d}] f1={score:.3f}  Q: {ch.question[:80]}")
 
 
def main() -> None:
    parser = HfArgumentParser(
        (GeneralArguments, ChunkerArguments, RetrieverArguments, ReaderArguments)
        )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file
        gen_args, chunk_args, ret_args, read_args  = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    elif len(sys.argv) == 2 and (sys.argv[1].endswith(".yaml") or sys.argv[1].endswith(".yml")):
        # If we pass only one argument to the script and it's the path to a yaml file
        gen_args, chunk_args, ret_args, read_args  = parser.parse_yaml_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        gen_args, chunk_args, ret_args, read_args = parser.parse_args_into_dataclasses()

    system = build_system(
        chunk_params=chunk_args,
        ret_params=ret_args,
        read_params=read_args,
        corpus_path=Path(gen_args.data_dir) / "corpus.jsonl.gz",
        index_dir=Path(gen_args.index_dir),
        force_rebuild=gen_args.rebuild,
    )
 
    run_eval(gen_args, system)
 
 
if __name__ == "__main__":
    main()
