"""Evaluate end-to-end answers: correctness and citation quality.

Run:  python -m neurorisk.evaluate_answers            (LLM if ANTHROPIC_API_KEY is set, else extractive)
      python -m neurorisk.evaluate_answers --extractive
Correctness is checked automatically against ground truth (all medications named, the right PHQ-9 number,
all ED dates). Citation checks: does the answer cite a correct document, and only documents it was given?
"""
import argparse
import re
from pathlib import Path

import pandas as pd

from neurorisk.answer import answer, llm_available
from neurorisk.documents import build_documents, load_tables
from neurorisk.retrieve import EmbeddingRetriever, HybridRetriever, PatientFilteredRetriever, TfidfRetriever

RESULTS = Path("results")


def is_correct(qtype, expected, text):
    t = text.lower()
    if qtype == "medications":
        return all(m.strip().lower() in t for m in expected.split(","))
    if qtype == "ed_visit":
        return all(d.strip() in text for d in expected.split(","))
    return re.search(rf"\b{re.escape(expected)}\b", text) is not None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=40, help="number of questions to evaluate")
    p.add_argument("--extractive", action="store_true", help="force the free extractive mode")
    args = p.parse_args()

    t = load_tables()
    docs = build_documents(t)
    qs = pd.read_csv("data/eval_questions.csv")
    qs["relevant"] = qs["relevant"].str.split(";")
    qs = qs.groupby("qtype").head(args.n // 4).reset_index(drop=True)

    tfidf, emb = TfidfRetriever(docs), EmbeddingRetriever(docs)
    retriever = HybridRetriever(PatientFilteredRetriever(tfidf), PatientFilteredRetriever(emb))
    use_llm = llm_available() and not args.extractive
    print(f"Evaluating {len(qs)} questions | mode: {'LLM' if use_llm else 'extractive'}")

    rows = []
    for q in qs.itertuples():
        out = answer(q.question, retriever, docs, use_llm=use_llm)
        rows.append({"qtype": q.qtype, "question": q.question, "expected": q.answer, "answer": out["answer"],
                     "correct": is_correct(q.qtype, str(q.answer), out["answer"]),
                     "cites_relevant": any(c in q.relevant for c in out["cited"]),
                     "cites_only_retrieved": set(out["cited"]) <= set(out["retrieved"]),
                     "has_citation": len(out["cited"]) > 0})
    res = pd.DataFrame(rows)
    mode = "llm" if use_llm else "extractive"
    RESULTS.mkdir(exist_ok=True)
    res.to_csv(RESULTS / f"answers_{mode}.csv", index=False)
    summary = res.groupby("qtype")[["correct", "cites_relevant", "cites_only_retrieved"]].mean().round(3)
    summary.loc["overall"] = res[["correct", "cites_relevant", "cites_only_retrieved"]].mean().round(3)
    summary.to_csv(RESULTS / f"answer_metrics_{mode}.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main()
