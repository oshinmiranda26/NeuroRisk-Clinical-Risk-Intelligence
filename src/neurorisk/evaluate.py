"""Evaluate retrieval against ground truth.

Run:  python -m neurorisk.evaluate
Metrics: hit@k (is at least one correct document in the top k?) and MRR (mean reciprocal rank of the
first correct document; 1.0 = always ranked first).
"""
import time
from pathlib import Path

import pandas as pd

from neurorisk.documents import build_documents, build_questions, load_tables
from neurorisk.retrieve import (EmbeddingRetriever, FilteredEmbeddingRetriever, HybridRetriever,
                                TfidfRetriever)

RESULTS = Path("results")


def score(retriever, questions, k_max=5):
    rows = []
    for q in questions.itertuples():
        hits = retriever.search(q.question, k_max)
        rank = next((i for i, d in enumerate(hits, start=1) if d in q.relevant), None)
        rows.append({"qtype": q.qtype, "hit@1": rank == 1, "hit@3": rank is not None and rank <= 3,
                     "hit@5": rank is not None, "rr": 1 / rank if rank else 0.0})
    return pd.DataFrame(rows)


def main():
    t = load_tables()
    docs = build_documents(t)
    questions = build_questions(t)
    print(f"{len(docs)} documents | {len(questions)} evaluation questions")
    questions.assign(relevant=questions.relevant.map(";".join)).to_csv("data/eval_questions.csv", index=False)

    tfidf = TfidfRetriever(docs)
    emb = EmbeddingRetriever(docs)
    retrievers = [tfidf, emb, HybridRetriever(tfidf, emb), FilteredEmbeddingRetriever(emb)]

    overall, by_type = [], []
    for r in retrievers:
        start = time.time()
        s = score(r, questions)
        ms = 1000 * (time.time() - start) / len(questions)
        overall.append({"retriever": r.name, "hit@1": s["hit@1"].mean(), "hit@3": s["hit@3"].mean(),
                        "hit@5": s["hit@5"].mean(), "MRR": s["rr"].mean(), "ms_per_query": ms})
        by_type.append(s.groupby("qtype")["hit@3"].mean().rename(r.name))

    overall = pd.DataFrame(overall).round(3)
    by_type = pd.concat(by_type, axis=1).round(3)
    RESULTS.mkdir(exist_ok=True)
    overall.to_csv(RESULTS / "retrieval_metrics.csv", index=False)
    by_type.to_csv(RESULTS / "retrieval_hit3_by_question_type.csv")
    print("\nOverall:\n", overall.to_string(index=False))
    print("\nhit@3 by question type:\n", by_type.to_string())


if __name__ == "__main__":
    main()
