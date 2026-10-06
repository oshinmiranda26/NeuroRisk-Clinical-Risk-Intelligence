import pandas as pd

from neurorisk.documents import build_documents, build_questions
from neurorisk.generate import generate
from neurorisk.retrieve import HybridRetriever, TfidfRetriever


def small():
    t = generate(n_patients=25, seed=5)
    return t, build_documents(t)


def test_one_summary_per_patient_and_one_doc_per_note():
    t, docs = small()
    assert (docs.doc_type == "summary").sum() == len(t["patients"])
    assert (docs.doc_type == "note").sum() == len(t["notes"])


def test_questions_point_to_existing_documents():
    t, docs = small()
    qs = build_questions(t, n_patients=10)
    ids = set(docs.doc_id)
    assert all(set(r) <= ids for r in qs.relevant)


def test_tfidf_retrieves_the_named_patients_documents():
    t, docs = small()
    r = TfidfRetriever(docs)
    patient_docs = set(docs.loc[docs.patient_id == "P003", "doc_id"])
    assert set(r.search("What medications is patient P003 taking?", k=3)) <= patient_docs


def test_hybrid_returns_k_unique_results():
    t, docs = small()
    tf = TfidfRetriever(docs)
    out = HybridRetriever(tf, tf).search("emergency department visit for P001", k=5)
    assert len(out) == len(set(out)) == 5
