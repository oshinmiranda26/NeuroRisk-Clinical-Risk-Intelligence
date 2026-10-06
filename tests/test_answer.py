from types import SimpleNamespace

from neurorisk.answer import CITATION, answer, extractive_answer, llm_answer
from neurorisk.documents import build_documents
from neurorisk.evaluate_answers import is_correct
from neurorisk.generate import generate
from neurorisk.retrieve import PatientFilteredRetriever, TfidfRetriever


def setup():
    docs = build_documents(generate(n_patients=20, seed=4))
    return docs, PatientFilteredRetriever(TfidfRetriever(docs))


def test_filtered_retrieval_stays_within_patient():
    docs, r = setup()
    out = r.search("What medications is patient P005 taking?", k=5)
    assert set(docs.set_index("doc_id").loc[out, "patient_id"]) == {"P005"}


def test_extractive_answer_cites_retrieved_docs():
    docs, r = setup()
    out = answer("What medications is patient P005 taking?", r, docs, use_llm=False)
    assert out["cited"] and set(out["cited"]) <= set(out["retrieved"])


def test_llm_answer_uses_client_and_parses_citations():
    docs, r = setup()
    fake = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: SimpleNamespace(
        content=[SimpleNamespace(type="text", text="Sertraline [SP005].")])))
    text = llm_answer("meds for P005?", docs, ["SP005"], client=fake)
    assert CITATION.findall(text) == ["SP005"]


def test_correctness_checks():
    assert is_correct("medications", "prazosin, sertraline", "Takes sertraline and prazosin [SP001].")
    assert not is_correct("max_phq9", "21", "Highest PHQ-9 was 12.")
    assert is_correct("ed_visit", "2024-01-02, 2024-05-06", "ED on 2024-01-02 and 2024-05-06.")
