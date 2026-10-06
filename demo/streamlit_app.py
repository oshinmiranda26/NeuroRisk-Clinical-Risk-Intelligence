"""NeuroRisk demo: ask questions about synthetic mental health records and see cited evidence."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))  # make the neurorisk package importable on Streamlit Cloud

import streamlit as st  # noqa: E402

from neurorisk.answer import answer  # noqa: E402
from neurorisk.documents import build_documents, load_tables  # noqa: E402
from neurorisk.retrieve import PatientFilteredRetriever, TfidfRetriever  # noqa: E402

st.set_page_config(page_title="NeuroRisk", page_icon="🧠", layout="wide")

try:  # use a key from Streamlit secrets if one is configured; run without it otherwise
    if "ANTHROPIC_API_KEY" in st.secrets:
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
except Exception:  # no secrets file at all
    pass


@st.cache_resource
def setup():
    docs = build_documents(load_tables(ROOT / "data"))
    return docs, PatientFilteredRetriever(TfidfRetriever(docs))


docs, retriever = setup()

st.title("NeuroRisk: Clinical Question Answering with Cited Evidence")
st.markdown(
    "Ask a question about a **synthetic** patient (P001 to P200). The system retrieves that patient's records "
    "and answers with citations to the exact notes or summary it used. "
    "[Code, data generator, and retrieval evaluation on GitHub]"
    "(https://github.com/oshinmiranda26/NeuroRisk-Clinical-Risk-Intelligence)"
)
st.caption("All data is synthetic. Research demo only, not for clinical use.")

examples = ["What medications is patient P012 taking?",
            "What is the highest PHQ-9 score recorded for patient P045?",
            "When did patient P010 visit the emergency department?",
            "Has patient P017 had a positive safety screening?"]
choice = st.selectbox("Example questions", ["Write your own"] + examples)
question = st.text_input("Question", value="" if choice == "Write your own" else choice)
llm_on = "ANTHROPIC_API_KEY" in os.environ
use_llm = st.toggle("Generate the answer with Claude", value=llm_on, disabled=not llm_on,
                    help=None if llm_on else "LLM generation is not configured for this deployment; "
                                                 "answers are extracted directly from the evidence.")

if st.button("Ask", type="primary") and question.strip():
    out = answer(question, retriever, docs, k=5, use_llm=use_llm)
    st.subheader("Answer")
    st.write(out["answer"])
    st.caption(f"Mode: {out['mode']} | Retriever: patient-filtered TF-IDF")
    st.subheader("Evidence retrieved")
    ev = docs.set_index("doc_id").loc[out["retrieved"]].reset_index()[["doc_id", "doc_type", "text"]]
    ev.insert(1, "cited", ev["doc_id"].isin(out["cited"]))
    st.dataframe(ev, hide_index=True, width="stretch")
