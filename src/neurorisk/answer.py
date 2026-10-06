"""Answer a question from retrieved evidence, with citations.

Two modes:
- LLM (Claude via the Anthropic API) when ANTHROPIC_API_KEY is set: writes an answer using only the
  retrieved documents and cites them as [doc_id]
- Extractive fallback (free, no API): returns the evidence sentences most similar to the question
"""
import os
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
SYSTEM = ("You answer questions about synthetic patient records using ONLY the evidence provided. "
          "Cite every fact with the document ID in square brackets, e.g. [N0012] or [SP003]. "
          "If the evidence does not contain the answer, say so plainly. Be concise: one to three sentences.")
CITATION = re.compile(r"\[(N\d{4}|SP\d{3})\]")


def evidence_block(docs, doc_ids):
    texts = docs.set_index("doc_id").loc[doc_ids, "text"]
    return "\n".join(f"[{d}] {t}" for d, t in texts.items())


def llm_available():
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def llm_answer(question, docs, doc_ids, model=DEFAULT_MODEL, client=None):
    if client is None:
        import anthropic
        client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model, max_tokens=300, system=SYSTEM,
        messages=[{"role": "user", "content": f"Evidence:\n{evidence_block(docs, doc_ids)}\n\nQuestion: {question}"}],
    )
    return "".join(block.text for block in msg.content if getattr(block, "type", "") == "text").strip()


PATIENT_ONLY = re.compile(r"^Patient P\d{3}( summary)?\.$")


def extractive_answer(question, docs, doc_ids, n_sentences=3):
    sents, seen = [], set()
    for d, text in docs.set_index("doc_id").loc[doc_ids, "text"].items():
        for s in re.split(r"(?<=[.;])\s+", text):
            s = s.strip()
            if s and not PATIENT_ONLY.match(s) and s not in seen:  # skip ID headers and repeated sentences
                seen.add(s)
                sents.append((d, s))
    q = re.sub(r"\bP\d{3}\b|\bpatient\b", " ", question, flags=re.I)  # the ID itself carries no answer
    vec = TfidfVectorizer(stop_words="english").fit([s for _, s in sents] + [q])
    sim = (vec.transform([s for _, s in sents]) @ vec.transform([q]).T).toarray().ravel()
    best = np.argsort(sim)[::-1][:n_sentences]
    return " ".join(f"{sents[i][1]} [{sents[i][0]}]" for i in best)


def answer(question, retriever, docs, k=5, use_llm=None, **llm_kwargs):
    doc_ids = retriever.search(question, k)
    use_llm = llm_available() if use_llm is None else use_llm
    text = llm_answer(question, docs, doc_ids, **llm_kwargs) if use_llm else extractive_answer(question, docs, doc_ids)
    return {"answer": text, "retrieved": doc_ids, "cited": CITATION.findall(text),
            "mode": "llm" if use_llm else "extractive"}
