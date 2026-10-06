"""Four retrieval strategies to compare.

- TF-IDF: keyword matching; strong on exact identifiers and dates
- Embeddings: semantic similarity (all-MiniLM-L6-v2); strong on meaning, weak on exact identifiers
- Hybrid: combines both rankings with reciprocal rank fusion
- Patient-filtered retrieval: restrict to the patient named in the question, then rank
  (how production clinical systems usually work); can wrap TF-IDF, embeddings, or both via hybrid fusion
"""
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

PATIENT_ID = re.compile(r"\bP\d{3}\b")


class TfidfRetriever:
    name = "tfidf"

    def __init__(self, docs):
        self.ids = docs["doc_id"].tolist()
        self.patients = docs["patient_id"].to_numpy()
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vec.fit_transform(docs["text"])

    def scores(self, query):
        return (self.matrix @ self.vec.transform([query]).T).toarray().ravel()

    def search(self, query, k=5):
        top = np.argsort(self.scores(query))[::-1][:k]
        return [self.ids[i] for i in top]


class EmbeddingRetriever:
    name = "embeddings"

    def __init__(self, docs, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.ids = docs["doc_id"].tolist()
        self.patients = docs["patient_id"].to_numpy()
        self.emb = self.model.encode(docs["text"].tolist(), normalize_embeddings=True,
                                     convert_to_numpy=True, show_progress_bar=False)

    def scores(self, query):
        q = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        return self.emb @ q  # cosine similarity (vectors are normalized)

    def search(self, query, k=5):
        top = np.argsort(self.scores(query))[::-1][:k]
        return [self.ids[i] for i in top]


class PatientFilteredRetriever:
    """Wrap any retriever that exposes scores(); search only the documents of the patient named in the question."""

    def __init__(self, base):
        self.base = base
        self.name = f"{base.name}_patient_filter"

    def search(self, query, k=5):
        s = self.base.scores(query)
        m = PATIENT_ID.search(query)
        if m:  # only consider documents belonging to the patient named in the question
            s = np.where(self.base.patients == m.group(), s, -np.inf)
        top = np.argsort(s)[::-1][:k]
        return [self.base.ids[i] for i in top if np.isfinite(s[i])]


class HybridRetriever:
    name = "hybrid_rrf"

    def __init__(self, *retrievers, depth=50, c=60):
        self.retrievers, self.depth, self.c = retrievers, depth, c

    def search(self, query, k=5):
        fused = {}
        for r in self.retrievers:  # reciprocal rank fusion: reward documents ranked high by any retriever
            for rank, doc_id in enumerate(r.search(query, self.depth), start=1):
                fused[doc_id] = fused.get(doc_id, 0) + 1 / (self.c + rank)
        return sorted(fused, key=fused.get, reverse=True)[:k]
