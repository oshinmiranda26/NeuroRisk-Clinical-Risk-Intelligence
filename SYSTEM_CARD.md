# System Card: NeuroRisk Cited Question Answering

*A retrieval-augmented generation (RAG) system: this card documents the system, not a single model.*

## System details
- **Retrieval:** patient-filtered hybrid retrieval (TF-IDF and all-MiniLM-L6-v2 sentence embeddings combined with
  reciprocal rank fusion), restricted to documents of the patient named in the question. The public demo uses
  patient-filtered TF-IDF to stay lightweight.
- **Answering:** extractive mode (default, no external API) or LLM generation via the Anthropic API when a key is
  configured. Both cite document IDs for every answer.
- **Developer:** Oshin Miranda. **License:** MIT.

## Intended use
- **Intended:** demonstrating how to build and evaluate a cited clinical question-answering system; teaching.
- **Out of scope:** real patients, real records, or any clinical decision. Not for clinical use.

## Data
Fully synthetic mental health records (200 patients, 982 notes, plus structured patient summaries) generated with
known ground truth. No real patient data.

## Evaluation
| Component | Metric | Result |
|---|---|---|
| Retrieval (176 ground-truth questions) | hit@5, patient-filtered hybrid | 0.989 |
| Retrieval | hit@3, embeddings only (no patient filter) | 0.182 |
| Answers, extractive mode (40 questions) | Correct | 0.63 |
| Answers, extractive mode | Cites a correct document | 0.95 |
| Answers, extractive mode | Cites only retrieved documents | 1.00 |

## Safeguards built in
- **Patient filtering:** prevents retrieving other patients' records, the main failure of embedding-only search.
- **Grounding and citations:** answers must cite the documents used; a citation check flags any document cited that
  was not retrieved (fabricated sources).
- **Visible evidence:** the demo shows the retrieved documents and marks which were cited, so users can verify.

## Risks and limitations
- Synthetic, template-based notes are far cleaner than real documentation; real performance would be lower.
- Questions name the patient explicitly; a real system would take patient context from the application.
- Extractive answers can cite the right note yet quote the wrong sentence (20% correct on visit-specific questions).
- Cohort-level questions require structured queries, not document retrieval.
- Any real deployment would need privacy review, access controls, audit logging, and clinician oversight.
