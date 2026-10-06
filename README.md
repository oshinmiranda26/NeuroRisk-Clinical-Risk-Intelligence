# NeuroRisk: Evaluated Retrieval-Augmented Question Answering over Mental Health Records

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://neurorisk-clinical.streamlit.app)

**[Try the live demo](https://neurorisk-clinical.streamlit.app)**: ask a question about a synthetic patient and get an answer with citations to the exact notes it used.

> **Finding:** for patient-level clinical questions, the design of retrieval mattered more than the choice of model. Pure embedding search failed on patient identifiers (hit@3 0.18), retrieving other patients' records; naive hybrid fusion made results worse; filtering to the named patient and then fusing keyword and semantic search reached hit@5 0.99. Yet good retrieval did not guarantee good answers: extractive answering was correct for only 63% of questions, showing that generation, not retrieval, was the bottleneck.

## Why this matters

Clinicians and researchers ask patient-specific questions ("What is this patient taking?", "When were they last in the emergency department?"). A retrieval-augmented system must find the right patient's records, ground its answer in them, and cite its sources. In a clinical setting, retrieving another patient's record is not a minor error. This project measures each of those steps against known ground truth instead of judging answers by eye.

## Data: synthetic, with known ground truth

`src/neurorisk/generate.py` creates a fully synthetic mental health EHR dataset (no real patient data):

- **200 patients** with age, sex, and veteran status
- **442 diagnoses** with ICD-10 codes: major depressive disorder, generalized anxiety disorder, PTSD, alcohol dependence, bipolar disorder, insomnia, plus hypertension and type 2 diabetes
- **405 medications** matched to diagnoses, with a clinical-realism rule: at most one serotonergic antidepressant (SSRI/SNRI) per patient, enforced by a test
- **982 encounters** (outpatient, emergency, inpatient) with PHQ-9 and GAD-7 trajectories that improve, stay stable, or worsen
- **982 clinical notes**, one per encounter, consistent with that visit's scores, medications, adherence, and safety screening
- A transparent **high-risk label** (64 of 200 patients): PHQ-9 of 20 or more, two or more ED visits, any inpatient stay, or a positive safety screen

Because the data is generated, every true fact is known, which makes objective evaluation possible.

## How it works

```
Question ("What medications is patient P012 taking?")
        |
Patient filter: restrict to documents of the patient named in the question
        |
Retrieval: TF-IDF and sentence embeddings (all-MiniLM-L6-v2), fused with reciprocal rank fusion
        |
Top-5 documents (clinical notes + a structured patient summary)
        |
Answer with citations: Claude (when an API key is set) or a free extractive mode
        |
"Current medications: ... [N0049]"
```

## Retrieval evaluation

176 questions generated from ground truth, in four types: current medications, highest PHQ-9, PHQ-9 at a specific visit, and emergency department visit dates. Each question records which documents contain the answer.

| Retriever | hit@1 | hit@3 | hit@5 | MRR | ms/query |
|---|---|---|---|---|---|
| TF-IDF | 0.716 | 0.841 | 0.915 | **0.784** | **0.3** |
| Embeddings | 0.097 | 0.182 | 0.227 | 0.145 | 3.8 |
| Hybrid (RRF) | 0.500 | 0.653 | 0.744 | 0.583 | 4.0 |
| TF-IDF, patient-filtered | 0.716 | 0.841 | 0.915 | 0.784 | 0.3 |
| Embeddings, patient-filtered | 0.625 | 0.852 | 0.949 | 0.747 | 3.7 |
| **Hybrid, patient-filtered** | 0.642 | **0.909** | **0.989** | 0.772 | 4.0 |

hit@k: share of questions with a correct document in the top k. MRR: mean reciprocal rank of the first correct document.

**What the results show**
- **Embeddings fail on identifiers.** To an embedding model, "P012" and "P021" are nearly identical, so it retrieves documents that sound right but belong to other patients.
- **Fusion is not automatically better.** Combining a strong and a weak retriever with equal weight dragged results down (hit@3 0.84 to 0.65).
- **Filtering by patient fixes embeddings** (hit@3 0.18 to 0.85), mirroring how production clinical systems scope searches to one patient.
- **Keyword and semantic retrieval are complementary.** Within a patient, embeddings excelled on ED-visit questions (hit@3 1.00 versus 0.58 for TF-IDF), while TF-IDF excelled on questions hinging on exact dates (1.00 versus 0.60). The filtered hybrid captures both, missing a correct document in the top 5 for only about 1% of questions.
- **TF-IDF still ranks first most often** (best MRR), but RAG passes the top 5 to the answer step, so recall at 5 matters more than rank 1.

A lesson from building the evaluation: an early version counted only the patient summary as correct evidence for medication questions, and TF-IDF scored 0.10. Every clinical note also lists current medications, so notes are equally valid evidence; after correcting the relevance definitions, TF-IDF scored 1.00. Defining relevance correctly is the hardest part of retrieval evaluation.

## Answer evaluation

40 questions (10 per type), retrieved with the patient-filtered hybrid and answered in the free extractive mode. Correctness is checked automatically against ground truth.

| Question type | Correct | Cites a correct document | Cites only retrieved documents |
|---|---|---|---|
| Medications | 1.00 | 1.00 | 1.00 |
| Highest PHQ-9 | 0.80 | 0.80 | 1.00 |
| ED visit dates | 0.50 | 1.00 | 1.00 |
| PHQ-9 at a specific visit | 0.20 | 1.00 | 1.00 |
| **Overall** | **0.63** | **0.95** | **1.00** |

**Retrieval was not the bottleneck.** For visit-specific questions, the answer cited the correct note every time but was correct only 20% of the time: extraction picked a PHQ-9 sentence from another visit, because sentences like "PHQ-9 score 14" and "PHQ-9 score 9" look equally similar to the question. Extraction also quotes visit descriptions without pulling out the dates. Closing this gap is the job of an LLM that reasons over the evidence. LLM-mode evaluation is supported (`python -m neurorisk.evaluate_answers`, requires an Anthropic API key); the results above are for the free extractive mode.

The "cites only retrieved documents" metric catches fabricated citations, a form of hallucination an LLM can produce; extractive mode cannot invent sources by design.

## Live demo

The [Streamlit app](https://neurorisk-clinical.streamlit.app) answers questions with citations and shows the retrieved evidence, marking which documents were cited. To stay lightweight on free hosting, it uses patient-filtered TF-IDF retrieval and extractive answers; LLM generation turns on automatically when an API key is configured.

## How to run

```bash
pip install -r requirements.txt
pip install -e .
pytest                                          # 12 tests, including the clinical-realism rule
python -m neurorisk.generate                    # create the synthetic dataset
python -m neurorisk.evaluate                    # retrieval evaluation (6 retrievers)
python -m neurorisk.evaluate_answers --extractive   # answer evaluation, free mode
streamlit run demo/streamlit_app.py             # demo app locally
```

## Limitations

- Notes are generated from templates, so they are cleaner and more uniform than real clinical notes (no abbreviations, typos, copy-forward text, or contradictory entries); real-world performance would be lower.
- Questions name the patient explicitly; real systems usually get patient context from the user interface instead.
- Cohort-level questions ("Which patients are high risk?") require structured queries over the tables, not document retrieval; this system is designed for patient-level questions.
- Results come from a single synthetic dataset and seed.

## Project structure

```
src/neurorisk/generate.py           synthetic EHR generator with ground truth
src/neurorisk/documents.py          notes and patient summaries as documents; evaluation questions
src/neurorisk/retrieve.py           TF-IDF, embeddings, hybrid fusion, patient filtering
src/neurorisk/evaluate.py           retrieval evaluation
src/neurorisk/answer.py             cited answers (Claude or extractive)
src/neurorisk/evaluate_answers.py   answer correctness and citation checks
demo/streamlit_app.py               live demo
```

## Author

**Oshin Miranda, PhD** | [LinkedIn](https://www.linkedin.com/in/oshin-miranda-ph-d-9551781b5/) | [Google Scholar](https://scholar.google.com/citations?hl=en&user=fkvhYbgAAAAJ&view_op=list_works&sortby=pubdate)
