"""Turn the synthetic EHR tables into retrievable documents and build a ground-truth evaluation set.

Two document types, mirroring real clinical RAG systems:
- one note document per encounter (unstructured text)
- one structured summary per patient (diagnoses, medications, utilization, scores)
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data")


def load_tables(data_dir=DATA):
    return {name: pd.read_csv(Path(data_dir) / f"{name}.csv")
            for name in ("patients", "conditions", "medications", "encounters", "notes")}


def build_documents(t):
    docs = []
    for row in t["notes"].merge(t["encounters"][["encounter_id", "encounter_type"]], on="encounter_id").itertuples():
        docs.append({"doc_id": row.note_id, "patient_id": row.patient_id, "doc_type": "note",
                     "text": f"Patient {row.patient_id}. {row.encounter_type.capitalize()} encounter on {row.date}. {row.text}"})
    enc = t["encounters"]
    for p in t["patients"].itertuples():
        dx = t["conditions"].query("patient_id == @p.patient_id")
        meds = t["medications"].query("patient_id == @p.patient_id")["medication"].tolist()
        e = enc.query("patient_id == @p.patient_id")
        sex = "female" if p.sex == "F" else "male"
        docs.append({"doc_id": f"S{p.patient_id}", "patient_id": p.patient_id, "doc_type": "summary",
                     "text": (f"Patient {p.patient_id} summary. {p.age}-year-old {sex}"
                              f"{', veteran' if p.veteran else ''}. "
                              f"Diagnoses: {'; '.join(dx['description'] + ' (' + dx['icd10'] + ')')}. "
                              f"Medications: {', '.join(meds) if meds else 'none'}. "
                              f"{len(e)} encounters: {int((e.encounter_type == 'emergency').sum())} emergency, "
                              f"{int((e.encounter_type == 'inpatient').sum())} inpatient. "
                              f"Highest PHQ-9 score: {int(e.phq9.max())}.")})
    return pd.DataFrame(docs)


def build_questions(t, n_patients=50, seed=7):
    """Questions with known answers and the documents that contain the evidence."""
    rng = np.random.default_rng(seed)
    pids = rng.choice(t["patients"]["patient_id"], size=n_patients, replace=False)
    enc, notes = t["encounters"], t["notes"]
    qs = []
    for pid in pids:
        meds = sorted(t["medications"].query("patient_id == @pid")["medication"])
        patient_notes = notes.query("patient_id == @pid")
        # Every note lists current medications, so the summary and all of this patient's notes are valid evidence
        qs.append({"qtype": "medications", "patient_id": pid,
                   "question": f"What medications is patient {pid} taking?",
                   "answer": ", ".join(meds), "relevant": [f"S{pid}"] + patient_notes.note_id.tolist()})
        e = enc.query("patient_id == @pid")
        # The summary states the maximum; the note(s) from the visit(s) with that score are also evidence
        max_enc = e.loc[e.phq9 == e.phq9.max(), "encounter_id"]
        qs.append({"qtype": "max_phq9", "patient_id": pid,
                   "question": f"What is the highest PHQ-9 score recorded for patient {pid}?",
                   "answer": str(int(e.phq9.max())),
                   "relevant": [f"S{pid}"] + notes.loc[notes.encounter_id.isin(max_enc), "note_id"].tolist()})
        visit = e.sample(1, random_state=int(rng.integers(1_000_000))).iloc[0]
        note_id = notes.loc[notes.encounter_id == visit.encounter_id, "note_id"].iloc[0]
        qs.append({"qtype": "visit_score", "patient_id": pid,
                   "question": f"What was the PHQ-9 score for patient {pid} at the visit on {visit.date}?",
                   "answer": str(int(visit.phq9)), "relevant": [note_id]})
        ed = e[e.encounter_type == "emergency"]
        if len(ed):
            ed_notes = notes[notes.encounter_id.isin(ed.encounter_id)]
            qs.append({"qtype": "ed_visit", "patient_id": pid,
                       "question": f"When did patient {pid} visit the emergency department?",
                       "answer": ", ".join(sorted(ed.date)), "relevant": ed_notes.note_id.tolist()})
    return pd.DataFrame(qs)
