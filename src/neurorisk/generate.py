"""Generate a fully synthetic mental health EHR dataset with known ground truth.

Because we create the data ourselves, we know every true fact (each patient's diagnoses, medications,
highest PHQ-9 score, number of ED visits, risk status). That lets us evaluate the RAG system objectively:
did retrieval find the right records, and is the answer correct?

Run:  python -m neurorisk.generate
Writes data/patients.csv, data/conditions.csv, data/medications.csv, data/encounters.csv, data/notes.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data")
SEED = 42
N_PATIENTS = 200

CONDITIONS = {  # ICD-10 code: (description, prevalence)
    "F33.1": ("Major depressive disorder, recurrent, moderate", 0.45),
    "F41.1": ("Generalized anxiety disorder", 0.40),
    "F43.10": ("Post-traumatic stress disorder", 0.25),
    "F10.20": ("Alcohol dependence", 0.15),
    "F31.9": ("Bipolar disorder", 0.08),
    "G47.00": ("Insomnia", 0.30),
    "I10": ("Essential hypertension", 0.30),
    "E11.9": ("Type 2 diabetes", 0.15),
}
SEROTONERGIC = {"sertraline", "escitalopram", "venlafaxine"}
TREATMENTS = {  # condition -> possible medications
    "F33.1": ["sertraline", "escitalopram", "bupropion", "venlafaxine"],
    "F41.1": ["escitalopram", "sertraline", "buspirone"],
    "F43.10": ["sertraline", "prazosin"],
    "F10.20": ["naltrexone", "acamprosate"],
    "F31.9": ["lithium", "quetiapine"],
    "G47.00": ["trazodone"],
    "I10": ["lisinopril", "amlodipine"],
    "E11.9": ["metformin"],
}


def phq9_severity(score):
    return ("minimal" if score < 5 else "mild" if score < 10 else "moderate" if score < 15
            else "moderately severe" if score < 20 else "severe")


def make_note(rng, pid, enc_type, phq9, gad7, meds, adherent, safety_positive):
    parts = []
    if enc_type == "emergency":
        parts.append(rng.choice([
            "Patient presented to the emergency department with acute anxiety and chest tightness.",
            "Patient brought to the emergency department after a panic episode at work.",
            "Emergency visit for worsening depressed mood and inability to sleep for several days.",
        ]))
    elif enc_type == "inpatient":
        parts.append("Patient admitted to the inpatient psychiatric unit for stabilization.")
    else:
        parts.append("Outpatient behavioral health follow-up visit.")
    if phq9 is not None:
        parts.append(f"PHQ-9 score {phq9}, consistent with {phq9_severity(phq9)} depressive symptoms.")
    if gad7 is not None:
        parts.append(f"GAD-7 score {gad7}.")
    if meds:
        parts.append("Current medications: " + ", ".join(meds) + ".")
        parts.append("Patient reports taking medications as prescribed." if adherent else
                     "Patient reports frequently missing doses due to side effects.")
    if safety_positive:
        parts.append("Safety screening positive for passive thoughts of death; safety plan reviewed and "
                     "crisis resources provided.")
    else:
        parts.append("Safety screening negative.")
    parts.append(rng.choice(["Follow-up in four weeks.", "Referred to psychotherapy.",
                             "Sleep hygiene counseling provided.", "Will coordinate with primary care."]))
    return " ".join(parts)


def generate(n_patients=N_PATIENTS, seed=SEED):
    rng = np.random.default_rng(seed)
    patients, conditions, medications, encounters, notes = [], [], [], [], []
    enc_id = 0
    for i in range(1, n_patients + 1):
        pid = f"P{i:03d}"
        patients.append({"patient_id": pid, "age": int(rng.integers(19, 85)),
                         "sex": str(rng.choice(["F", "M"])), "veteran": bool(rng.random() < 0.25)})

        codes = [c for c, (_, p) in CONDITIONS.items() if rng.random() < p] or ["F41.1"]
        for c in codes:
            conditions.append({"patient_id": pid, "icd10": c, "description": CONDITIONS[c][0]})
        meds = []
        for c in codes:
            m = str(rng.choice(TREATMENTS[c]))
            # Clinical realism: at most one serotonergic antidepressant (SSRI/SNRI) per patient
            if m in SEROTONERGIC and any(x in SEROTONERGIC for x in meds):
                continue
            meds.append(m)
        meds = sorted(set(meds))
        for m in meds:
            medications.append({"patient_id": pid, "medication": m})

        adherent = bool(rng.random() < 0.7)
        depressed = "F33.1" in codes or "F31.9" in codes
        base = rng.integers(8, 18) if depressed else rng.integers(0, 9)
        trend = rng.choice([-2, 0, 2])  # improving, stable, or worsening across visits
        dates = sorted(pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 900, rng.integers(3, 8)), unit="D"))
        for k, d in enumerate(dates):
            enc_id += 1
            enc_type = str(rng.choice(["outpatient", "outpatient", "outpatient", "emergency", "inpatient"],
                                      p=[0.3, 0.3, 0.25, 0.1, 0.05]))
            phq9 = int(np.clip(base + trend * k + rng.integers(-2, 3), 0, 27))
            gad7 = int(np.clip(phq9 * 0.7 + rng.integers(-3, 4), 0, 21))
            safety = bool(phq9 >= 20 and rng.random() < 0.5)
            encounters.append({"encounter_id": f"E{enc_id:04d}", "patient_id": pid, "date": d.date().isoformat(),
                               "encounter_type": enc_type, "phq9": phq9, "gad7": gad7,
                               "safety_screen_positive": safety})
            notes.append({"note_id": f"N{enc_id:04d}", "encounter_id": f"E{enc_id:04d}", "patient_id": pid,
                          "date": d.date().isoformat(),
                          "text": make_note(rng, pid, enc_type, phq9, gad7, meds, adherent, safety)})

    patients, conditions, medications = map(pd.DataFrame, (patients, conditions, medications))
    encounters, notes = pd.DataFrame(encounters), pd.DataFrame(notes)

    # Ground-truth risk label, defined by a transparent rule
    agg = encounters.groupby("patient_id").agg(
        max_phq9=("phq9", "max"),
        ed_visits=("encounter_type", lambda s: int((s == "emergency").sum())),
        inpatient=("encounter_type", lambda s: int((s == "inpatient").sum())),
        safety_positive=("safety_screen_positive", "any"),
    ).reset_index()
    agg["high_risk"] = (agg["max_phq9"] >= 20) | (agg["ed_visits"] >= 2) | (agg["inpatient"] >= 1) | agg["safety_positive"]
    patients = patients.merge(agg[["patient_id", "high_risk"]], on="patient_id")
    return {"patients": patients, "conditions": conditions, "medications": medications,
            "encounters": encounters, "notes": notes}


def main():
    DATA.mkdir(exist_ok=True)
    tables = generate()
    for name, df in tables.items():
        df.to_csv(DATA / f"{name}.csv", index=False)
        print(f"{name:<12} {len(df):>5} rows -> data/{name}.csv")
    print(f"High-risk patients: {int(tables['patients']['high_risk'].sum())} of {len(tables['patients'])}")


if __name__ == "__main__":
    main()
