from neurorisk.generate import generate


def test_generation_is_reproducible():
    a, b = generate(n_patients=20, seed=1), generate(n_patients=20, seed=1)
    assert a["notes"].equals(b["notes"])


def test_tables_are_consistent():
    t = generate(n_patients=30, seed=2)
    pids = set(t["patients"]["patient_id"])
    for name in ("conditions", "medications", "encounters", "notes"):
        assert set(t[name]["patient_id"]) <= pids
    assert t["encounters"]["phq9"].between(0, 27).all()
    assert len(t["notes"]) == len(t["encounters"])


def test_every_patient_has_a_condition_and_encounters():
    t = generate(n_patients=30, seed=3)
    assert set(t["conditions"]["patient_id"]) == set(t["patients"]["patient_id"])
    assert t["encounters"].groupby("patient_id").size().min() >= 3


def test_at_most_one_serotonergic_antidepressant_per_patient():
    from neurorisk.generate import SEROTONERGIC
    t = generate(n_patients=200, seed=42)
    per_patient = t["medications"][t["medications"]["medication"].isin(SEROTONERGIC)].groupby("patient_id").size()
    assert (per_patient <= 1).all()
