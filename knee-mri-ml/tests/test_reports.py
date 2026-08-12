from knee_mri.data.reports import RSNA_TASKS, ReportLabeler, State, soft_targets


def test_positive_negative_unmentioned():
    labeler = ReportLabeler()
    text = (
        "MRI of the knee. There is a tear of the anterior cruciate ligament. "
        "The medial meniscus is normal. Moderate joint effusion."
    )
    states = labeler.label(text)
    assert states["acl"] is State.POSITIVE
    assert states["medial_meniscus"] is State.NEGATIVE
    assert states["effusion"] is State.POSITIVE
    assert states["fracture"] is State.UNMENTIONED


def test_negation_cues():
    labeler = ReportLabeler()
    states = labeler.label("No fracture. The ACL is intact. No popliteal cyst.")
    assert states["fracture"] is State.NEGATIVE
    assert states["acl"] is State.NEGATIVE
    assert states["bakers_cyst"] is State.NEGATIVE


def test_positive_outranks_negative():
    labeler = ReportLabeler()
    # Same finding asserted in one sentence, denied in another → positive wins.
    states = labeler.label("Tear of the medial meniscus. The medial meniscus is normal.")
    assert states["medial_meniscus"] is State.POSITIVE


def test_multilingual_starter_terms():
    labeler = ReportLabeler()
    states = labeler.label("Ruptura del ligamento cruzado anterior. Derrame articular.")
    assert states["acl"] is State.POSITIVE
    assert states["effusion"] is State.POSITIVE


def test_empty_report_is_all_unmentioned():
    labeler = ReportLabeler()
    assert all(s is State.UNMENTIONED for s in labeler.label("").values())


def test_soft_targets_weights_ordering():
    states = {
        "acl": State.POSITIVE,
        "mcl": State.NEGATIVE,
        "fracture": State.UNMENTIONED,
    }
    targets, weights = soft_targets(states, ["acl", "mcl", "fracture"])
    t = dict(zip(["acl", "mcl", "fracture"], targets))
    w = dict(zip(["acl", "mcl", "fracture"], weights))
    # Target ordering: positive > unmentioned > explicit negative.
    assert t["acl"] > t["fracture"] > t["mcl"]
    # Unmentioned findings carry reduced confidence.
    assert w["fracture"] < w["acl"] == w["mcl"]


def test_all_rsna_tasks_have_term_tables():
    labeler = ReportLabeler(RSNA_TASKS)
    assert set(labeler.tasks) == set(RSNA_TASKS)
    assert len(RSNA_TASKS) == 12
