"""Rule-based extraction of finding labels from radiology report text.

In the RSNA Knee Abnormality Detection dataset only ~1% of training studies
carry gold image-level labels; the rest have free-text radiology reports (in
12 languages in the real data). This module turns a report into one of three
states per finding:

    POSITIVE     — the finding is asserted ("full-thickness tear of the ACL")
    NEGATIVE     — the finding is explicitly denied ("the ACL is intact")
    UNMENTIONED  — the report says nothing about the finding

States are then converted to *soft targets* + *confidence weights* for
training (see :func:`soft_targets`), so report-derived studies contribute a
weaker, calibrated signal next to the small gold set.

The matcher is sentence-scoped: a sentence votes NEGATIVE if it contains a
finding term alongside a negation cue, POSITIVE if it contains the term
without one. POSITIVE evidence anywhere in the report wins over NEGATIVE
(reports often deny one structure and assert another).

The term tables below cover English plus a starter set of common non-English
radiology terms. Extending coverage = adding strings here; the state machine
does not change.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Dict, Iterable, List, Sequence, Tuple

# The 12 competition findings, in submission order.
RSNA_TASKS: List[str] = [
    "acl",
    "mcl",
    "medial_meniscus",
    "lateral_meniscus",
    "medial_oa",
    "lateral_oa",
    "pf_oa",
    "effusion",
    "synovitis",
    "bakers_cyst",
    "contusion",
    "fracture",
]


class State(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNMENTIONED = "unmentioned"


# Phrases that assert a finding for each task. Matched case-insensitively as
# whole words after basic normalization. Non-English entries mark the pattern
# the real multilingual tables follow.
_FINDING_TERMS: Dict[str, List[str]] = {
    "acl": [
        r"acl", r"anterior cruciate ligament", r"ligamento cruzado anterior",
        r"vorderes? kreuzband", r"ligament croise anterieur",
    ],
    "mcl": [
        r"mcl", r"medial collateral ligament", r"ligamento colateral medial",
        r"innenband",
    ],
    "medial_meniscus": [
        r"medial meniscus", r"menisco (?:medial|interno)", r"innenmeniskus",
        r"menisque interne",
    ],
    "lateral_meniscus": [
        r"lateral meniscus", r"menisco (?:lateral|externo)", r"aussenmeniskus",
        r"menisque externe",
    ],
    "medial_oa": [
        r"medial (?:compartment|tibiofemoral) (?:osteoarthritis|arthrosis|degenerative change)",
        r"medial joint space (?:narrowing|loss)",
        r"medial (?:femoral|tibial) (?:condyle |plateau )?(?:chondral|cartilage) (?:loss|thinning|defect|wear)",
    ],
    "lateral_oa": [
        r"lateral (?:compartment|tibiofemoral) (?:osteoarthritis|arthrosis|degenerative change)",
        r"lateral joint space (?:narrowing|loss)",
        r"lateral (?:femoral|tibial) (?:condyle |plateau )?(?:chondral|cartilage) (?:loss|thinning|defect|wear)",
    ],
    "pf_oa": [
        r"patellofemoral (?:osteoarthritis|arthrosis|degenerative change|compartment)",
        r"(?:patellar|trochlear) (?:chondral|cartilage) (?:loss|thinning|defect|wear|damage)",
        r"chondromalacia patell?ae?",
    ],
    "effusion": [
        r"(?:joint )?effusion", r"derrame articular", r"gelenkerguss",
        r"epanchement",
    ],
    "synovitis": [
        r"synovitis", r"synovial (?:thickening|proliferation|hypertrophy)",
        r"sinovitis", r"synovite",
    ],
    "bakers_cyst": [
        r"baker'?s? cyst", r"popliteal cyst", r"quiste de baker",
        r"baker-?zyste", r"kyste poplite",
    ],
    "contusion": [
        r"(?:bone|osseous|marrow) (?:contusion|bruise)", r"bone marrow (?:edema|oedema)",
        r"contusion osea", r"knochenkontusion", r"contusion osseuse",
    ],
    "fracture": [
        r"fracture", r"fractura", r"fraktur",
    ],
}

# Negation / normality cues. A sentence containing a finding term AND one of
# these votes NEGATIVE for that finding.
_NEGATION_CUES: List[str] = [
    r"\bno\b", r"\bwithout\b", r"\bnot?\s+(?:seen|identified|present|evident|demonstrated)\b",
    r"\bintact\b", r"\bnormal\b", r"\bunremarkable\b", r"\bpreserved\b",
    r"\babsen(?:t|ce)\b", r"\bnegative for\b", r"\bfrei\b", r"\bkein[e]?\b",
    r"\bsin\b", r"\bsans\b", r"\bpas de\b", r"\bruled? out\b",
]

_SENTENCE_SPLIT = re.compile(r"[.;\n]+")


def _normalize(text: str) -> str:
    text = text.lower()
    # Strip accents crudely (enough for the starter term tables).
    for src, dst in (("é", "e"), ("è", "e"), ("ê", "e"), ("á", "a"), ("í", "i"),
                     ("ó", "o"), ("ú", "u"), ("ñ", "n"), ("ä", "a"), ("ö", "o"),
                     ("ü", "u"), ("ß", "ss")):
        text = text.replace(src, dst)
    return re.sub(r"\s+", " ", text)


class ReportLabeler:
    """Extract per-finding states from radiology report text."""

    def __init__(self, tasks: Sequence[str] = RSNA_TASKS) -> None:
        unknown = set(tasks) - set(_FINDING_TERMS)
        if unknown:
            raise ValueError(f"no term table for tasks: {sorted(unknown)}")
        self.tasks = list(tasks)
        self._term_res: Dict[str, re.Pattern] = {
            task: re.compile(r"\b(?:" + "|".join(_FINDING_TERMS[task]) + r")\b")
            for task in self.tasks
        }
        self._negation_re = re.compile("|".join(_NEGATION_CUES))

    def label(self, report_text: str) -> Dict[str, State]:
        """Return a state for every task given one report."""
        states: Dict[str, State] = {t: State.UNMENTIONED for t in self.tasks}
        if not report_text or not report_text.strip():
            return states

        for sentence in _SENTENCE_SPLIT.split(_normalize(report_text)):
            if not sentence.strip():
                continue
            negated = bool(self._negation_re.search(sentence))
            for task in self.tasks:
                if not self._term_res[task].search(sentence):
                    continue
                if negated:
                    if states[task] is State.UNMENTIONED:
                        states[task] = State.NEGATIVE
                else:
                    # Positive evidence anywhere outranks negative elsewhere.
                    states[task] = State.POSITIVE
        return states


def soft_targets(
    states: Dict[str, State],
    tasks: Sequence[str],
    positive_target: float = 0.9,
    negative_target: float = 0.05,
    unmentioned_target: float = 0.15,
    report_weight: float = 1.0,
    unmentioned_weight_scale: float = 0.25,
) -> Tuple[List[float], List[float]]:
    """Convert report states into (targets, weights) lists aligned with ``tasks``.

    Report-derived labels are noisy (~82% agreement with gold reads in the
    RSNA data), so targets are soft rather than hard 0/1, and unmentioned
    findings get both a low prior target and a reduced weight.
    """
    targets: List[float] = []
    weights: List[float] = []
    for task in tasks:
        state = states.get(task, State.UNMENTIONED)
        if state is State.POSITIVE:
            targets.append(positive_target)
            weights.append(report_weight)
        elif state is State.NEGATIVE:
            targets.append(negative_target)
            weights.append(report_weight)
        else:
            targets.append(unmentioned_target)
            weights.append(report_weight * unmentioned_weight_scale)
    return targets, weights
