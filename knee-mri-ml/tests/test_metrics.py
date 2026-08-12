import math

from knee_mri.engine.metrics import compute_metrics


def test_perfect_predictions():
    labels = [[1, 0], [0, 1], [1, 1], [0, 0]]
    probs = [[0.9, 0.1], [0.2, 0.8], [0.99, 0.7], [0.05, 0.3]]
    m = compute_metrics(labels, probs, ["a", "b"])
    assert m["a"]["auc"] == 1.0
    assert m["b"]["auc"] == 1.0
    assert m["macro"]["auc"] == 1.0


def test_single_class_auc_is_nan_but_acc_defined():
    labels = [[1], [1], [1]]           # only one class present
    probs = [[0.7], [0.2], [0.9]]
    m = compute_metrics(labels, probs, ["only"])
    assert math.isnan(m["only"]["auc"])
    assert 0.0 <= m["only"]["acc"] <= 1.0
    # Macro AUC ignores NaN tasks.
    assert math.isnan(m["macro"]["auc"])
