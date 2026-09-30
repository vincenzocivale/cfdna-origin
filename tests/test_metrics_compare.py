import numpy as np
import pandas as pd
import pytest

from cfdna_origin.evaluation.compare import arm_probabilities, compare_arms, paired_bootstrap, paired_permutation
from cfdna_origin.evaluation.metrics import all_metrics, primary_metrics, softmax

CLASSES = ["A", "B", "C"]
Y = np.array([0, 0, 1, 1, 2, 2])
P = np.array([[.7, .2, .1], [.4, .5, .1], [.1, .8, .1], [.2, .6, .2], [.1, .2, .7], [.5, .1, .4]])


def test_all_metrics_hand_example():
    m = all_metrics(Y, P, CLASSES)
    assert m["confusion_matrix"]["matrix"] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert m["primary"]["macro_f1"] == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)
    assert m["primary"]["balanced_accuracy"] == pytest.approx(2 / 3)
    assert m["primary"]["accuracy"] == pytest.approx(4 / 6)
    assert [m["per_class"][c]["sensitivity"] for c in CLASSES] == pytest.approx([0.5, 1.0, 0.5])
    assert [m["per_class"][c]["specificity"] for c in CLASSES] == pytest.approx([0.75, 0.75, 1.0])
    assert [m["per_class"][c]["n"] for c in CLASSES] == [2, 2, 2]
    assert m["secondary"]["top2_accuracy"] == 1.0
    assert 0.0 <= m["secondary"]["ece"] <= 1.0
    assert m["n_samples"] == 6


def test_ece_perfectly_calibrated_is_zero():
    y = np.array([0, 1, 0, 1])
    assert all_metrics(y, np.eye(2)[y], ["a", "b"])["secondary"]["ece"] == pytest.approx(0.0)


def test_all_metrics_input_validation():
    with pytest.raises(ValueError, match="sum to 1"):
        all_metrics(Y, P * 1.5, CLASSES)
    with pytest.raises(ValueError):
        all_metrics(Y, P[:, :2], CLASSES)


def test_softmax_rows_sum_to_one():
    s = softmax(np.array([[1000.0, 0.0, -5.0], [1.0, 2.0, 3.0]]))
    np.testing.assert_allclose(s.sum(1), 1.0)
    assert np.isfinite(s).all()


def _arms(n=30, seed=0):
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(3), n // 3)
    good = np.full((n, 3), 0.1); good[np.arange(n), y] = 0.8
    bad = rng.dirichlet(np.ones(3), n)
    return y, good, bad


def test_paired_bootstrap():
    y, good, bad = _arms()
    same = paired_bootstrap(y, bad, bad, 3, n_boot=20, seed=0)
    for r in same.values():
        assert r["delta"] == 0 and r["ci95"] == [0.0, 0.0]
    better = paired_bootstrap(y, good, bad, 3, n_boot=40, seed=0)
    for m in ("macro_f1", "balanced_accuracy", "auroc_macro"):
        assert better[m]["delta"] > 0 and better[m]["ci95"][0] > 0
    assert better == paired_bootstrap(y, good, bad, 3, n_boot=40, seed=0)


def test_paired_permutation():
    y, good, bad = _arms()
    same = paired_permutation(y, bad, bad, 3, n_perm=20, seed=0)
    assert all(r["p_two_sided"] == 1.0 for r in same.values())
    diff = paired_permutation(y, good, bad, 3, n_perm=40, seed=0)
    assert all(0 < r["p_two_sided"] <= 1 for r in diff.values())
    assert diff["macro_f1"]["p_two_sided"] < 0.05
    assert diff == paired_permutation(y, good, bad, 3, n_perm=40, seed=0)


def _pred_frame(rep, probs_per_seed, y, ids=None):
    ids = ids or [f"S{i}" for i in range(len(y))]
    rows = []
    for seed, prob in probs_per_seed.items():
        df = pd.DataFrame({"sample_id": ids, "patient_id": [f"P{i}" for i in ids],
                           "true_label": [CLASSES[i] for i in y],
                           "representation": rep, "seed": seed})
        for j, c in enumerate(CLASSES):
            df[f"prob_{c}"] = prob[:, j]
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def test_compare_arms_averages_seeds():
    y, good, bad = _arms()
    rng = np.random.default_rng(1)
    noisy = [rng.dirichlet(np.ones(3), len(y)) for _ in range(2)]
    preds = pd.concat([_pred_frame("f", {1: good, 2: good}, y), _pred_frame("r", {1: noisy[0], 2: noisy[1]}, y)])
    yy, prob, ids = arm_probabilities(preds[preds.representation == "r"], CLASSES)
    order = [int(i[1:]) for i in ids]  # returned in sample-id order
    np.testing.assert_allclose(prob, ((noisy[0] + noisy[1]) / 2)[order])
    np.testing.assert_array_equal(yy, y[order])
    out = compare_arms(preds, CLASSES, [("f", "r"), ("f", "absent")], n_boot=30, n_perm=30)
    assert set(out.metric) == {"macro_f1", "balanced_accuracy", "auroc_macro", "accuracy"}
    row = out.set_index("metric").loc["macro_f1"]
    assert row.b == pytest.approx(primary_metrics(yy, prob, 3)["macro_f1"]) and row.n_patients == len(y)


def test_compare_arms_refuses_different_patients():
    y, good, bad = _arms()
    other_ids = [f"T{i}" for i in range(len(y))]
    preds = pd.concat([_pred_frame("f", {1: good}, y), _pred_frame("r", {1: bad}, y, ids=other_ids)])
    with pytest.raises(ValueError, match="same patients"):
        compare_arms(preds, CLASSES, [("f", "r")], n_boot=5, n_perm=5)


def test_arm_probabilities_incomplete_seeds():
    y, good, _ = _arms()
    preds = _pred_frame("f", {1: good, 2: good}, y)
    with pytest.raises(ValueError, match="incomplete"):
        arm_probabilities(preds.iloc[1:], CLASSES)
