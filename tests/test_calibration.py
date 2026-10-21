import math

import pytest

from utils.calibration import calibrate_confidence, compute_ece, temperature_scale


def test_known_value_from_readme():
    # p=0.95, T=1.5 -> ~0.877 (worked example in the README)
    assert calibrate_confidence(0.95, 1.5) == pytest.approx(0.877, abs=1e-3)


def test_t_equal_one_is_identity():
    for p in (0.1, 0.5, 0.73, 0.99):
        assert calibrate_confidence(p, 1.0) == pytest.approx(p, abs=1e-4)


def test_t_above_one_softens_overconfidence():
    for p in (0.6, 0.8, 0.95, 0.99):
        assert calibrate_confidence(p, 1.5) < p


def test_calibration_preserves_ordering():
    raw = [0.55, 0.7, 0.85, 0.97]
    cal = [calibrate_confidence(p, 1.5) for p in raw]
    assert cal == sorted(cal)


def test_boundary_probabilities_pass_through():
    # logit is undefined at 0 and 1; the function returns the input unchanged
    assert calibrate_confidence(0.0, 1.5) == 0.0
    assert calibrate_confidence(1.0, 1.5) == 1.0


def test_temperature_scale_returns_a_distribution():
    probs = temperature_scale([2.0, 1.0, 0.1], temperature=1.5)
    assert sum(probs) == pytest.approx(1.0)
    assert probs == sorted(probs, reverse=True)


def test_temperature_scale_rejects_non_positive_t():
    with pytest.raises(ValueError):
        temperature_scale([1.0, 2.0], temperature=0)


def test_ece_perfect_and_worst_case():
    assert compute_ece([]) == 0.0
    # Always confident and always right -> perfectly calibrated
    assert compute_ece([(1.0, True)] * 10) == 0.0
    # Always confident and always wrong -> maximal error
    assert compute_ece([(1.0, False)] * 10) == pytest.approx(1.0)


def test_ece_is_bounded():
    preds = [(i / 20, i % 3 == 0) for i in range(20)]
    ece = compute_ece(preds)
    assert 0.0 <= ece <= 1.0 and not math.isnan(ece)
