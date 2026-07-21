import json
import math
import os

import pytest

from utils.calibration import (
    DEFAULT_TEMPERATURE,
    _CONFIG_PATH,
    calibrate_confidence,
    compute_ece,
    temperature_scale,
)


def test_default_temperature_matches_fitted_config():
    with open(_CONFIG_PATH) as f:
        config = json.load(f)
    assert DEFAULT_TEMPERATURE == pytest.approx(config["temperature"])
    assert DEFAULT_TEMPERATURE != 1.5, "must not regress to the old stale default"


def test_calibrate_confidence_identity_at_temperature_one():
    assert calibrate_confidence(0.95, temperature=1.0) == pytest.approx(0.95, abs=1e-9)


def test_calibrate_confidence_softens_overconfidence_above_one():
    raw = 0.95
    calibrated = calibrate_confidence(raw, temperature=1.144)
    assert calibrated < raw
    assert calibrated == pytest.approx(0.9292, abs=1e-3)


def test_calibrate_confidence_passes_through_boundary_values():
    assert calibrate_confidence(1.0, temperature=1.144) == 1.0
    assert calibrate_confidence(0.0, temperature=1.144) == 0.0


def test_calibrate_confidence_rejects_non_positive_temperature():
    with pytest.raises(ValueError):
        temperature_scale([1.0, 2.0], temperature=0)


def test_temperature_scale_sums_to_one():
    probs = temperature_scale([2.0, 1.0, 0.1], temperature=1.0)
    assert sum(probs) == pytest.approx(1.0)
    assert probs == [pytest.approx(v, abs=1e-6) for v in
                      [0.6590011388859679, 0.24243297070471392, 0.09856589040931818]]


def test_temperature_scale_flattens_distribution_as_temperature_rises():
    low_t = temperature_scale([2.0, 1.0, 0.1], temperature=1.0)
    high_t = temperature_scale([2.0, 1.0, 0.1], temperature=2.0)
    assert max(high_t) < max(low_t), "higher T must flatten (de-sharpen) the distribution"
    assert sum(high_t) == pytest.approx(1.0)


def test_compute_ece_matches_hand_worked_example():
    # One bin (conf=0.9), 4 samples, 2 correct -> |0.9 - 0.5| weighted by 4/4
    predictions = [(0.9, True), (0.9, True), (0.9, False), (0.9, False)]
    assert compute_ece(predictions) == pytest.approx(0.4)


def test_compute_ece_is_zero_for_perfect_calibration():
    # avg confidence (0.75) == avg accuracy (3/4 = 0.75) within the single bin -> ECE == 0
    predictions = [(0.75, True), (0.75, True), (0.75, True), (0.75, False)]
    ece = compute_ece(predictions, n_bins=4)
    assert ece == pytest.approx(0.0, abs=1e-9)


def test_compute_ece_empty_predictions_is_zero():
    assert compute_ece([]) == 0.0
