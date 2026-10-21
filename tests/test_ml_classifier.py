"""Smoke tests against the shipped Tier 2 model (models/risk_classifier.joblib)."""
import pytest

from utils.ml_classifier import load_model, predict_label


@pytest.fixture(scope="module")
def model():
    try:
        return load_model()
    except Exception as exc:  # e.g. Git LFS pointer not pulled
        pytest.skip(f"Tier 2 model not available: {exc}")


def test_model_file_has_expected_format(model):
    assert "pipeline" in model
    assert set(model["labels"]) == {
        "Arbitration", "Liability", "None", "Obligation", "Penalty", "Termination"
    }


def test_prediction_is_a_known_label_with_probability(model):
    label, conf = predict_label("The contractor shall indemnify and hold harmless.")
    assert label in model["labels"]
    assert 0.0 <= conf <= 1.0


def test_empty_input_short_circuits():
    assert predict_label("   ") == ("None", 0.0)
