import pytest

from utils.ml_classifier import predict_label


@pytest.fixture(autouse=True)
def _require_model():
    import os

    if not os.path.exists(os.path.join("models", "risk_classifier.joblib")):
        pytest.skip("models/risk_classifier.joblib not present in this checkout")


def test_empty_sentence_returns_none_with_zero_confidence():
    assert predict_label("") == ("None", 0.0)
    assert predict_label("   ") == ("None", 0.0)


def test_predict_label_returns_a_known_class_with_valid_confidence():
    label, confidence = predict_label(
        "The contractor shall indemnify and hold harmless all parties from any claims."
    )
    assert label in {"Arbitration", "Liability", "None", "Obligation", "Penalty", "Termination"}
    assert 0.0 <= confidence <= 1.0


def test_predict_label_is_deterministic_for_the_same_input():
    sentence = "This agreement shall be governed by the laws of the State of Delaware."
    first = predict_label(sentence)
    second = predict_label(sentence)
    assert first == second


def test_predict_label_confidence_is_never_negative_or_above_one():
    sentences = [
        "The parties agree to have lunch together sometime next week.",
        "Late payments are subject to a 1.5% monthly surcharge.",
        "Any dispute shall be resolved through binding arbitration.",
    ]
    for sentence in sentences:
        _, confidence = predict_label(sentence)
        assert 0.0 <= confidence <= 1.0
