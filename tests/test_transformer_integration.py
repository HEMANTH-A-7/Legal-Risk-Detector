"""
Integration tests against the real fine-tuned Legal-BERT weights and the
TF-IDF+LogReg pipeline. These load actual model artifacts (~450MB) and are
slower than the unit tests in test_calibration.py / test_severity_estimator.py
/ test_cascade.py, but they are what actually catch a broken model directory,
a broken calibration wire-up, or a cascade that silently stops routing to
Tier 1.
"""

import os

import pytest

TRANSFORMER_DIR = os.path.join("models", "transformer_risk_classifier")

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(TRANSFORMER_DIR, "model.safetensors")),
    reason="fine-tuned Legal-BERT weights are not present in this checkout",
)

LABELS = {"Arbitration", "Liability", "None", "Obligation", "Penalty", "Termination"}


def test_predict_transformer_high_margin_liability_sentence():
    from utils.transformer_classifier import predict_transformer

    label, confidence = predict_transformer(
        "The contractor shall indemnify and hold harmless all parties "
        "from any claims arising out of its performance."
    )
    assert label == "Liability"
    assert confidence > 0.9


def test_predict_transformer_high_margin_arbitration_sentence():
    from utils.transformer_classifier import predict_transformer

    label, confidence = predict_transformer(
        "This agreement shall be governed by the laws of the State of Delaware, "
        "and any dispute shall be resolved exclusively through binding arbitration."
    )
    assert label == "Arbitration"
    assert confidence > 0.9


def test_predict_transformer_empty_input_short_circuits():
    from utils.transformer_classifier import predict_transformer

    assert predict_transformer("") == ("None", 0.0)


def test_calibration_is_actually_applied_before_thresholding():
    """detect_risk_transformer must return the *calibrated* confidence, not raw softmax."""
    from utils.calibration import calibrate_confidence
    from utils.nlp_utils import detect_risk_transformer
    from utils.transformer_classifier import predict_transformer

    sentence = (
        "The contractor shall indemnify and hold harmless all parties "
        "from any claims arising out of its performance."
    )
    raw_label, raw_confidence = predict_transformer(sentence)
    routed_label, routed_confidence = detect_risk_transformer(sentence, enforce_threshold=True)

    assert routed_label == raw_label
    assert routed_confidence == pytest.approx(calibrate_confidence(raw_confidence), abs=1e-6)
    # T > 1 must soften an over-confident raw score.
    assert routed_confidence < raw_confidence


def test_full_auto_cascade_routes_high_margin_sentence_through_transformer():
    from utils.nlp_utils import analyze_risks, segment_sentences

    os.environ["RISK_DETECTOR"] = "auto"
    text = "The contractor shall indemnify and hold harmless all parties from any claims."
    results = analyze_risks(segment_sentences(text))

    assert len(results) == 1
    risk = results[0]
    assert risk["risk_type"] == "Liability"
    assert risk["detector"] == "transformer"
    assert risk["severity"] == "High"
    assert 0.0 <= risk["confidence"] <= 1.0


def test_full_auto_cascade_falls_back_below_transformer_threshold(monkeypatch):
    """A near-impossible threshold forces every sentence to fall through to Tier 2/3."""
    from utils.nlp_utils import analyze_risks, segment_sentences

    monkeypatch.setenv("RISK_DETECTOR", "auto")
    monkeypatch.setenv("TRANSFORMER_THRESHOLD", "0.999999")
    text = "The contractor shall indemnify and hold harmless all parties from any claims."
    results = analyze_risks(segment_sentences(text))

    assert len(results) == 1
    assert results[0]["detector"] in {"ml", "keyword"}
