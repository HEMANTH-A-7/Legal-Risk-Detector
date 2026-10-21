"""
Routing tests for the 3-tier cascade.

Tier 1 (transformer) and Tier 2 (TF-IDF + LogReg) are replaced with stubs so
these tests run in milliseconds, without torch or model weights, and assert
only the routing rules: which tier answers, and when a sentence falls through.
"""
import sys
import types

import pytest

from utils import nlp_utils

RISKY = "The contractor shall indemnify all parties."


@pytest.fixture
def tiers(monkeypatch):
    """Install controllable fakes for Tier 1 and Tier 2."""
    state = {"t1": ("Liability", 0.99), "t2": ("Penalty", 0.9), "t1_calls": 0, "t2_calls": 0}

    def fake_transformer(sentence):
        state["t1_calls"] += 1
        return state["t1"]

    def fake_ml(sentence):
        state["t2_calls"] += 1
        return state["t2"]

    fake_module = types.ModuleType("utils.transformer_classifier")
    fake_module.predict_transformer = fake_transformer
    monkeypatch.setitem(sys.modules, "utils.transformer_classifier", fake_module)
    monkeypatch.setattr(nlp_utils, "predict_label", fake_ml)
    return state


def _detect(sentence=RISKY):
    results = nlp_utils.analyze_risks([sentence])
    return results[0] if results else None


def test_confident_tier1_answers_and_skips_tier2(tiers):
    r = _detect()
    assert r["detector"] == "transformer"
    assert r["risk_type"] == "Liability"
    assert tiers["t2_calls"] == 0


def test_confidence_is_calibrated_before_threshold(tiers, monkeypatch):
    # Raw 0.65 would clear a 0.62 threshold, but calibrated with T=1.5 it is
    # ~0.602, so the sentence must fall through to Tier 2.
    monkeypatch.setenv("TRANSFORMER_THRESHOLD", "0.62")
    tiers["t1"] = ("Liability", 0.65)
    r = _detect()
    assert r["detector"] == "ml"
    assert tiers["t2_calls"] == 1


def test_reported_confidence_is_the_calibrated_one(tiers):
    tiers["t1"] = ("Liability", 0.95)
    r = _detect()
    assert r["confidence"] == pytest.approx(0.877, abs=1e-3)


def test_low_tier1_confidence_falls_to_tier2(tiers):
    tiers["t1"] = ("Liability", 0.55)
    r = _detect()
    assert r["detector"] == "ml"
    assert r["risk_type"] == "Penalty"


def test_tier1_none_label_falls_through(tiers):
    tiers["t1"] = ("None", 0.99)
    r = _detect()
    assert r["detector"] == "ml"


def test_both_models_unsure_falls_to_keywords(tiers):
    tiers["t1"] = ("Liability", 0.55)
    tiers["t2"] = ("Penalty", 0.30)
    r = _detect()
    assert r["detector"] == "keyword"
    assert r["risk_type"] == "Liability"  # "indemnify" keyword


def test_benign_sentence_produces_no_risk(tiers):
    tiers["t1"] = ("None", 0.99)
    tiers["t2"] = ("None", 0.99)
    assert _detect("This agreement is written in English.") is None


def test_tier1_failure_degrades_to_tier2(tiers, monkeypatch):
    def boom(sentence):
        raise RuntimeError("model failed to load")

    sys.modules["utils.transformer_classifier"].predict_transformer = boom
    r = _detect()
    assert r["detector"] == "ml"


def test_tier2_failure_degrades_to_keywords(tiers, monkeypatch):
    tiers["t1"] = ("Liability", 0.4)

    def boom(sentence):
        raise RuntimeError("joblib missing")

    monkeypatch.setattr(nlp_utils, "predict_label", boom)
    r = _detect()
    assert r["detector"] == "keyword"


@pytest.mark.parametrize(
    "mode, expected",
    [("transformer", "transformer"), ("ml", "ml"), ("keyword", "keyword")],
)
def test_single_tier_modes(tiers, monkeypatch, mode, expected):
    monkeypatch.setenv("RISK_DETECTOR", mode)
    r = _detect()
    assert r["detector"] == expected


def test_single_tier_modes_ignore_thresholds(tiers, monkeypatch):
    monkeypatch.setenv("RISK_DETECTOR", "transformer")
    tiers["t1"] = ("Liability", 0.3)
    assert _detect()["detector"] == "transformer"


def test_keyword_mode_never_calls_models(tiers, monkeypatch):
    monkeypatch.setenv("RISK_DETECTOR", "keyword")
    _detect()
    assert tiers["t1_calls"] == 0 and tiers["t2_calls"] == 0


def test_payload_shape(tiers):
    r = _detect()
    assert set(r) >= {
        "original_sentence",
        "risk_type",
        "severity",
        "severity_score",
        "confidence",
        "detector",
        "explanation",
        "explanation_source",
    }
    assert r["severity"] in {"High", "Medium", "Low"}


def test_keyword_tier_categories():
    assert nlp_utils.detect_risk_types("Disputes go to binding arbitration.") == ["Arbitration"]
    assert nlp_utils.detect_risk_types("A late payment surcharge applies.") == ["Penalty"]
    assert nlp_utils.detect_risk_types("The sky is blue.") == []


def test_segmentation_handles_empty_input():
    assert nlp_utils.segment_sentences("") == []
    assert len(nlp_utils.segment_sentences("One clause. Another clause.")) == 2
