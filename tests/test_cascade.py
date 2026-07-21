import os

import pytest


@pytest.fixture(autouse=True)
def _keyword_only_mode(monkeypatch):
    # Tier 3 (keyword) is deterministic and requires no model files, so the
    # cascade-routing logic itself can be tested without loading Legal-BERT
    # or the TF-IDF pipeline. Transformer/ML routing is covered separately
    # in tests/test_transformer_integration.py.
    monkeypatch.setenv("RISK_DETECTOR", "keyword")


def test_segment_sentences_splits_on_sentence_boundaries():
    from utils.nlp_utils import segment_sentences

    text = (
        "The contractor shall indemnify all parties. "
        "Late payments are subject to a 1.5% monthly surcharge. "
        "This is a purely administrative clause."
    )
    sentences = segment_sentences(text)
    assert len(sentences) == 3
    assert sentences[0].startswith("The contractor shall indemnify")


def test_segment_sentences_handles_empty_input():
    from utils.nlp_utils import segment_sentences

    assert segment_sentences("") == []
    assert segment_sentences(None) == []


def test_detect_risk_types_matches_expected_categories():
    from utils.nlp_utils import detect_risk_types

    assert detect_risk_types("The contractor shall indemnify all parties.") == [
        "Liability",
        "Obligation",
    ]
    assert detect_risk_types("no risky words appear in this sentence at all") == []


def test_keyword_cascade_finds_expected_risks_and_skips_benign_sentences():
    from utils.nlp_utils import analyze_risks, segment_sentences

    text = (
        "The contractor shall indemnify all parties. "
        "Late payments are subject to a 1.5% monthly surcharge. "
        "This is a purely administrative clause."
    )
    results = analyze_risks(segment_sentences(text))

    # The benign third sentence must NOT produce a risk entry.
    assert len(results) == 2

    liability, penalty = results
    assert liability["risk_type"] == "Liability"
    assert liability["severity"] == "High"
    assert liability["detector"] == "keyword"
    assert "confidence" not in liability  # Tier 3 never sets a confidence score

    assert penalty["risk_type"] == "Penalty"
    assert penalty["severity"] == "Medium"
    assert penalty["detector"] == "keyword"


def test_keyword_cascade_produces_no_results_for_purely_benign_text():
    from utils.nlp_utils import analyze_risks, segment_sentences

    text = "The meeting is scheduled for next Tuesday afternoon."
    results = analyze_risks(segment_sentences(text))
    assert results == []


def test_assess_severity_legacy_wrapper_delegates_to_hse():
    from utils.nlp_utils import assess_severity
    from utils.severity_estimator import estimate_severity

    sentence = "The contractor shall indemnify and hold harmless all parties."
    assert assess_severity(sentence, "Liability") == estimate_severity(
        sentence, "Liability", transformer_confidence=None
    )


def test_generate_explanation_reflects_severity_language():
    from utils.nlp_utils import generate_explanation

    high = generate_explanation("...", "Liability", "High")
    low = generate_explanation("...", "Liability", "Low")
    assert "damages or indemnification" in high
    assert "bounded liability" in low
    assert high != low


def test_generate_explanation_unknown_risk_type_has_generic_fallback():
    from utils.nlp_utils import generate_explanation

    explanation = generate_explanation("...", "NotARealCategory", "Low")
    assert "may affect your rights" in explanation
