import pytest

from utils.severity_estimator import estimate_severity, severity_score


def test_high_severity_absolute_obligation_with_transformer_confidence():
    sentence = "The contractor shall indemnify and hold harmless all parties."
    assert estimate_severity(sentence, "Liability", 0.91) == "High"
    assert severity_score(sentence, "Liability", 0.91) == pytest.approx(0.777, abs=1e-3)


def test_high_severity_without_transformer_confidence_redistributes_weight():
    sentence = "The contractor shall indemnify and hold harmless all parties."
    assert estimate_severity(sentence, "Liability", None) == "High"
    assert severity_score(sentence, "Liability", None) == pytest.approx(0.76, abs=1e-3)


def test_medium_severity_permissive_modal():
    sentence = "The vendor may terminate this agreement upon 30 days notice."
    assert estimate_severity(sentence, "Termination", 0.70) == "Medium"
    assert severity_score(sentence, "Termination", 0.70) == pytest.approx(0.495, abs=1e-3)


def test_suppressor_phrase_pulls_score_down_but_modal_keeps_it_high():
    sentence = (
        "Company shall not be liable for indirect damages, "
        "and is exempt from consequential losses."
    )
    # "shall not" / "exempt from" are suppressors (-0.10 each), but the "shall" modal
    # (0.90) and Liability category weight (0.75) still dominate the composite.
    assert estimate_severity(sentence, "Liability", 0.80) == "High"
    assert severity_score(sentence, "Liability", 0.80) == pytest.approx(0.685, abs=1e-3)


def test_low_severity_no_modal_no_risk_language():
    sentence = "This is a purely administrative clause with no obligations."
    assert estimate_severity(sentence, "Obligation", None) == "Low"
    assert severity_score(sentence, "Obligation", None) == pytest.approx(0.265, abs=1e-3)


def test_severity_score_is_always_clamped_to_unit_interval():
    extreme_amplified = "This is unlimited, sole, absolute, irrevocable, and forfeited without notice."
    score = severity_score(extreme_amplified, "Liability", 0.99)
    assert 0.0 <= score <= 1.0


def test_unknown_risk_type_falls_back_to_default_category_weight():
    sentence = "Miscellaneous clause."
    # CATEGORY_BASE_WEIGHTS.get(unknown, 0.50) — should not raise, should stay in range.
    score = severity_score(sentence, "NotARealCategory", None)
    assert 0.0 <= score <= 1.0
