import pytest

from utils.severity_estimator import estimate_severity, severity_score

STRONG_LIABILITY = "The contractor shall indemnify and hold harmless all parties."


def test_strong_liability_clause_is_high():
    assert estimate_severity(STRONG_LIABILITY, "Liability", 0.91) == "High"


def test_label_agrees_with_score_thresholds():
    cases = [
        (STRONG_LIABILITY, "Liability", 0.91),
        ("Disputes may be referred to arbitration.", "Arbitration", None),
        ("Late payments are subject to a surcharge.", "Penalty", None),
    ]
    for sentence, risk, conf in cases:
        score = severity_score(sentence, risk, conf)
        label = estimate_severity(sentence, risk, conf)
        expected = "High" if score >= 0.65 else "Medium" if score >= 0.35 else "Low"
        assert label == expected


def test_modal_strength_orders_scores():
    must = severity_score("The vendor must pay all damages.", "Liability")
    may = severity_score("The vendor may pay all damages.", "Liability")
    assert must > may


def test_amplifiers_raise_and_suppressors_lower():
    base = severity_score("The supplier shall pay the fee.", "Penalty")
    amplified = severity_score("The supplier shall pay the fee immediately and irrevocably.", "Penalty")
    suppressed = severity_score("The supplier is not required to pay the fee.", "Penalty")
    assert amplified > base
    assert suppressed < amplified


def test_modal_match_is_whole_word():
    # "will" must not match inside "Willis"
    assert severity_score("Willis signed the contract.", "Obligation") == severity_score(
        "Smith signed the contract.", "Obligation"
    )


@pytest.mark.parametrize("conf", [None, 0.0, 0.5, 1.0])
def test_score_is_clamped_to_unit_interval(conf):
    s = "The party must immediately, irrevocably and absolutely forfeit unlimited sums."
    assert 0.0 <= severity_score(s, "Liability", conf) <= 1.0
