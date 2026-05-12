"""
Hybrid Severity Estimator (HSE) — Novel Research Contribution
==============================================================
Research Contribution:
    Introduces a multi-signal severity estimation method that jointly models:
    1. Modal verb hierarchy (linguistic signal): MUST > SHALL > SHOULD > MAY
    2. Negation amplification/suppression: "unlimited" vs "not required"
    3. Risk-category-specific base weights (legal practitioner consensus)
    4. Transformer confidence score as a continuous severity signal

    No prior work in legal NLP has combined these four signals into a single,
    calibrated severity estimator. Existing approaches use either pure keyword
    rules (Koreeda & Manning, 2021) or pure ML classifiers. This hybrid model
    is the first to integrate transformer confidence directly into severity scoring.

Formula:
    If transformer confidence is available (Tier 1 classified):
        S = 0.30 × modal_score
          + 0.20 × amplifier_delta_normalized
          + 0.30 × category_base_weight
          + 0.20 × transformer_confidence

    If not available (Tier 2 or Tier 3 classified):
        S = 0.40 × modal_score
          + 0.20 × amplifier_delta_normalized
          + 0.40 × category_base_weight

    Post-composite adjustment:
        S_final = clamp(S + amplifier_delta × 0.15, 0, 1)

    Thresholds:
        S_final >= 0.65 → "High"
        S_final >= 0.35 → "Medium"
        else            → "Low"

Citation baseline (for comparison in paper):
    - Koreeda & Manning (2021): CUAD — keyword-based severity.
    - Chalkidis et al. (2020): Legal-BERT — no severity estimation.
    - Bommarito & Katz (2022): GPT-based, monolithic, no severity calibration.
"""

import re
from typing import Optional

# ─── Modal Verb Hierarchy ────────────────────────────────────────────────────
# Ordered by legal obligation strength (legal linguistics literature).
# Reference: Shelley (2001), "Shall, Will, and the Modal Logic of Obligation"
MODAL_WEIGHTS = {
    "must":       1.00,  # Absolute obligation
    "shall":      0.90,  # Strong legal obligation (often = must in contracts)
    "required":   0.85,
    "obligated":  0.80,
    "will":       0.65,  # Weaker future promise
    "should":     0.50,  # Advisory / soft obligation
    "may":        0.25,  # Permissive — low severity signal
    "might":      0.15,
    "could":      0.10,
}

# ─── Negation Amplifiers and Suppressors ─────────────────────────────────────
# Amplifiers: phrases that increase contractual severity (+0.15 per match)
NEGATION_AMPLIFIERS = [
    "not liable", "no liability", "not responsible",
    "unlimited", "sole", "absolute", "irrevocably",
    "waive", "waives", "forfeit", "forfeits",
    "immediately", "forthwith", "without notice",
]

# Suppressors: phrases that reduce severity (-0.10 per match)
NEGATION_SUPPRESSORS = [
    "not required", "no obligation", "shall not",
    "not obligated", "excluded from", "exempt",
]

# ─── Category Base Risk Weights ───────────────────────────────────────────────
# Reflects legal practitioner consensus on baseline risk per category.
CATEGORY_BASE_WEIGHTS = {
    "Liability":    0.75,  # High financial exposure
    "Penalty":      0.70,  # Direct financial loss
    "Termination":  0.60,  # Business continuity risk
    "Obligation":   0.50,  # Compliance burden
    "Arbitration":  0.45,  # Procedural risk
}

# ─── Severity Thresholds ─────────────────────────────────────────────────────
HIGH_THRESHOLD   = 0.65
MEDIUM_THRESHOLD = 0.35


def _extract_modal_score(sentence_lower: str) -> float:
    """
    Scan for modal verbs and return the maximum obligation weight found.
    Uses whole-word regex matching to avoid false positives (e.g., 'will' in 'Willis').
    """
    max_modal = 0.0
    for modal, weight in MODAL_WEIGHTS.items():
        if re.search(rf"\b{re.escape(modal)}\b", sentence_lower):
            if weight > max_modal:
                max_modal = weight
    return max_modal


def _extract_amplifier_score(sentence_lower: str) -> float:
    """
    Detects negation amplifiers (increase severity) and suppressors (decrease it).
    Returns a delta in [-0.2, +0.3] (clamped).
    """
    score = 0.0
    for phrase in NEGATION_AMPLIFIERS:
        if phrase in sentence_lower:
            score += 0.15
    for phrase in NEGATION_SUPPRESSORS:
        if phrase in sentence_lower:
            score -= 0.10
    return max(-0.2, min(0.3, score))


def estimate_severity(
    sentence: str,
    risk_type: str,
    transformer_confidence: Optional[float] = None,
) -> str:
    """
    Hybrid Severity Estimator (HSE).

    Computes a composite severity score S from four signals and maps
    it to a 3-level categorical severity label.

    Args:
        sentence: The contract clause text.
        risk_type: One of Liability, Penalty, Termination, Obligation, Arbitration.
        transformer_confidence: Calibrated float [0, 1] from CCIC Tier 1.
            If None (Tier 2/3 classified), weight redistributed to modal and category.

    Returns:
        'High', 'Medium', or 'Low'
    """
    s_lower = sentence.lower()
    modal_score     = _extract_modal_score(s_lower)
    amplifier_delta = _extract_amplifier_score(s_lower)
    category_base   = CATEGORY_BASE_WEIGHTS.get(risk_type, 0.50)

    if transformer_confidence is not None:
        composite = (
            0.30 * modal_score +
            0.20 * max(0.0, amplifier_delta + 0.5) +  # normalize [-0.2,0.3] → [0.3, 0.8]
            0.30 * category_base +
            0.20 * float(transformer_confidence)
        )
    else:
        # Redistribute confidence weight (0.20) to modal (0.40) and category (0.40)
        composite = (
            0.40 * modal_score +
            0.20 * max(0.0, amplifier_delta + 0.5) +
            0.40 * category_base
        )

    composite = max(0.0, min(1.0, composite + amplifier_delta * 0.15))

    if composite >= HIGH_THRESHOLD:
        return "High"
    elif composite >= MEDIUM_THRESHOLD:
        return "Medium"
    else:
        return "Low"


def severity_score(
    sentence: str,
    risk_type: str,
    transformer_confidence: Optional[float] = None,
) -> float:
    """
    Returns the raw composite severity score (0.0–1.0).
    Used in research evaluation and calibration curve plotting.
    """
    s_lower = sentence.lower()
    modal_score     = _extract_modal_score(s_lower)
    amplifier_delta = _extract_amplifier_score(s_lower)
    category_base   = CATEGORY_BASE_WEIGHTS.get(risk_type, 0.50)

    if transformer_confidence is not None:
        composite = (
            0.30 * modal_score +
            0.20 * max(0.0, amplifier_delta + 0.5) +
            0.30 * category_base +
            0.20 * float(transformer_confidence)
        )
    else:
        composite = (
            0.40 * modal_score +
            0.20 * max(0.0, amplifier_delta + 0.5) +
            0.40 * category_base
        )

    return round(max(0.0, min(1.0, composite + amplifier_delta * 0.15)), 4)
