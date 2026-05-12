"""
NLP Utilities — Core Pipeline for LexGuard
==========================================
Implements the Confidence-Calibrated Inference Cascade (CCIC):

    CCIC Architecture (Novel Contribution):
    ┌─────────────────────────────────────────────────────────┐
    │  Tier 1: Legal-BERT Transformer (domain-adapted)        │
    │    → If calibrated_confidence ≥ τ_t (default 0.60):    │
    │        use result, pass confidence to HSE               │
    │    → Else: fall to Tier 2                               │
    ├─────────────────────────────────────────────────────────┤
    │  Tier 2: TF-IDF + Logistic Regression (ML)              │
    │    → If confidence ≥ τ_m (default 0.50): use result     │
    │    → Else: fall to Tier 3                               │
    ├─────────────────────────────────────────────────────────┤
    │  Tier 3: Modal-Keyword Matching (deterministic)         │
    │    → Always produces a definitive label or None         │
    └─────────────────────────────────────────────────────────┘

    Confidence scores from Tier 1 are post-hoc calibrated via
    Temperature Scaling (Guo et al., 2017) before threshold comparison.
    Calibration formula:
        calibrated = sigmoid(logit(p) / T)
    where T = CCIC_TEMPERATURE (default 1.5).

    Severity is estimated using the Hybrid Severity Estimator (HSE),
    which jointly models:
        S = 0.30 × modal_score
          + 0.20 × amplifier_delta_normalized
          + 0.30 × category_base_weight
          + 0.20 × transformer_confidence  (if available)
    Reference: see utils/severity_estimator.py

Research References (for paper):
    [1] Guo et al. (2017) — Temperature scaling calibration. ICML.
    [2] Chalkidis et al. (2020) — Legal-BERT. EMNLP Findings.
    [3] Hendrycks et al. (2021) — CUAD dataset. NeurIPS.
    [4] Bommarito & Katz (2022) — GPT-based legal NLP. arXiv.
"""

import os
import re
import pdfplumber
import nltk
from nltk.tokenize import sent_tokenize

from utils.ml_classifier import predict_label
from utils.severity_estimator import estimate_severity, severity_score
from utils.calibration import calibrate_confidence

# Set local nltk_data path — must be registered before any sent_tokenize call
nltk_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'nltk_data')
if nltk_data_dir not in nltk.data.path:
    nltk.data.path.append(nltk_data_dir)

# ─── Risk Keywords Dictionary ─────────────────────────────────────────────────
# Used in Tier 3 (deterministic keyword fallback) of the CCIC.
# Expanded to cover modal verbs that directly signal obligation strength.
RISK_CATEGORIES = {
    'Liability':    ['liability', 'responsible', 'damages', 'indemnify',
                     'indemnification', 'hold harmless', 'indemnified'],
    'Penalty':      ['penalty', 'fine', 'liquidated damages', 'interest',
                     'charge', 'forfeit', 'late payment', 'surcharge'],
    'Termination':  ['terminate', 'termination', 'cancel', 'cancellation',
                     'rescind', 'breach', 'expiration', 'expires'],
    'Obligation':   ['must', 'shall', 'obligated', 'requirement',
                     'mandatory', 'duty', 'required to', 'bound to'],
    'Arbitration':  ['arbitration', 'dispute resolution', 'governing law',
                     'jurisdiction', 'venue', 'mediation', 'arbitrator'],
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def extract_text_from_pdf(file_path):
    """Extracts text from a PDF file using pdfplumber."""
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"Error extracting PDF: {e}")
    return text


def segment_sentences(text):
    """Splits text into individual sentences using NLTK punkt tokenizer."""
    if not text:
        return []
    return sent_tokenize(text)


def detect_risk_types(sentence: str):
    """
    Tier 3 of CCIC: deterministic keyword matching.
    Returns list of matched risk categories (first keyword match per category).
    This is the final fallback — no confidence score, always deterministic.
    """
    sentence_lower = sentence.lower()
    detected = []
    for category, keywords in RISK_CATEGORIES.items():
        for keyword in keywords:
            if keyword in sentence_lower:
                detected.append(category)
                break
    return detected


def _detector_mode() -> str:
    return (os.getenv("RISK_DETECTOR", "auto") or "auto").strip().lower()


def _transformer_threshold() -> float:
    try:
        return float(os.getenv("TRANSFORMER_THRESHOLD", "0.6"))
    except ValueError:
        return 0.6


def _ml_threshold() -> float:
    try:
        return float(os.getenv("ML_RISK_THRESHOLD", "0.5"))
    except ValueError:
        return 0.5


# ─── CCIC Tier Functions ──────────────────────────────────────────────────────

def detect_risk_ml(sentence: str, enforce_threshold: bool):
    """
    CCIC Tier 2: TF-IDF + Logistic Regression classifier.

    Returns (label, confidence) or (None, confidence) when:
    - The label is "None" (benign sentence)
    - enforce_threshold=True and confidence < τ_m

    Args:
        sentence: Input clause text.
        enforce_threshold: If True, apply τ_m threshold (auto mode).
                           If False, always return the ML prediction (ml-only mode).
    """
    try:
        label, confidence = predict_label(sentence)
    except Exception:
        return None, 0.0

    if label == "None":
        return None, confidence
    if enforce_threshold and confidence and confidence < _ml_threshold():
        return None, confidence
    return label, confidence


def detect_risk_transformer(sentence: str, enforce_threshold: bool):
    """
    CCIC Tier 1: Legal-BERT fine-tuned transformer (domain-adapted).

    Key novel step: applies temperature-scaled calibration (Guo et al., 2017)
    to the raw softmax probability BEFORE comparing against threshold τ_t.
    This corrects the known over-confidence of BERT-based models.

    Calibration: calibrated = sigmoid(logit(p) / T)
    where T = CCIC_TEMPERATURE (env var, default 1.5).

    Returns (label, calibrated_confidence) or (None, calibrated_confidence).
    The calibrated confidence — not raw — is passed to the HSE for severity scoring.

    Args:
        sentence: Input clause text.
        enforce_threshold: If True, apply τ_t threshold (auto mode).
    """
    try:
        from utils.transformer_classifier import predict_transformer
    except Exception:
        return None, 0.0

    try:
        label, confidence = predict_transformer(sentence)
    except Exception:
        return None, 0.0

    # "None" label = transformer correctly identified a non-risky sentence → fall to Tier 2
    if label == "None":
        return None, confidence

    # ── Novel Step: Apply temperature-scaled calibration (CCIC contribution) ──
    calibrated_conf = calibrate_confidence(confidence)

    if enforce_threshold and calibrated_conf < _transformer_threshold():
        return None, calibrated_conf
    return label, calibrated_conf


# ─── Main Analysis Pipeline ───────────────────────────────────────────────────

def analyze_risks(sentences):
    """
    Runs the full CCIC pipeline on each sentence.

    Returns a list of risk payloads, each containing:
        original_sentence   : str
        risk_type           : str (Liability | Penalty | Termination | Obligation | Arbitration)
        severity            : str (High | Medium | Low)  [via HSE]
        severity_score      : float  [raw HSE composite score — for calibration curve plots]
        confidence          : float  [calibrated transformer score, if available]
        detector            : str  (transformer | ml | keyword)
        explanation         : str
        explanation_source  : str
    """
    results = []
    mode = _detector_mode()

    for sentence in sentences:
        risk_type     = None
        confidence    = None
        detector_used = "keyword"

        # ── Tier 1: Transformer ───────────────────────────────────────────────
        if mode in {"transformer", "auto"}:
            predicted, conf = detect_risk_transformer(
                sentence, enforce_threshold=(mode == "auto")
            )
            if predicted:
                risk_type     = predicted
                confidence    = conf
                detector_used = "transformer"

        # ── Tier 2: ML Classifier ─────────────────────────────────────────────
        if not risk_type and mode in {"ml", "auto"}:
            predicted, conf = detect_risk_ml(
                sentence, enforce_threshold=(mode == "auto")
            )
            confidence = conf
            risk_type  = predicted
            if risk_type:
                detector_used = "ml"

        # ── Tier 3: Keyword Fallback ──────────────────────────────────────────
        if not risk_type:
            detected_risks = detect_risk_types(sentence)
            if detected_risks:
                risk_type = detected_risks[0]

        if risk_type:
            # ── HSE: Hybrid Severity Estimation (Novel Contribution) ──────────
            # Passes calibrated transformer confidence (or None) to weight the score
            severity = estimate_severity(sentence, risk_type, confidence)
            raw_sev  = severity_score(sentence, risk_type, confidence)
            explanation = generate_explanation(sentence, risk_type, severity)

            payload = {
                'original_sentence':  sentence,
                'risk_type':          risk_type,
                'severity':           severity,
                'severity_score':     raw_sev,   # Exposes raw score for evaluation plots
                'explanation':        explanation,
                'explanation_source': 'template',
            }
            if confidence is not None:
                payload['confidence'] = round(float(confidence), 4)
            payload['detector'] = detector_used

            results.append(payload)

    return results


# ─── Legacy (kept for API backward-compatibility) ─────────────────────────────

def assess_severity(sentence, risk_type):
    """
    Backward-compatible wrapper. Delegates to the HSE.
    Use estimate_severity() from severity_estimator.py for research work.
    """
    return estimate_severity(sentence, risk_type, transformer_confidence=None)


def generate_explanation(sentence, risk_type, severity):
    """
    Template-based explanation generator (Tier 3 explanation fallback).
    The /explain endpoint calls the LLM-based explainer (llm_explainer.py)
    when OPENAI_API_KEY is set. This function is the offline fallback.
    """
    templates = {
        'Liability':
            f"This clause defines financial responsibility. Severity is '{severity}' because "
            f"it {'broadly exposes you to damages or indemnification obligations' if severity == 'High' else 'contains bounded liability language'}.",
        'Penalty':
            f"This clause imposes financial penalties for non-compliance. '{severity}' severity "
            f"{'suggests unlimited or disproportionate charges' if severity == 'High' else 'indicates capped or conditional charges'}.",
        'Termination':
            f"This clause governs contract termination. '{severity}' severity indicates "
            f"{'immediate or unilateral termination rights exist' if severity == 'High' else 'standard notice-based termination terms'}.",
        'Obligation':
            f"This clause creates a mandatory obligation (note modal verb strength). "
            f"'{severity}' severity means this is {'a non-negotiable, absolute requirement' if severity == 'High' else 'a conditional or soft obligation'}.",
        'Arbitration':
            f"This clause mandates dispute resolution outside of regular courts. '{severity}' severity "
            f"{'indicates binding arbitration with limited appeal rights' if severity == 'High' else 'suggests standard arbitration procedures apply'}.",
    }
    return templates.get(
        risk_type,
        "This sentence contains legal terms that may affect your rights or responsibilities."
    )
