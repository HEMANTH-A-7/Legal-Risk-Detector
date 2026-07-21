"""
Temperature Scaling Calibration — Research Contribution Component
=================================================================
Part of the Confidence-Calibrated Inference Cascade (CCIC) architecture.

Motivation (for paper):
    Neural classifiers are known to be over-confident (Guo et al., 2017).
    Temperature scaling is a post-hoc calibration method (single parameter T)
    that rescales softmax probabilities without changing model accuracy.

Mathematical formulation:
    Given raw probability p from softmax:
        logit = log(p / (1 - p))
        scaled_logit = logit / T
        calibrated = sigmoid(scaled_logit)

    For T > 1: calibrated < p  (softens over-confident predictions)
    For T = 1: calibrated = p  (identity, no scaling)

References:
    - Guo et al. (2017) "On Calibration of Modern Neural Networks" ICML.
"""

import math
import os
from typing import List, Tuple

DEFAULT_TEMPERATURE = float(os.getenv("CCIC_TEMPERATURE", "1.5"))


def temperature_scale(logits: List[float], temperature: float = DEFAULT_TEMPERATURE) -> List[float]:
    """
    Apply temperature scaling to raw logits.
    Returns calibrated probabilities via numerically stable softmax.

    Args:
        logits: Raw model logits for each class.
        temperature: Scaling factor T > 0.
    Returns:
        List of calibrated probabilities (sums to 1.0).
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")
    scaled = [l / temperature for l in logits]
    max_val = max(scaled)
    exp_vals = [math.exp(v - max_val) for v in scaled]
    total = sum(exp_vals)
    return [e / total for e in exp_vals]


def calibrate_confidence(raw_confidence: float, temperature: float = DEFAULT_TEMPERATURE) -> float:
    """
    Scalar temperature scaling: operates in logit space on the top-1 probability.

    Used in CCIC Tier 1 before threshold comparison. Corrects over-confidence
    of BERT-based models (Guo et al., 2017).

    Derivation:
        calibrated = sigmoid(logit(p) / T)
        where logit(p) = log(p / (1-p))

    Example: p=0.95, T=1.5 → logit=2.944 → scaled=1.963 → calibrated=0.877
    This ~7.7% reduction prevents over-confident wrong labels from
    bypassing Tier 2.

    Args:
        raw_confidence: Raw softmax probability (0.0–1.0).
        temperature: Calibration temperature (default: CCIC_TEMPERATURE env var).
    Returns:
        Calibrated confidence score (0.0–1.0).
    """
    if not (0.0 < raw_confidence < 1.0):
        return raw_confidence
    logit = math.log(raw_confidence / (1.0 - raw_confidence))
    scaled_logit = logit / temperature
    calibrated = 1.0 / (1.0 + math.exp(-scaled_logit))
    return round(calibrated, 4)


def compute_ece(predictions: List[Tuple[float, bool]], n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE) — for paper's calibration analysis.

    ECE = Σ_b (|B_b| / n) × |acc(B_b) − conf(B_b)|

    Lower ECE = better calibration. Perfect calibration = 0.0.

    Args:
        predictions: List of (confidence, is_correct) tuples.
        n_bins: Number of equal-width bins. Default: 10.
    Returns:
        ECE score (lower is better).
    """
    bins = [[] for _ in range(n_bins)]
    for conf, correct in predictions:
        bin_idx = min(int(conf * n_bins), n_bins - 1)
        bins[bin_idx].append((conf, correct))

    ece = 0.0
    n = len(predictions)
    if n == 0:
        return 0.0

    for b in bins:
        if not b:
            continue
        avg_conf = sum(c for c, _ in b) / len(b)
        avg_acc  = sum(1 for _, ok in b if ok) / len(b)
        ece += (len(b) / n) * abs(avg_conf - avg_acc)

    return round(ece, 4)
