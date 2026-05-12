import os
from functools import lru_cache
from typing import Optional, Tuple


DEFAULT_DIR = os.path.join("models", "transformer_risk_classifier")


@lru_cache(maxsize=1)
def _load_pipeline(model_dir: Optional[str] = None):
    model_path = model_dir or os.getenv("TRANSFORMER_MODEL_DIR", DEFAULT_DIR)
    try:
        from transformers import pipeline
    except Exception as e:
        raise ImportError("transformers is not installed") from e

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Transformer model directory not found: {model_path}")

    try:
        device = int(os.getenv("TRANSFORMER_DEVICE", "-1"))
    except ValueError:
        device = -1

    return pipeline(
        task="text-classification",
        model=model_path,
        tokenizer=model_path,
        top_k=None,          # Return scores for all labels
        device=device,       # -1 = CPU; 0 = first GPU
    )


def predict_transformer(sentence: str, model_dir: Optional[str] = None) -> Tuple[str, float]:
    """
    Run the fine-tuned transformer on a single sentence.

    Returns (label, confidence) where:
    - label: highest-scoring class name (or "None" if benign)
    - confidence: raw softmax probability of the top-1 class

    Note: calibration (temperature scaling) is applied OUTSIDE this function
    in nlp_utils.detect_risk_transformer() before threshold comparison.
    """
    s = (sentence or "").strip()
    if not s:
        return "None", 0.0

    clf = _load_pipeline(model_dir)
    out = clf(s)

    # HuggingFace pipeline returns [[{label, score}, ...]] with top_k=None
    if isinstance(out, list) and out and isinstance(out[0], list):
        scores = out[0]
    elif isinstance(out, list):
        scores = out
    else:
        scores = []

    if not scores:
        return "None", 0.0

    best = max(scores, key=lambda x: x.get("score", 0.0))
    label = str(best.get("label", "None"))
    score = float(best.get("score", 0.0))

    # If model outputs raw LABEL_N indices, the id2label mapping was not saved
    if label.startswith("LABEL_"):
        return "None", score

    return label, score
