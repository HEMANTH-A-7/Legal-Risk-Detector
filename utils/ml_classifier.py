import os
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

import joblib


DEFAULT_MODEL_PATH = os.path.join("models", "risk_classifier.joblib")


@lru_cache(maxsize=1)
def load_model(model_path: Optional[str] = None) -> Dict[str, Any]:
    path = model_path or os.getenv("RISK_MODEL_PATH", DEFAULT_MODEL_PATH)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at: {path}")
    obj = joblib.load(path)
    if not isinstance(obj, dict) or "pipeline" not in obj:
        raise ValueError("Invalid model file format")
    return obj


def predict_label(sentence: str, model_path: Optional[str] = None) -> Tuple[str, float]:
    """
    Run TF-IDF + Logistic Regression on a single sentence.

    Returns (label, confidence) where confidence is the max class probability
    from predict_proba, or 0.0 if the model does not support probability estimates.
    """
    sentence = (sentence or "").strip()
    if not sentence:
        return "None", 0.0

    model = load_model(model_path)
    pipeline = model["pipeline"]

    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba([sentence])[0]
        classes = list(getattr(pipeline, "classes_", []))
        if not classes:
            label = pipeline.predict([sentence])[0]
            return str(label), 0.0
        best_idx = max(range(len(probs)), key=lambda i: probs[i])
        return str(classes[best_idx]), float(probs[best_idx])

    label = pipeline.predict([sentence])[0]
    return str(label), 0.0
