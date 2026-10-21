import os
import sys
import warnings

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# The app resolves model paths relative to the repo root.
os.chdir(ROOT)

# The pickled TF-IDF model may have been saved with a slightly different
# scikit-learn patch version; that warning is not what these tests check.
warnings.filterwarnings("ignore", message="Trying to unpickle estimator")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Every test starts from the documented defaults, with no API key."""
    for var in (
        "RISK_DETECTOR",
        "TRANSFORMER_THRESHOLD",
        "ML_RISK_THRESHOLD",
        "CCIC_TEMPERATURE",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
