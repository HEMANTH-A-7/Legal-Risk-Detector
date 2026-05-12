"""
Vercel Serverless Entry Point — LexGuard
=========================================
Vercel has a 250 MB bundle limit, so the 328 MB transformer model cannot
be deployed here. The CCIC automatically falls back to Tier 2 (TF-IDF+LR)
+ Tier 3 (keyword) when the transformer model directory does not exist.

We explicitly set RISK_DETECTOR=ml here as a safety guard to avoid any
attempt to import or load transformers in the serverless environment.
"""

import os

# Safety guard: force Tier 2+3 only on Vercel (no transformer)
# This is set in vercel.json build.env as well, but we enforce it here
# in case the environment variable is missing.
if not os.path.isdir(os.path.join("models", "transformer_risk_classifier")):
    os.environ.setdefault("RISK_DETECTOR", "ml")

from app import app
