# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Create a non-root user (required by HuggingFace Spaces)
RUN useradd -m -u 1000 user

# Set working directory
WORKDIR /app
RUN chown user:user /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY --chown=user:user requirements.txt requirements-dev.txt ./

# Install Python dependencies (both runtime + dev for training scripts on HF Spaces)
USER user
ENV PATH="/home/user/.local/bin:$PATH"

RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt gunicorn

# Download NLTK data (punkt + punkt_tab for sentence segmentation)
COPY --chown=user:user download_nltk.py .
RUN python download_nltk.py

# Copy the rest of the application
COPY --chown=user:user . .

# ── CCIC Environment Defaults ──────────────────────────────────────────────────
# All thresholds configurable at runtime via HF Spaces "Variables and secrets"
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV RISK_DETECTOR=auto
ENV CCIC_TEMPERATURE=1.5
ENV TRANSFORMER_THRESHOLD=0.6
ENV ML_RISK_THRESHOLD=0.5
ENV TRANSFORMER_MODEL_DIR=models/transformer_risk_classifier

# Railway injects $PORT at runtime; fallback to 8000 for local runs
EXPOSE ${PORT:-8000}

# Run with gunicorn — single worker. Each worker loads its own copy of
# torch/transformers/Legal-BERT (~830MB RSS measured locally after one request);
# 2 workers need ~1.6GB+, well past Render free tier's 512MB RAM cap, causing
# OOM kills mid-request. -w 1 roughly halves peak memory but may still exceed
# 512MB — a paid plan with >=1GB RAM (or a smaller/quantized model) is the
# actual fix; this just stops the worst-case 2x blowup. --timeout 120 gives the
# slow first (cold) model load room to finish instead of being killed at
# gunicorn's default 30s. Bind to Render/Railway's dynamic port.
CMD ["sh", "-c", "gunicorn -b 0.0.0.0:${PORT:-8000} -w 1 --timeout 120 app:app"]
