---
title: LexGuard Legal Risk Detector
emoji: ⚖️
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# LexGuard — A Confidence-Calibrated Inference Cascade for Legal Contract Risk Detection

> **"LexGuard: A Confidence-Calibrated Inference Cascade with Hybrid Severity Estimation for Legal Contract Risk Detection"**
>
> *Research Prototype — [Hemanth021/legal-risk-detector](https://huggingface.co/spaces/Hemanth021/legal-risk-detector)*

---

## Overview

LexGuard is a full-stack NLP system that identifies and explains risky clauses in legal contracts for non-lawyers. It implements **four novel research contributions** formalized for academic publication:

1. **Confidence-Calibrated Inference Cascade (CCIC)** — formally defined 3-tier fallback with temperature-scaled confidence thresholds
2. **Hybrid Severity Estimator (HSE)** — multi-signal severity model combining linguistic, semantic, and ML signals
3. **Legal-BERT Model Upgrade** — ablation study comparing domain-adapted vs. general pre-training
4. **Cross-Dataset Corpus Construction** — CUAD (41 clause types) + LEDGAR (60+ categories) mapped to a 5-class taxonomy

---

## Architecture — Confidence-Calibrated Inference Cascade (CCIC)

```
Input Text / PDF  →  NLTK Sentence Segmentation
                              │
                              ▼
┌──────────────────────────────────────────────────────────┐
│  Tier 1: Legal-BERT (nlpaueb/legal-bert-base-uncased)    │
│    ↓ Raw softmax probability p                           │
│    ↓ Temperature Scaling: calibrated = sigmoid(logit(p)/T)│
│      where T = CCIC_TEMPERATURE (default 1.5)            │
│    ↓ calibrated ≥ τ_t (0.60)?                            │
│      → YES: use label + pass confidence to HSE           │
│      → NO:  fall to Tier 2 (label="None" also falls)     │
├──────────────────────────────────────────────────────────┤
│  Tier 2: TF-IDF + Logistic Regression                    │
│    ↓ confidence ≥ τ_m (0.50)?                            │
│      → YES: use label (confidence NOT passed to HSE)     │
│      → NO:  fall to Tier 3                               │
├──────────────────────────────────────────────────────────┤
│  Tier 3: Keyword Matching (deterministic)                │
│    → Always produces a label or skips the sentence       │
└──────────────────────────────────────────────────────────┘
                              │
                              ▼
        Hybrid Severity Estimator (HSE)
        S = 0.30×modal + 0.20×amplifier + 0.30×category + 0.20×confidence
                              │
                              ▼
             JSON Response → Frontend Dashboard
```

### Risk Categories (5-class taxonomy)

| Class | Examples |
|---|---|
| **Liability** | Indemnification, damages, hold harmless, cap on liability |
| **Penalty** | Liquidated damages, late fees, surcharges, revenue restrictions |
| **Termination** | Cancellation, breach, change of control, agreement term |
| **Obligation** | Shall/must clauses, non-compete, audit rights, insurance |
| **Arbitration** | Governing law, dispute resolution, jurisdiction, venue |

---

## Novel Contributions

### 1. Confidence-Calibrated Inference Cascade (CCIC)

The CCIC formalizes a 3-tier fallback pipeline where each tier only handles a sentence if the previous tier's confidence is below a configurable threshold. The key novelty is **post-hoc temperature scaling** (Guo et al., 2017) applied to transformer softmax outputs *before* threshold comparison:

```
calibrated = sigmoid(logit(p) / T)
where logit(p) = log(p / (1 - p))
```

For T=1.5 and a typical over-confident prediction p=0.95:
- Raw: 0.95
- Calibrated: ~0.877 (-7.7%)

This prevents over-confident but incorrect Tier 1 predictions from suppressing Tier 2, which is the critical safety property for legal risk detection.

**No prior work** has formally defined or empirically evaluated this 3-tier cascade with calibrated thresholds for legal clause risk classification.

### 2. Hybrid Severity Estimator (HSE)

Four signals combined into a composite score:

| Signal | Description | Weight (w/ transformer) | Weight (w/o transformer) |
|---|---|---|---|
| Modal verb strength | `must`→1.00, `shall`→0.90, `may`→0.25 | 30% | 40% |
| Negation amplifiers | `unlimited`→+0.15, `not required`→−0.10 | 20% | 20% |
| Category base weight | Liability→0.75, Arbitration→0.45 | 30% | 40% |
| Transformer confidence | Calibrated Tier 1 score | 20% | — |

Thresholds: composite ≥ 0.65 → **High** · ≥ 0.35 → **Medium** · else → **Low**

### 3. Legal-BERT Ablation Study

| Model | Pre-training | Macro-F1 | Weighted-F1 | Eval Loss |
|---|---|---|---|---|
| distilroberta-base (BASELINE) | General | 94.57% | 96.17% | 0.1502 |
| nlpaueb/legal-bert-base-uncased (PROPOSED) | Legal corpus | *to be trained* | *to be trained* | *to be trained* |

### 4. Cross-Dataset Corpus Construction

| Dataset | Source | Clause Types | LexGuard Sentences |
|---|---|---|---|
| Original | Manually annotated | 5 | 4,919 |
| CUAD | Hendrycks et al. (2021) | 41 → 5 | ~10,000 |
| LEDGAR | Tuggener et al. (2020) | 60+ → 5 | ~5,000 |
| **Merged** | All three | 5 | **~17,000** |

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env — set OPENAI_API_KEY if you want GPT-4o-mini explanations
```

### 3. Download NLTK data
```bash
python download_nltk.py
```

### 4. Run the app
```bash
python app.py
# Open http://localhost:8000
```

---

## Training

### Train ML model (CCIC Tier 2 — always needed)
```bash
python train_model.py --mode ml --data data/training_merged.csv
# Saves: models/risk_classifier.joblib
```

### Fine-tune Legal-BERT (CCIC Tier 1 — PROPOSED model)
```bash
# Requires GPU with 6GB+ VRAM (or Google Colab T4)
pip install -r requirements-dev.txt
python train_model.py --mode transformer --model legal-bert \
    --data data/training_merged.csv --epochs 3 --batch_size 16
# Saves: models/transformer_risk_classifier/ + training_meta.json (ablation_role: proposed)
```

### Fine-tune distilroberta (CCIC Tier 1 — BASELINE for ablation)
```bash
python train_model.py --mode transformer --model distilroberta \
    --data data/training_merged.csv --epochs 3 --batch_size 16
# Saves: training_meta.json (ablation_role: baseline)
```

### Build merged training corpus (CUAD + LEDGAR)
```bash
pip install datasets pandas
python scripts/map_cuad.py --dataset both
python scripts/map_cuad.py --merge --output data/training_merged.csv
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `RISK_DETECTOR` | `auto` | CCIC mode: `auto` \| `transformer` \| `ml` \| `keyword` |
| `TRANSFORMER_THRESHOLD` | `0.60` | Tier 1 confidence threshold τ_t |
| `ML_RISK_THRESHOLD` | `0.50` | Tier 2 confidence threshold τ_m |
| `CCIC_TEMPERATURE` | `1.5` | Temperature scaling factor T (Guo et al., 2017) |
| `TRANSFORMER_MODEL_DIR` | `models/transformer_risk_classifier` | Path to fine-tuned model |
| `OPENAI_API_KEY` | *(optional)* | Enables GPT-4o-mini for /explain route |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model name |
| `MAX_UPLOAD_BYTES` | `10485760` | Max upload size (10 MB) |

---

## API Reference

### `POST /analyze`

Accepts `multipart/form-data` with `file` (PDF/TXT) or `text` (string).

**Response:**
```json
{
  "risks": [
    {
      "original_sentence": "The party shall indemnify and hold harmless...",
      "risk_type": "Liability",
      "severity": "High",
      "severity_score": 0.7821,
      "confidence": 0.9124,
      "detector": "transformer",
      "explanation": "...",
      "explanation_source": "template"
    }
  ],
  "ccic": {
    "mode": "auto",
    "temperature": 1.5,
    "transformer_threshold": 0.6,
    "ml_threshold": 0.5,
    "detector_distribution": { "transformer": 30, "ml": 5, "keyword": 3 }
  },
  "summary": {
    "total_sentences": 53,
    "risky_count": 38,
    "risk_percentage": 71.7,
    "severity_distribution": { "High": 12, "Medium": 20, "Low": 6 }
  }
}
```

### `POST /explain`

Accepts JSON: `{ "sentence": "...", "risk_type": "Liability", "severity": "High" }`

Returns GPT-4o-mini explanation if `OPENAI_API_KEY` is set, otherwise template-based fallback.

---

## Deployment

### Hugging Face Spaces (Full CCIC — all 3 tiers)
🔗 [huggingface.co/spaces/Hemanth021/legal-risk-detector](https://huggingface.co/spaces/Hemanth021/legal-risk-detector)

Deployed via Docker. All three CCIC tiers active (Legal-BERT + ML + Keyword).

### Vercel (Tier 2 + Tier 3 only)
🔗 [nlp-project-alpha.vercel.app](https://nlp-project-alpha.vercel.app)

Vercel's 250 MB serverless limit excludes the 328 MB transformer. `RISK_DETECTOR=ml` is set automatically. Uses `requirements-vercel.txt` (no torch/transformers).

---

## References

1. Guo et al. (2017). *On Calibration of Modern Neural Networks*. ICML.
2. Chalkidis et al. (2020). *LEGAL-BERT: The Muppets straight out of Law School*. EMNLP Findings.
3. Hendrycks et al. (2021). *CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review*. NeurIPS.
4. Tuggener et al. (2020). *LEDGAR: A Large-Scale Multi-label Corpus for Text Classification of Legal Provisions*. LREC.
5. Bommarito & Katz (2022). *GPT Takes the Bar Exam*. arXiv:2212.14402.
6. Shelley (2001). *Shall, Will, and the Modal Logic of Obligation*. Legal Writing Institute.

---

## Citation

If you use this work, please cite:
```bibtex
@inproceedings{lexguard2025,
  title     = {LexGuard: A Confidence-Calibrated Inference Cascade with Hybrid
               Severity Estimation for Legal Contract Risk Detection},
  author    = {[Your Name]},
  booktitle = {Proceedings of JURIX / COLING / AICL Workshop},
  year      = {2025}
}
```
