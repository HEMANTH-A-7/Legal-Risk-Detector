<div align="center">

# LexGuard - Legal Contract Risk Detector

### Automated legal risk detection and severity estimation using a 3-tier Confidence-Calibrated Inference Cascade

<br/>

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-000000?style=for-the-badge&logo=render&logoColor=46E3B7)](https://legal-risk-detector.onrender.com/)
[![GitHub](https://img.shields.io/badge/GitHub-HEMANTH--A--7%2FLegal--Risk--Detector-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/HEMANTH-A-7/Legal-Risk-Detector)
[![License: MIT](https://img.shields.io/badge/License-MIT-6C3483?style=for-the-badge)](LICENSE)

<br/>

**Tech Stack**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white)
![React](https://img.shields.io/badge/React-20232A?style=flat-square&logo=react&logoColor=61DAFB)
![TypeScript](https://img.shields.io/badge/TypeScript-007ACC?style=flat-square&logo=typescript&logoColor=white)
![Vite](https://img.shields.io/badge/Vite-646CFF?style=flat-square&logo=vite&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=flat-square&logo=tailwind-css&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace_Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)
![Render](https://img.shields.io/badge/Render-0B0D0E?style=flat-square&logo=render&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI_API-412991?style=flat-square&logo=openai&logoColor=white)

<br/>

> ### [Try the Live App — lexguard-app-production.up.render.app](https://lexguard-app-production.up.render.app)
> *Paste a contract clause or upload a PDF — get instant risk analysis in seconds*

<br/>

| Architecture | Performance | Training Data | Deployment |
|:---:|:---:|:---:|:---:|
| 3-Tier CCIC + HSE | 92.24% Macro-F1 | 23,525 sentences | Docker on Render |

</div>

---

## Screenshots

| Section | Link |
|---|---|
| Hero — Landing Page | [screenshots/hero.png](screenshots/hero.png) |
| Contract Analyzer — Upload Console | [screenshots/analyzer.png](screenshots/analyzer.png) |
| Analysis Results — Summary Dashboard | [screenshots/results_summary.png](screenshots/results_summary.png) |
| Risk Clause Detection — Anomaly List | [screenshots/risk_clauses.png](screenshots/risk_clauses.png) |
| Contact — Footer | [screenshots/contact.png](screenshots/contact.png) |

---

# LexGuard — Automated Legal Risk Detection and Severity Estimation in Contracts Using NLP-Driven Confidence-Calibrated Inference Cascades

> **"LexGuard: Automated Legal Risk Detection and Severity Estimation in Contracts Using NLP-Driven Confidence-Calibrated Inference Cascades"**
>
> Hemanth Kumar Amarthi, Dr. Purushottama Rao K, Dhanesh Sai M — Dept. of Computer Science and Engineering, SRM University-AP
>
> *Research Prototype — deployed at [lexguard-app-production.up.render.app](https://lexguard-app-production.up.render.app)*

---

## Overview

LexGuard is a full-stack NLP system that identifies, classifies, and scores the severity of risky clauses in legal contracts. Existing tools are either brittle rule-based systems or monolithic transformers that give an uncalibrated, binary risk/no-risk answer. LexGuard addresses this with three contributions, evaluated on the largest reported merged legal-risk corpus to date:

1. **Confidence-Calibrated Inference Cascade (CCIC)** — a 3-tier fallback pipeline (Legal-BERT → TF-IDF + Logistic Regression → deterministic keyword matcher) gated by post-hoc temperature-scaled confidence, correcting for the systematic over-confidence of transformer softmax outputs.
2. **Hybrid Severity Estimator (HSE)** — a four-signal composite model that turns a raw classification into a High / Medium / Low severity label using modal-verb strength, negation amplifiers/suppressors, category base risk, and calibrated transformer confidence.
3. **Merged CUAD + LEDGAR training corpus** — 23,525 labeled sentences across six classes, the largest reported merged domain corpus for this task, evaluated with strict ablation against Legal-BERT and distilroberta-base baselines.

The fine-tuned Legal-BERT model achieves **92.24% Macro-F1 / 92.08% Weighted-F1 / 92.13% Accuracy** on six-class contract risk classification, and calibrated cascade routing cuts transformer inference by roughly **83%** for a 0.5-point Macro-F1 cost.

---

## Architecture — Confidence-Calibrated Inference Cascade (CCIC)

```
Input Text / PDF  ->  NLTK Punkt Sentence Segmentation
                              |
                              v
+------------------------------------------------------------+
|  Tier 1: Legal-BERT (nlpaueb/legal-bert-base-uncased)      |
|    -> Raw softmax probability p over 6 classes             |
|    -> Post-hoc Temperature Scaling: calibrated = softmax(  |
|       logits / T),  T = 1.144 (fitted on validation set    |
|       by minimizing NLL)                                   |
|    -> calibrated confidence >= tau_t (0.60)?                |
|       YES: accept Tier 1 label, pass confidence to HSE      |
|       NO:  fall to Tier 2                                   |
+------------------------------------------------------------+
|  Tier 2: TF-IDF (bigrams, 20k features) + Logistic          |
|          Regression (C=1.0, balanced class weights)         |
|    -> confidence >= tau_m (0.50)?                            |
|       YES: accept Tier 2 label                              |
|       NO:  fall to Tier 3                                   |
+------------------------------------------------------------+
|  Tier 3: Modal-Keyword Matching (deterministic)             |
|    -> No threshold; always returns a definite label         |
|       (e.g. indemnify -> Liability, arbitration -> Arbitration) |
+------------------------------------------------------------+
                              |
                              v
        Hybrid Severity Estimator (HSE)
        S = 0.30*modal + 0.20*negation_delta + 0.30*category + 0.20*confidence
                              |
                              v
     JSON Response (risk type + severity + confidence + routing tier)
                              |
                              v
        Optional LLM Explanation Layer -> plain-English rationale
```

### Risk Categories (6-class taxonomy)

Built from a merged CUAD (510 lawyer-annotated commercial contracts, 41 clause types) + LEDGAR (SEC-filing provisions) corpus, collapsed to the six most consequential risk categories:

| Class | Examples | Samples | % of Corpus |
|---|---|---:|---:|
| **Liability** | Indemnification, damages, hold harmless, cap on liability | 4,000 | 17.0% |
| **Obligation** | Shall/must clauses, non-compete, audit rights, insurance | 4,000 | 17.0% |
| **Termination** | Cancellation, breach, change of control, agreement term | 4,000 | 17.0% |
| **Penalty** | Liquidated damages, late fees, surcharges | 3,525 | 15.0% |
| **Arbitration** | Governing law, dispute resolution, jurisdiction, venue | 4,000 | 17.0% |
| **None** | No identifiable risk language | 4,000 | 17.0% |
| **Total** | | **23,525** | **100%** |

> **Provenance:** the 23,525-sentence pre-dedup total and 16,404 / 3,495 / 3,468 train/val/test
> split (23,367 rows post-dedup, 158 exact duplicates dropped) are backed by
> [`docs/results/table1_corpus_split.csv`](docs/results/table1_corpus_split.csv). The **per-class**
> breakdown above is not yet independently backed by a checked-in artifact — the merged corpus
> itself isn't committed to this repo (see [`docs/results/README.md`](docs/results/README.md#known-gap)
> for how to regenerate and verify it).

---

## Novel Contributions

> All numbers in this section are backed by committed artifacts under
> [`docs/results/`](docs/results/) (see [`docs/results/README.md`](docs/results/README.md) for the
> full provenance table), derived from the Kaggle notebook
> [`lexguard-final`](https://www.kaggle.com/code/hemanthkumaramarthi/lexguard-final) (seed 42). The
> notebook itself is currently an unversioned draft — treat the committed CSVs/JSON in
> `docs/results/` as the source of truth, not the live Kaggle draft, until a pinned notebook
> version exists.

### 1. Confidence-Calibrated Inference Cascade (CCIC)

Post-hoc temperature scaling (Guo et al., 2017) is applied to Legal-BERT's softmax logits before the threshold comparison, using a temperature `T = 1.144` fitted on the validation set by minimizing negative log-likelihood. The **calibrated** confidence — not the raw softmax — drives both cascade routing and HSE severity scoring, preventing over-confident but incorrect Tier 1 predictions from suppressing fallback tiers. This is the critical safety property for a system whose outputs can directly affect contractual liability decisions.

No prior work formally defines or empirically evaluates a 3-tier legal-clause cascade gated on calibrated confidence.

### 2. Hybrid Severity Estimator (HSE)

Four signals combined into a composite score `S ∈ [0, 1]`:

| Signal | Description | Weight |
|---|---|---:|
| Modal verb obligation strength | Linguistic strength of the governing modal (Table below) | 0.30 |
| Negation amplifier/suppressor delta | `+0.15` per amplifier (e.g. "unlimited", "irrevocably"); `−0.10` per suppressor (e.g. "not required", "exempt"); clamped to [−0.20, +0.30] | 0.20 |
| Category base weight | Prior risk weight of the detected class (e.g. Liability > Arbitration) | 0.30 |
| Calibrated transformer confidence | Applies when Tier 1 produced the classification | 0.20 |

**Modal Verb Obligation Hierarchy:**

| Modal Verb | Weight | Modal Verb | Weight |
|---|---:|---|---:|
| must | 1.00 | will | 0.65 |
| shall | 0.90 | should | 0.50 |
| required | 0.85 | may | 0.25 |
| obligated | 0.80 | might | 0.15 |
| | | could | 0.10 |

Severity thresholds: `S ≥ 0.65` → **High** · `0.35 ≤ S < 0.65` → **Medium** · `S < 0.35` → **Low**

HSE is, per the paper's literature review, the first multi-signal severity estimator to explicitly combine linguistic, domain-expert, and neural confidence signals for contract-level legal NLP.

### 3. Ablation Study: Model Performance Comparison

| Model | Macro-F1 | Weighted-F1 | Accuracy | Eval Loss |
|---|---:|---:|---:|---:|
| Keyword-Only (Tier 3) | 30.20% | 30.19% | 30.45% | — |
| TF-IDF + LogReg (Tier 2) | 89.37% | 89.13% | 89.16% | — |
| distilroberta-base (baseline) | 90.83% | 90.62% | 90.69% | 0.6585 |
| **Legal-BERT — LexGuard (proposed)** | **92.24%** | **92.08%** | **92.13%** | **0.5518** |

Legal-BERT's domain-specific pre-training outperforms distilroberta-base by +1.40 Macro-F1 points and reduces evaluation loss by 16.2%, confirming the value of legal-corpus pre-training over general-purpose pre-training for this task.

> **Source:** [`docs/results/table2_ablation.csv`](docs/results/table2_ablation.csv), derived from
> [`docs/results/eval_summary.json`](docs/results/eval_summary.json) (Kaggle notebook
> [`lexguard-final`](https://www.kaggle.com/code/hemanthkumaramarthi/lexguard-final), seed 42).

**Per-class F1 (Legal-BERT, proposed model):**

| Class | F1 | Notes |
|---|---:|---|
| Arbitration | 0.982 | Strongest; distinctive lexical markers |
| Penalty | 0.948 | High; financial language is distinctive |
| Liability | 0.942 | High; occasional overlap with Obligation |
| Obligation | 0.895 | Moderate; overlaps with None and Liability |
| Termination | 0.898 | Moderate; lowest recall, leaks to None/Obligation |
| None | 0.870 | Lowest; confused with Obligation and Termination |

> **Source:** [`docs/results/table3_perclass.csv`](docs/results/table3_perclass.csv) — transcribed
> from the notebook's printed `classification_report` (not yet included in `eval_summary.json`).

### 4. Calibration Analysis

Expected Calibration Error (ECE) before and after temperature scaling, with **zero change in accuracy**:

| Configuration | ECE (↓ better) | Avg. Confidence | Accuracy |
|---|---:|---:|---:|
| Raw softmax (T = 1.0) | 0.0337 | 0.9547 | 0.9213 |
| Temperature scaled (T = 1.144) | 0.0175 | 0.9380 | 0.9213 |

Temperature scaling reduces ECE by **48%**, confirming that raw Legal-BERT is meaningfully over-confident and that calibration materially improves the reliability of cascade routing decisions.

> **Source:** [`docs/results/table4_calibration.csv`](docs/results/table4_calibration.csv) —
> `T_fitted`, `ece_raw`, `ece_scaled` from
> [`docs/results/eval_summary.json`](docs/results/eval_summary.json); the fitted temperature is
> loaded at runtime from [`config/calibration.json`](config/calibration.json) (see
> [`utils/calibration.py`](utils/calibration.py)), not hardcoded.

### 5. CCIC Routing Efficiency

**Confidence-first routing** (Tier 1 evaluated first) on the evaluation set:

| Tier | Model Used | Clauses Handled | % of Total |
|---|---|---:|---:|
| Tier 1 | Legal-BERT (calibrated conf ≥ 0.60) | 3,367 | 97.1% |
| Tier 2 | TF-IDF + LogReg (conf ≥ 0.50) | 36 | 1.0% |
| Tier 3 | Keyword matching (fallback) | 65 | 1.9% |

End-to-end Macro-F1 under this configuration: **92.24%**.

**Lightweight-first routing** (cheap tiers evaluated before the transformer) sends only 16.2% of clauses to Legal-BERT, cutting total inference time by roughly **83%** (7.4s vs. 43.7s on the test set; 0.09ms/sentence for TF-IDF+LR vs. 12.6ms/sentence for Legal-BERT) at a cost of 0.5 Macro-F1 points (91.72% vs. 92.24%) — a favorable latency/accuracy trade-off for high-throughput deployments.

> **Source:** [`docs/results/table5_cascade.csv`](docs/results/table5_cascade.csv) and
> [`docs/results/latency_ms.csv`](docs/results/latency_ms.csv), derived from
> [`docs/results/eval_summary.json`](docs/results/eval_summary.json).

### 6. Comparison with Prior Legal NLP Work

| System | Method | Task | Best F1/Acc | Severity Estimation |
|---|---|---|---:|:---:|
| Milosevic et al. | Rule-based | Binary risk | 72.0% F1 | No |
| Indukuri & Krishna | SVM + BoW | NDA clause | 79.3% Acc | No |
| Chalkidis et al. (Legal-BERT) | Transformer | Multi-label | 71.4% F1 | No |
| Lippi et al. (CLAUDETTE) | SVM | Binary unfair | 79.0% F1 | No |
| Zheng et al. | RoBERTa | Binary unfairness | 81.2% F1 | No |
| Tuggener et al. (LEDGAR) | BERT | 100-class | 83.6% F1 | No |
| Koreeda & Manning (CUAD) | RoBERTa-large | 41-class | 45.2% F1 | No |
| **LexGuard (this work)** | **CCIC + Legal-BERT + HSE** | **6-class risk** | **92.24% F1** | **High/Medium/Low** |

LexGuard improves 8.6 Macro-F1 points over Tuggener et al. and 20.8 points over Chalkidis et al., and is the only system in this comparison that produces a calibrated, fine-grained severity estimate alongside classification.

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

### Fine-tune Legal-BERT (CCIC Tier 1 — proposed model)
```bash
# Requires GPU with 6GB+ VRAM (or Google Colab T4)
pip install -r requirements-dev.txt
python train_model.py --mode transformer --model legal-bert \
    --data data/training_merged.csv --epochs 3 --batch_size 16
# AdamW, lr=2e-5, weight decay=0.01, warmup ratio=0.10, max seq len=256
# Saves: models/transformer_risk_classifier/ + training_meta.json (ablation_role: proposed)
```

### Fine-tune distilroberta (CCIC Tier 1 — baseline for ablation)
```bash
python train_model.py --mode transformer --model distilroberta \
    --data data/training_merged.csv --epochs 3 --batch_size 16
# Saves: training_meta.json (ablation_role: baseline)
```

### Build merged training corpus (CUAD + LEDGAR, 23,525 sentences)
```bash
pip install datasets pandas
python scripts/map_cuad.py --dataset both
python scripts/map_cuad.py --merge --output data/training_merged.csv
```

---

## Testing

```bash
pip install -r requirements-dev.txt
pytest tests/
```

`tests/test_calibration.py`, `test_severity_estimator.py`, and `test_cascade.py` are pure-Python
unit tests (no model files required). `tests/test_ml_classifier.py` and
`tests/test_transformer_integration.py` load the real `models/risk_classifier.joblib` and
`models/transformer_risk_classifier/` artifacts and are skipped automatically if those files aren't
present in the checkout.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `RISK_DETECTOR` | `auto` | CCIC mode: `auto` \| `transformer` \| `ml` \| `keyword` |
| `TRANSFORMER_THRESHOLD` | `0.60` | Tier 1 calibrated confidence threshold τ_t |
| `ML_RISK_THRESHOLD` | `0.50` | Tier 2 confidence threshold τ_m |
| `CCIC_TEMPERATURE` | `1.144` (from `config/calibration.json`) | Temperature scaling factor T, fitted on the validation set (Guo et al., 2017); overrides the config file value when set |
| `TRANSFORMER_MODEL_DIR` | `models/transformer_risk_classifier` | Path to fine-tuned model |
| `OPENAI_API_KEY` | *(optional)* | Enables GPT-4o-mini for `/explain` route |
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
    "temperature": 1.144,
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

Returns a GPT-4o-mini explanation if `OPENAI_API_KEY` is set, otherwise a template-based fallback.

---

## Deployment

LexGuard is containerized via Docker and deployed to **Render**; the underlying architecture is also designed to run on Hugging Face Spaces or Vercel serverless environments.

> [!TIP]
> **Live Demo:** [lexguard-app-production.up.render.app](https://lexguard-app-production.up.render.app)

---

## Limitations and Future Work

- The **None** class has the lowest per-class F1 (0.87), driven by confusion with Obligation and Termination — sharper separation of these categories is an open problem.
- HSE signal weights (0.30 / 0.20 / 0.30 / 0.20) are heuristically tuned; learning them from annotated severity labels via multi-task learning is a natural next step.
- LexGuard currently classifies at the sentence level without clause-boundary awareness, which can miss risk constructions that span multiple sentences.
- Extending the approach to multilingual contracts using multilingual Legal-BERT variants is a promising direction.

---

## References

1. Y. Koreeda and C. D. Manning, "ContractNLI: A dataset for document-level natural language inference for contracts," in *Proc. EMNLP Findings*, 2021, pp. 1-8.
2. D. Tuggener, P. von Däniken, T. Peetz, and M. Cieliebak, "LEDGAR: A large-scale multi-label corpus for text classification of legal provisions in contracts," in *Proc. LREC*, 2020, pp. 1235-1241.
3. D. M. Katz, M. J. Bommarito, and J. Blackman, "A general approach for predicting the behavior of the Supreme Court of the United States," *PLOS ONE*, vol. 12, no. 4, 2017.
4. J. Devlin, M. W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of deep bidirectional transformers for language understanding," in *Proc. NAACL-HLT*, 2019, pp. 4171-4186.
5. Y. Liu et al., "RoBERTa: A robustly optimized BERT pretraining approach," *arXiv:1907.11692*, 2019.
6. I. Chalkidis et al., "LEGAL-BERT: The muppets straight out of law school," in *Proc. EMNLP Findings*, 2020, pp. 2898-2904.
7. C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in *Proc. ICML*, 2017, pp. 1321-1330.
8. T. Xin, A. Ghosh, and G. Carenini, "Cascade inference for efficient NLP: A survey," *arXiv:2210.07996*, 2022.
9. N. Milosevic, C. Cerovic, and M. Gaber, "A framework for information extraction from financial documents," in *Proc. JURIX*, 2017, pp. 111-120.
10. K. K. Indukuri and P. R. Krishna, "Mining e-contract documents to classify clauses," in *Proc. ACM India*, 2010, pp. 1-5.
11. Y. Koreeda and C. D. Manning, "CUAD: An expert-annotated NLP dataset for legal contract review," in *Proc. NeurIPS Datasets and Benchmarks*, 2021.
12. D. Tuggener et al., "LEDGAR," in *Proc. LREC*, 2020, pp. 1235-1241.
13. L. Zheng et al., "When does pretraining help? Assessing self-supervised learning for law and the CaseHOLD dataset," in *Proc. ICAIL*, 2021, pp. 159-168.
14. S. Desai and G. Durrett, "Calibration of pre-trained transformers," in *Proc. EMNLP*, 2020, pp. 295-302.
15. M. Bommarito and D. M. Katz, "GPT takes the bar exam," *arXiv:2212.14402*, 2022.
16. M. Lippi et al., "CLAUDETTE: An automated detector of potentially unfair clauses in online terms of service," *Artif. Intell. Law*, vol. 27, no. 2, pp. 117-139, 2019.
17. M. Leivaditi, J. Rossi, and A. Nourbakhsh, "Benchmark for contract understanding," *arXiv:2011.05765*, 2020.
18. E. Loper and S. Bird, "NLTK: The natural language toolkit," in *Proc. ACL Workshop*, 2002, pp. 63-70.
19. T. Wolf et al., "HuggingFace's Transformers: State-of-the-art NLP," in *Proc. EMNLP (System Demonstrations)*, 2020, pp. 38-45.
20. I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in *Proc. ICLR*, 2019.
21. C. Shelley, "Shall, will, and the modal logic of obligation," *J. Legal Language*, vol. 18, no. 1, pp. 22-45, 2001.
22. OpenAI, "GPT-4 technical report," *arXiv:2303.08774*, 2023.
23. I. Chalkidis et al., "LexGLUE: A benchmark dataset for legal language understanding in English," in *Proc. ACL*, 2022, pp. 4310-4330.

*Full reference list (32 sources) is available in the accompanying paper.*

---

## Citation

If you use this work, please cite:
```bibtex
@inproceedings{lexguard2025,
  title     = {LexGuard: Automated Legal Risk Detection and Severity Estimation in
               Contracts Using NLP-Driven Confidence-Calibrated Inference Cascades},
  author    = {Amarthi, Hemanth Kumar and Rao K, Purushottama and M, Dhanesh Sai},
  booktitle = {Proceedings of JURIX / COLING / AICL Workshop},
  year      = {2025}
}
```
