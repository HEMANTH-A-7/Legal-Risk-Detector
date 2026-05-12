"""
LexGuard Training Script — Ablation-Ready
==========================================
Supports two training modes:
    1. ML mode (TF-IDF + Logistic Regression) — CCIC Tier 2
    2. Transformer mode (Legal-BERT / distilroberta) — CCIC Tier 1

Ablation study flags (for research paper Table):
    --model legal-bert    → nlpaueb/legal-bert-base-uncased (proposed)
    --model distilroberta → distilroberta-base (baseline)

Usage:
    # Train ML model (Tier 2):
    python train_model.py --mode ml --data data/training_merged.csv

    # Fine-tune Legal-BERT (Tier 1, PROPOSED):
    python train_model.py --mode transformer --model legal-bert \\
        --data data/training_merged.csv --epochs 3

    # Fine-tune distilroberta (Tier 1, BASELINE):
    python train_model.py --mode transformer --model distilroberta \\
        --data data/training_merged.csv --epochs 3
"""

import argparse
import csv
import json
import os

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score


# ─── Model Registry ───────────────────────────────────────────────────────────
TRANSFORMER_MODELS = {
    # Proposed (domain-adapted, legal corpus pre-training)
    "legal-bert":       "nlpaueb/legal-bert-base-uncased",
    # Baseline (general, no legal pre-training)
    "distilroberta":    "distilroberta-base",
    # Optional (larger, better if GPU available)
    "legalbert-large":  "pile-of-law/legalbert-large-1.7M-2",
}


# ─── Data Loader ──────────────────────────────────────────────────────────────

def load_data(data_path: str):
    X, y = [], []
    with open(data_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sentence = (row.get("sentence") or "").strip()
            label = (row.get("label") or "").strip()
            if not sentence or not label:
                continue
            X.append(sentence)
            y.append(label)
    if len(X) < 20:
        raise RuntimeError("Training data too small. Add more labeled examples.")
    return X, y


# ─── ML Training (CCIC Tier 2) ───────────────────────────────────────────────

def train_ml(args, X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42,
        stratify=y if len(set(y)) > 1 else None,
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, args.ngram_max),
            min_df=1,
            max_features=args.max_features,
            sublinear_tf=True,           # Research addition: log-scaled TF
        )),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            C=args.reg_c,
        )),
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    print(report)
    print(f"Macro-F1    (ML): {macro_f1:.4f}")
    print(f"Weighted-F1 (ML): {weighted_f1:.4f}")

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
    joblib.dump(
        {
            "pipeline":  pipeline,
            "labels":    sorted(set(y)),
            "meta": {
                "model_type":   "tfidf_logreg",
                "data_path":    args.data,
                "macro_f1":     round(macro_f1, 4),
                "weighted_f1":  round(weighted_f1, 4),
                "ablation_role": "tier2_baseline",
            },
        },
        args.out,
    )
    print(f"Saved ML model to: {args.out}")
    return macro_f1


# ─── Transformer Fine-Tuning (CCIC Tier 1) ───────────────────────────────────

def train_transformer(args, X, y):
    try:
        from transformers import (
            AutoTokenizer, AutoModelForSequenceClassification,
            TrainingArguments, Trainer, DataCollatorWithPadding,
        )
        import torch
        from torch.utils.data import Dataset as TorchDataset
    except ImportError:
        print("ERROR: Install transformers and torch: pip install transformers torch")
        return

    model_name = TRANSFORMER_MODELS.get(args.model, args.model)
    print(f"\nFine-tuning: {model_name}")
    print(f"  (This is the {'PROPOSED' if args.model == 'legal-bert' else 'BASELINE'} model)")

    labels = sorted(set(y))
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    y_ids = [label2id[label] for label in y]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_ids, test_size=0.2, random_state=42,
        stratify=y_ids if len(set(y_ids)) > 1 else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    class ClauseDataset(TorchDataset):
        def __init__(self, texts, labels):
            self.enc = tokenizer(texts, truncation=True, max_length=256, padding=False)
            self.labels = labels
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

    train_dataset = ClauseDataset(X_train, y_train)
    eval_dataset  = ClauseDataset(X_test,  y_test)
    data_collator = DataCollatorWithPadding(tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    out_dir = args.transformer_out or os.path.join("models", "transformer_risk_classifier")

    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_ratio=0.1,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_dir=os.path.join(out_dir, "logs"),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("\nStarting training...")
    trainer.train()

    # Evaluate and compute Macro-F1 / Weighted-F1 for ablation table
    predictions = trainer.predict(eval_dataset)
    y_pred_ids = predictions.predictions.argmax(axis=-1)
    y_pred_labels = [id2label[i] for i in y_pred_ids]
    y_true_labels = [id2label[i] for i in y_test]

    report = classification_report(y_true_labels, y_pred_labels, zero_division=0)
    macro_f1 = f1_score(y_true_labels, y_pred_labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true_labels, y_pred_labels, average="weighted", zero_division=0)

    print("\n" + report)
    print(f"Macro-F1    ({args.model}): {macro_f1:.4f}")
    print(f"Weighted-F1 ({args.model}): {weighted_f1:.4f}")

    # Save model + tokenizer
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    # BUG FIX: was predictions.metrics.get("test_loss", 0) — HF Trainer uses "eval_loss"
    eval_loss_val = predictions.metrics.get("eval_loss", 0.0)

    # Write ablation-ready training_meta.json (flat format per research spec)
    meta = {
        "base_model":       model_name,
        "model_alias":      args.model,
        "labels":           labels,
        "macro_f1":         round(macro_f1, 4),
        "weighted_f1":      round(weighted_f1, 4),
        "eval_loss":        round(float(eval_loss_val), 4),
        "epochs":           args.epochs,
        "data_path":        args.data,
        "ablation_role":    "proposed" if args.model == "legal-bert" else "baseline",
    }
    meta_path = os.path.join(out_dir, "training_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved transformer model to: {out_dir}")
    print(f"Ablation meta saved to:     {meta_path}")
    return macro_f1


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="LexGuard training script (ML + Transformer)")

    # Common
    parser.add_argument("--data",        default=os.path.join("data", "training_data.csv"))
    parser.add_argument("--mode",        choices=["ml", "transformer"], default="ml",
                        help="Training mode: ml (TF-IDF+LR) or transformer (Legal-BERT/distilroberta)")

    # ML-specific
    parser.add_argument("--out",         default=os.path.join("models", "risk_classifier.joblib"))
    parser.add_argument("--max_features",type=int,   default=20000)
    parser.add_argument("--ngram_max",   type=int,   default=2)
    parser.add_argument("--reg_c",       type=float, default=1.0,
                        help="Logistic Regression regularization (higher = less reg)")

    # Transformer-specific
    parser.add_argument("--model",       default="legal-bert",
                        choices=list(TRANSFORMER_MODELS.keys()),
                        help="legal-bert (proposed) or distilroberta (baseline)")
    parser.add_argument("--transformer_out", default=None,
                        help="Output directory for transformer model")
    parser.add_argument("--epochs",      type=int,   default=3)
    parser.add_argument("--batch_size",  type=int,   default=16)

    args = parser.parse_args()

    print(f"Loading data from: {args.data}")
    X, y = load_data(args.data)
    print(f"Loaded {len(X)} samples, {len(set(y))} classes: {sorted(set(y))}")

    if args.mode == "ml":
        train_ml(args, X, y)
    else:
        train_transformer(args, X, y)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
