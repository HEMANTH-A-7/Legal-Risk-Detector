import argparse
import csv
import json
import os
import random
import sys
from typing import Dict, List, Tuple


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _read_csv(path: str) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames or "sentence" not in r.fieldnames or "label" not in r.fieldnames:
            raise ValueError("Training CSV must have columns: sentence,label")
        for row in r:
            s = (row.get("sentence") or "").strip()
            l = (row.get("label") or "").strip()
            if not s or not l:
                continue
            rows.append((" ".join(s.split()), l))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join("data", "training_merged.csv"))
    parser.add_argument("--out", default=os.path.join("models", "transformer_risk_classifier"))
    parser.add_argument("--base_model", default=os.getenv("TRANSFORMER_BASE_MODEL", "distilroberta-base"))
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="Limit number of examples for quick smoke runs")
    args = parser.parse_args()

    try:
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
        import numpy as np
        from sklearn.metrics import classification_report
    except Exception as e:
        raise RuntimeError(
            "Missing training dependencies. Install with: pip3 install -r requirements-dev.txt"
        ) from e

    random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    rows = _read_csv(args.data)
    if args.limit and args.limit > 0:
        random.shuffle(rows)
        rows = rows[: int(args.limit)]

    labels = sorted({l for _, l in rows})
    label2id: Dict[str, int] = {l: i for i, l in enumerate(labels)}
    id2label: Dict[int, str] = {i: l for l, i in label2id.items()}

    texts = [s for s, _ in rows]
    y = [label2id[l] for _, l in rows]

    idx = list(range(len(texts)))
    random.shuffle(idx)
    split = int(0.9 * len(idx))
    train_idx = idx[:split]
    eval_idx = idx[split:]

    train_ds = Dataset.from_dict({"text": [texts[i] for i in train_idx], "label": [y[i] for i in train_idx]})
    eval_ds = Dataset.from_dict({"text": [texts[i] for i in eval_idx], "label": [y[i] for i in eval_idx]})

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=int(args.max_len))

    train_ds = train_ds.map(tokenize, batched=True)
    eval_ds = eval_ds.map(tokenize, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(labels),
        label2id=label2id,
        id2label={str(k): v for k, v in id2label.items()},
    )

    def compute_metrics(eval_pred):
        logits, labels_arr = eval_pred
        preds = np.argmax(logits, axis=-1)
        report = classification_report(labels_arr, preds, output_dict=True, zero_division=0)
        return {
            "accuracy": float(report["accuracy"]),
            "macro_f1": float(report["macro avg"]["f1-score"]),
            "weighted_f1": float(report["weighted avg"]["f1-score"]),
        }

    training_args = TrainingArguments(
        output_dir=os.path.join(args.out, "_runs"),
        learning_rate=float(args.lr),
        per_device_train_batch_size=int(args.batch),
        per_device_eval_batch_size=int(args.batch),
        num_train_epochs=float(args.epochs),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        save_only_model=True,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=50,
        report_to=[],
        seed=int(args.seed),
        use_cpu=True,
        do_train=True,
        do_eval=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()

    os.makedirs(args.out, exist_ok=True)
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)

    meta = {
        "base_model": args.base_model,
        "labels": labels,
        "label2id": label2id,
        "metrics": metrics,
        "data": args.data,
        "max_len": int(args.max_len),
    }
    with open(os.path.join(args.out, "training_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved transformer model to: {args.out}")
    print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
