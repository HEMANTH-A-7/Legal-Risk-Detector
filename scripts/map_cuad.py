"""
CUAD -> LexGuard Taxonomy Mapper
================================
Research Contribution: Dataset Construction (Contribution 4)

    Maps the CUAD dataset (41 expert-annotated clause types) and LEDGAR
    (60+ provision categories) to the LexGuard 5-class risk taxonomy.

    This cross-dataset remapping creates a larger, more diverse training corpus
    than any prior work on this specific 5-class taxonomy, enabling reproducible
    evaluation on expert-annotated data.

Datasets:
    - CUAD: Hendrycks et al. (2021), 41 clause types, ~13k QA pairs
    - LEDGAR: Tuggener et al. (2020), 60+ SEC filing provision categories

Output:
    5-class taxonomy: Liability | Penalty | Termination | Obligation | Arbitration

Usage:
    python scripts/map_cuad.py --dataset cuad --output data/cuad_mapped.csv
    python scripts/map_cuad.py --dataset ledgar --output data/ledgar_mapped.csv
    python scripts/map_cuad.py --dataset both
    python scripts/map_cuad.py --merge --output data/training_merged.csv

Requirements:
    pip install datasets pandas
"""

import argparse
import csv
import os
from pathlib import Path


# ─── CUAD → LexGuard 5-Class Mapping ─────────────────────────────────────────
# Reference: Hendrycks et al. (2021) CUAD — 41 clause types mapped to 5 risk classes.
CUAD_TO_LEXGUARD = {
    # ── Liability ──────────────────────────────────────────────────────────────
    "Cap On Liability":                 "Liability",
    "Indemnification Of Indemnitee":    "Liability",
    "Indemnification Of Indemnitor":    "Liability",
    "Limitation Of Liability":          "Liability",
    "Warranty Duration":                "Liability",
    "General Damages":                  "Liability",
    "Consequential Damages":            "Liability",

    # ── Penalty ────────────────────────────────────────────────────────────────
    "Liquidated Damages":               "Penalty",
    "Price Restrictions":               "Penalty",
    "Minimum Commitment":               "Penalty",
    "Revenue/Profit Sharing":           "Penalty",
    "Volume Restriction":               "Penalty",

    # ── Termination ────────────────────────────────────────────────────────────
    "Termination For Convenience":      "Termination",
    "Change Of Control":                "Termination",
    "Anti-Assignment":                  "Termination",
    "Agreement Term":                   "Termination",
    "Renewal Term":                     "Termination",
    "Post-Termination Services":        "Termination",

    # ── Obligation ─────────────────────────────────────────────────────────────
    "Non-Compete":                      "Obligation",
    "Non-Solicitation":                 "Obligation",
    "Exclusivity":                      "Obligation",
    "No-Solicit Of Customers":          "Obligation",
    "No-Solicit Of Employees":          "Obligation",
    "Ip Ownership Assignment":          "Obligation",
    "License Grant":                    "Obligation",
    "Audit Rights":                     "Obligation",
    "Insurance":                        "Obligation",

    # ── Arbitration ────────────────────────────────────────────────────────────
    "Governing Law":                    "Arbitration",
    "Dispute Resolution":               "Arbitration",
    "Venue":                            "Arbitration",
    "Arbitration":                      "Arbitration",
    "Third Party Beneficiary":          "Arbitration",
    "Covenant Not To Sue":              "Arbitration",
}

# ─── LEDGAR → LexGuard 5-Class Mapping ────────────────────────────────────────
# Reference: LEDGAR dataset (Tuggener et al., 2020) — SEC filing provisions.
LEDGAR_TO_LEXGUARD = {
    # ── Liability ──────────────────────────────────────────────────────────────
    "Indemnification":                  "Liability",
    "Liabilities":                      "Liability",
    "Indemnification And Insurance":    "Liability",
    "Damages":                          "Liability",

    # ── Penalty ────────────────────────────────────────────────────────────────
    "Fees":                             "Penalty",
    "Penalties":                        "Penalty",
    "Late Charges":                     "Penalty",
    "Interest":                         "Penalty",
    "Taxes":                            "Penalty",

    # ── Termination ────────────────────────────────────────────────────────────
    "Term":                             "Termination",
    "Termination":                      "Termination",
    "Expiration":                       "Termination",
    "Survival":                         "Termination",

    # ── Obligation ─────────────────────────────────────────────────────────────
    "Obligations":                      "Obligation",
    "Covenants":                        "Obligation",
    "Compliance With Laws":             "Obligation",
    "Reporting Obligations":            "Obligation",
    "Confidentiality":                  "Obligation",
    "Non-Solicitation":                 "Obligation",

    # ── Arbitration ────────────────────────────────────────────────────────────
    "Governing Law":                    "Arbitration",
    "Dispute Resolution":               "Arbitration",
    "Arbitration":                      "Arbitration",
    "Jurisdiction":                     "Arbitration",
}


def download_and_map_cuad(output_path: str, max_samples_per_class: int = 2000):
    """
    Downloads CUAD from Hugging Face and maps to LexGuard taxonomy.
    Saves mapped sentences to a CSV file.
    """
    try:
        from datasets import load_dataset
        import pandas as pd
    except ImportError:
        print("ERROR: Install dependencies: pip install datasets pandas")
        return

    print("Downloading CUAD dataset from Hugging Face...")
    dataset = load_dataset("theatticusproject/cuad-qa", trust_remote_code=True)

    rows = []
    for split_name in dataset:
        split = dataset[split_name]
        for example in split:
            for qa in example.get("qas", []):
                question_title = qa.get("id", "").split("__")[1] if "__" in qa.get("id", "") else ""
                label = CUAD_TO_LEXGUARD.get(question_title)
                if not label:
                    continue
                for answer in qa.get("answers", {}).get("text", []):
                    sentence = answer.strip()
                    if sentence and len(sentence) > 20:
                        rows.append({"sentence": sentence, "label": label, "source": "cuad"})

    print(f"  Mapped {len(rows)} sentences from CUAD")
    _balance_and_save(rows, output_path, max_samples_per_class)


def download_and_map_ledgar(output_path: str, max_samples_per_class: int = 1500):
    """
    Downloads LEDGAR from Hugging Face (via LexGLUE) and maps to LexGuard taxonomy.
    """
    try:
        from datasets import load_dataset
        import pandas as pd
    except ImportError:
        print("ERROR: Install dependencies: pip install datasets pandas")
        return

    print("Downloading LEDGAR dataset from Hugging Face...")
    dataset = load_dataset("coastalcph/lex_glue", "ledgar", trust_remote_code=True)

    rows = []
    label_names = dataset["train"].features["label"].names if hasattr(
        dataset["train"].features["label"], "names") else []

    for split_name in dataset:
        split = dataset[split_name]
        for example in split:
            text = example.get("text", "").strip()
            label_idx = example.get("label", -1)
            if not text or label_idx < 0:
                continue
            label_name = label_names[label_idx] if label_idx < len(label_names) else ""
            mapped = LEDGAR_TO_LEXGUARD.get(label_name)
            if not mapped:
                continue
            rows.append({"sentence": text, "label": mapped, "source": "ledgar"})

    print(f"  Mapped {len(rows)} sentences from LEDGAR")
    _balance_and_save(rows, output_path, max_samples_per_class)


def merge_datasets(existing_csv: str, cuad_csv: str, ledgar_csv: str, output_path: str):
    """
    Merges existing training data with CUAD and LEDGAR mapped data.
    De-duplicates on sentence text to avoid data leakage.
    """
    try:
        import pandas as pd
    except ImportError:
        print("ERROR: Install pandas: pip install pandas")
        return

    dfs = []
    for path, source_name in [(existing_csv, "original"), (cuad_csv, "cuad"), (ledgar_csv, "ledgar")]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            if "source" not in df.columns:
                df["source"] = source_name
            dfs.append(df)
            print(f"  Loaded {len(df)} rows from {source_name}")
        else:
            print(f"  WARNING: {path} not found, skipping.")

    if not dfs:
        print("ERROR: No data files found.")
        return

    merged = pd.concat(dfs, ignore_index=True)
    before = len(merged)
    merged = merged.drop_duplicates(subset=["sentence"]).reset_index(drop=True)
    print(f"  De-duplicated: {before} -> {len(merged)} rows")

    print("\n  Class distribution:")
    print(merged["label"].value_counts().to_string())

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"\n  Merged dataset saved to: {output_path}")


def _balance_and_save(rows, output_path, max_per_class):
    """Balances classes by capping at max_per_class and writes to CSV."""
    try:
        import pandas as pd
    except ImportError:
        _write_csv_plain(rows, output_path)
        return

    df = pd.DataFrame(rows)
    print("\n  Raw class distribution:")
    print(df["label"].value_counts().to_string())

    balanced = (
        df.groupby("label", group_keys=False)
          .apply(lambda x: x.sample(min(len(x), max_per_class), random_state=42))
          .reset_index(drop=True)
    )

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    balanced.to_csv(output_path, index=False)
    print(f"\n  Saved {len(balanced)} balanced rows to: {output_path}")


def _write_csv_plain(rows, output_path):
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sentence", "label", "source"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Map CUAD/LEDGAR to LexGuard 5-class taxonomy"
    )
    parser.add_argument("--dataset", choices=["cuad", "ledgar", "both"], default="cuad",
                        help="Which dataset to download and map")
    parser.add_argument("--output", default=os.path.join("data", "cuad_mapped.csv"),
                        help="Output CSV path")
    parser.add_argument("--merge", action="store_true",
                        help="Merge with existing training data")
    parser.add_argument("--existing", default=os.path.join("data", "training_data.csv"),
                        help="Path to your existing training CSV")
    parser.add_argument("--max-per-class", type=int, default=2000,
                        help="Maximum samples per class (for class balancing)")
    args = parser.parse_args()

    if args.merge:
        cuad_csv   = os.path.join("data", "cuad_mapped.csv")
        ledgar_csv = os.path.join("data", "ledgar_mapped.csv")
        merge_datasets(args.existing, cuad_csv, ledgar_csv, args.output)
    elif args.dataset in ("cuad", "both"):
        out = os.path.join("data", "cuad_mapped.csv") if args.dataset == "both" else args.output
        download_and_map_cuad(out, args.max_per_class)
        if args.dataset == "both":
            download_and_map_ledgar(os.path.join("data", "ledgar_mapped.csv"), args.max_per_class)
    elif args.dataset == "ledgar":
        download_and_map_ledgar(args.output, args.max_per_class)


if __name__ == "__main__":
    main()
