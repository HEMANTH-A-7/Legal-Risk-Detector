import argparse
import csv
import os
import random
import sys
from typing import Dict, List, Set, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.nlp_utils import detect_risk_types, segment_sentences


POSITIVE_COLUMN_TO_LABEL = {
    "Uncapped Liability": "Liability",
    "Cap On Liability": "Liability",
    "Liquidated Damages": "Penalty",
    "Termination For Convenience": "Termination",
    "Post-Termination Services": "Termination",
    "Audit Rights": "Obligation",
    "Insurance": "Obligation",
    "Governing Law": "Arbitration",
}


NONE_COLUMNS = [
    "Document Name",
    "Parties",
    "Agreement Date",
    "Effective Date",
    "Expiration Date",
    "Renewal Term",
    "Notice to Terminate Renewal",
]


def _clean(text: str) -> str:
    t = (text or "").replace("<omitted>", " ").strip()
    return " ".join(t.split())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=os.path.join("data", "hf", "cuad", "CUAD_v1", "master_clauses.csv"),
    )
    parser.add_argument(
        "--out",
        default=os.path.join("data", "training_data_cuad_hf.csv"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--none_ratio", type=float, default=1.0)
    args = parser.parse_args()

    random.seed(args.seed)

    positives: List[Tuple[str, str]] = []
    seen_pos: Set[Tuple[str, str]] = set()
    none_candidates: List[str] = []
    seen_none: Set[str] = set()

    with open(args.input, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for col, label in POSITIVE_COLUMN_TO_LABEL.items():
                raw = row.get(col) or ""
                text = _clean(raw)
                if not text:
                    continue
                for sent in segment_sentences(text):
                    s = _clean(sent)
                    if not s:
                        continue
                    key = (s, label)
                    if key in seen_pos:
                        continue
                    seen_pos.add(key)
                    positives.append(key)

            for col in NONE_COLUMNS:
                raw = row.get(col) or ""
                text = _clean(raw)
                if not text:
                    continue
                for sent in segment_sentences(text):
                    s = _clean(sent)
                    if not s:
                        continue
                    if detect_risk_types(s):
                        continue
                    if s in seen_none:
                        continue
                    seen_none.add(s)
                    none_candidates.append(s)

    target_none = int(len(positives) * args.none_ratio)
    random.shuffle(none_candidates)
    none_samples = none_candidates[:target_none]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sentence", "label"])
        for s, label in positives:
            writer.writerow([s, label])
        for s in none_samples:
            writer.writerow([s, "None"])

    counts: Dict[str, int] = {}
    for _, label in positives:
        counts[label] = counts.get(label, 0) + 1
    counts["None"] = len(none_samples)

    print(f"Wrote: {args.out}")
    print("Counts:", counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
