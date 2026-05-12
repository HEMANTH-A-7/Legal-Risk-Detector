import argparse
import csv
import json
import os
import random
import sys
from typing import Dict, List, Optional, Set, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.nlp_utils import segment_sentences


def map_question_to_category(question: str) -> Optional[str]:
    q = (question or "").lower()

    if "arbitration" in q or "venue" in q or "jurisdiction" in q or "governing law" in q:
        return "Arbitration"

    if "termination" in q or "renewal" in q or "survival" in q or "change of control" in q:
        return "Termination"

    if "liquidated" in q or "late payment" in q or "interest" in q or "penalty" in q or "fine" in q:
        return "Penalty"

    if "indemn" in q or "limitation of liability" in q or "cap on liability" in q or "warranty" in q:
        return "Liability"

    if (
        "confidential" in q
        or "insurance" in q
        or "audit" in q
        or "non-compete" in q
        or "non-solicit" in q
        or "assignment" in q
        or "notice" in q
        or "payment" in q
        or "deliver" in q
        or "oblig" in q
    ):
        return "Obligation"

    return None


def _iter_squad_paragraphs(obj: Dict) -> List[Tuple[str, List[Dict]]]:
    out: List[Tuple[str, List[Dict]]] = []
    for item in obj.get("data", []):
        for p in item.get("paragraphs", []):
            context = p.get("context") or ""
            qas = p.get("qas") or []
            if context and qas:
                out.append((context, qas))
    return out


def _find_json_files(input_dir: str) -> List[str]:
    files = []
    for root, _, filenames in os.walk(input_dir):
        for fn in filenames:
            if fn.lower().endswith(".json"):
                files.append(os.path.join(root, fn))
    return sorted(files)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default=os.path.join("data", "kaggle", "aok_beta"))
    parser.add_argument("--out", default=os.path.join("data", "training_data_cuad.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--none_ratio", type=float, default=1.0, help="None samples per positive sample")
    args = parser.parse_args()

    random.seed(args.seed)

    json_files = _find_json_files(args.input_dir)
    if not json_files:
        raise RuntimeError(f"No .json files found in: {args.input_dir}")

    positives: List[Tuple[str, str]] = []
    none_candidates: List[str] = []
    seen_pos: Set[Tuple[str, str]] = set()
    seen_none: Set[str] = set()

    for path in json_files:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        for context, qas in _iter_squad_paragraphs(obj):
            paragraph_positive_snippets: List[str] = []

            for qa in qas:
                category = map_question_to_category(qa.get("question") or "")
                if not category:
                    continue

                for ans in qa.get("answers") or []:
                    text = (ans.get("text") or "").strip()
                    if not text or text.lower() in {"n/a", "na"}:
                        continue

                    paragraph_positive_snippets.append(text)
                    for sent in segment_sentences(text):
                        sent = sent.strip()
                        if not sent:
                            continue
                        key = (sent, category)
                        if key in seen_pos:
                            continue
                        seen_pos.add(key)
                        positives.append(key)

            for sent in segment_sentences(context):
                s = sent.strip()
                if not s:
                    continue
                if any(snippet in s for snippet in paragraph_positive_snippets):
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
        for sent, label in positives:
            writer.writerow([sent, label])
        for sent in none_samples:
            writer.writerow([sent, "None"])

    print(f"Wrote: {args.out}")
    print(f"Positives: {len(positives)}")
    print(f"None: {len(none_samples)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
