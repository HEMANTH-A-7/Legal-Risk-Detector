import argparse
import csv
import os
import random
import sys
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple


def _clean(text: str) -> str:
    t = (text or "").strip()
    t = " ".join(t.split())
    return t


def _read_rows(path: str) -> Iterable[Tuple[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return
        if "sentence" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError(f"{path}: expected columns 'sentence' and 'label'")

        for row in reader:
            s = _clean(row.get("sentence") or "")
            l = _clean(row.get("label") or "")
            if not s or not l:
                continue
            yield s, l


def _resolve_conflicts(
    items: List[Tuple[str, str]],
    strategy: str,
    none_label: str,
) -> List[Tuple[str, str]]:
    by_sentence: Dict[str, List[str]] = defaultdict(list)
    for s, l in items:
        by_sentence[s].append(l)

    out: List[Tuple[str, str]] = []
    for s, labels in by_sentence.items():
        uniq = list(dict.fromkeys(labels))
        if len(uniq) == 1:
            out.append((s, uniq[0]))
            continue

        if strategy == "keep_all":
            for l in uniq:
                out.append((s, l))
            continue

        if strategy == "prefer_non_none":
            non_none = [l for l in uniq if l != none_label]
            out.append((s, non_none[0] if non_none else uniq[0]))
            continue

        if strategy == "majority":
            c = Counter(labels)
            out.append((s, c.most_common(1)[0][0]))
            continue

        raise ValueError(f"Unknown conflict strategy: {strategy}")

    return out


def _balance(
    items: List[Tuple[str, str]],
    mode: str,
    seed: int,
) -> List[Tuple[str, str]]:
    if mode == "none":
        return items

    random.seed(seed)
    by_label: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for s, l in items:
        by_label[l].append((s, l))

    sizes = {k: len(v) for k, v in by_label.items()}
    if not sizes:
        return []

    if mode == "downsample_max":
        target = min(sizes.values())
    elif mode.startswith("downsample_to="):
        target = int(mode.split("=", 1)[1])
    else:
        raise ValueError("balance must be one of: none, downsample_max, downsample_to=N")

    out: List[Tuple[str, str]] = []
    for label, rows in by_label.items():
        if len(rows) <= target:
            out.extend(rows)
        else:
            out.extend(random.sample(rows, target))
    random.shuffle(out)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more CSV files with columns: sentence,label",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output CSV path",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Deduplicate identical (sentence,label) pairs",
    )
    parser.add_argument(
        "--conflicts",
        default="prefer_non_none",
        choices=["prefer_non_none", "majority", "keep_all"],
        help="How to handle same sentence with multiple labels",
    )
    parser.add_argument(
        "--none_label",
        default="None",
        help="Label name used for non-risk examples",
    )
    parser.add_argument(
        "--balance",
        default="none",
        help="Balancing strategy: none | downsample_max | downsample_to=N",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    merged: List[Tuple[str, str]] = []
    for p in args.inputs:
        merged.extend(list(_read_rows(p)))

    if args.dedupe:
        merged = list(dict.fromkeys(merged))

    merged = _resolve_conflicts(merged, args.conflicts, args.none_label)
    merged = _balance(merged, args.balance, args.seed)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sentence", "label"])
        for s, l in merged:
            w.writerow([s, l])

    counts = Counter([l for _, l in merged])
    print(f"Wrote: {args.out}")
    print("Counts:", dict(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

