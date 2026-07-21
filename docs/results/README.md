# Result Artifacts

Backing data for every number cited in the top-level `README.md`. Source: Kaggle notebook
[`lexguard-final`](https://www.kaggle.com/code/hemanthkumaramarthi/lexguard-final) (seed 42),
training script `train_model.py` / `scripts/map_cuad.py`.

| File | Backs | Source |
|---|---|---|
| `eval_summary.json` | Tables II, IV, V (ablation, calibration, cascade, latency) | Verbatim eval summary exported from the notebook run |
| `table1_corpus_split.csv` | Table I totals (train/val/test, dedup) | Derived from `eval_summary.json` split counts |
| `table2_ablation.csv` | Table II (ablation) | Derived from `eval_summary.json` |
| `table3_perclass.csv` | Table III (per-class F1, Legal-BERT) | Manually transcribed from the notebook's printed `classification_report` — not yet in `eval_summary.json` |
| `table4_calibration.csv` | Table IV (ECE / temperature scaling) | `ece_raw`/`ece_scaled`/`T_fitted` from `eval_summary.json`; `avg_confidence` transcribed from notebook output |
| `table5_cascade.csv` | Table V (CCIC routing) | Derived from `eval_summary.json` `cascade` array |
| `latency_ms.csv` | Table V (per-sentence latency) | `eval_summary.json` `latency_ms` |

## Known gap

Table I's per-class corpus breakdown (e.g. "Liability: 4,000 / 17.0%") in the top-level README is
**not yet backed by a committed artifact**. The merged 23,367-row CUAD+LEDGAR corpus used for this
run is not checked into this repo (`data/training_merged.csv` here is a stale 4,919-row version
predating the merge). Regenerate the merged corpus via `scripts/map_cuad.py --merge` and export a
per-class `value_counts()` to replace this note before treating that specific table as verified.
