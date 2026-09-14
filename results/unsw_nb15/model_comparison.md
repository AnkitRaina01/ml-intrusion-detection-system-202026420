# Phase 4 — within-dataset model comparison (UNSW-NB15, binary task)

_Generated 2026-09-09T11:57:06Z · seed 42_

> **Within-dataset evaluation.** All figures are computed from real predictions on the UNSW-NB15 held-out **test split** by `scripts/train_eval_datasets.py`. Cross-dataset generalisation is reported separately in `results/cross_dataset/generalization.md`.

fit on train split, hyper-parameters chosen by macro-F1 on the validation split, test split evaluated once here; no refit on train+val; class_weight='balanced'. Identical to Phase 3 (NSL-KDD) so the datasets are directly comparable.

Features: 20 selected by mutual information on the training split only (leakage-safe).

| model | test n | accuracy | precision (attack) | recall (attack) | F1 (attack) | macro-F1 | ROC-AUC | latency (ms/sample) | throughput (/s) | train (s) | val macro-F1 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Logistic Regression | 82,332 | 0.7707 | 0.7122 | 0.9795 | 0.8247 | 0.7467 | 0.7438 | 0.0002 | 5,824,465 | 4.4 | 0.8746 |
| Random Forest | 82,332 | 0.8608 | 0.8098 | 0.9766 | 0.8854 | 0.8541 | 0.9769 | 0.0279 | 35,790 | 181.3 | 0.9208 |
| SVM (RBF) | 82,332 | 0.8076 | 0.7416 | 0.9986 | 0.8511 | 0.7897 | 0.8991 | 0.2569 | 3,893 | 55.2 | 0.9006 |

## Per-class detail & confusion matrices
See `metrics.json` (`per_class`, `confusion_matrix`) and `plots/confusion_matrix_*.png`.

## Notes

- **svm_rbf**: trained on a seeded stratified subsample; ROC-AUC from decision_function
- **lstm**: excluded for flow datasets - see DEVELOPMENT_NOTES.md 11.4
