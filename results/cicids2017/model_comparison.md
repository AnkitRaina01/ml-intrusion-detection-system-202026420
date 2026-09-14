# Phase 4 — within-dataset model comparison (CIC-IDS2017, binary task)

_Generated 2026-09-09T11:51:30Z · seed 42_

> **Within-dataset evaluation.** All figures are computed from real predictions on the CIC-IDS2017 held-out **test split** by `scripts/train_eval_datasets.py`. Cross-dataset generalisation is reported separately in `results/cross_dataset/generalization.md`.

fit on train split, hyper-parameters chosen by macro-F1 on the validation split, test split evaluated once here; no refit on train+val; class_weight='balanced'. Identical to Phase 3 (NSL-KDD) so the datasets are directly comparable.

Features: 20 selected by mutual information on the training split only (leakage-safe).

| model | test n | accuracy | precision (attack) | recall (attack) | F1 (attack) | macro-F1 | ROC-AUC | latency (ms/sample) | throughput (/s) | train (s) | val macro-F1 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Logistic Regression | 30,000 | 0.8317 | 0.7946 | 0.6946 | 0.7412 | 0.8083 | 0.9151 | 0.0004 | 2,849,743 | 9.7 | 0.8494 |
| Random Forest | 30,000 | 0.9978 | 0.9962 | 0.9975 | 0.9968 | 0.9976 | 0.9998 | 0.0075 | 133,750 | 329.1 | 0.9974 |
| SVM (RBF) | 30,000 | 0.8953 | 0.7800 | 0.9727 | 0.8657 | 0.8900 | 0.9595 | 0.2441 | 4,097 | 46.7 | 0.8938 |

## Per-class detail & confusion matrices
See `metrics.json` (`per_class`, `confusion_matrix`) and `plots/confusion_matrix_*.png`.

## Notes

- **svm_rbf**: trained on a seeded stratified subsample; ROC-AUC from decision_function
- **lstm**: excluded for flow datasets - see DEVELOPMENT_NOTES.md 11.4
