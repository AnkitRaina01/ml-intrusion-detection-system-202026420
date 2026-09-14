# Phase 3 — model comparison (nsl_kdd, binary task)

_Generated 2026-09-08T19:19:17Z · seed 42_

> All figures computed from real predictions on the held-out NSL-KDD test set (KDDTest+) by `scripts/evaluate_models.py`. No values are taken from the repository README.

fit on train, hyper-parameters chosen by macro-F1 on validation, test (KDDTest+) evaluated once here; no refit on train+val; class_weight='balanced' (LSTM: pos_weight in BCE loss).

| model | test n | accuracy | precision (attack) | recall (attack) | F1 (attack) | macro-F1 | ROC-AUC | latency (ms/sample) | throughput (/s) | train (s) | val macro-F1 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Logistic Regression | 22,544 | 0.7194 | 0.9074 | 0.5648 | 0.6962 | 0.7178 | 0.8890 | 0.0001 | 9,879,062 | 3.3 | 0.9606 |
| Random Forest | 22,544 | 0.7706 | 0.9696 | 0.6164 | 0.7537 | 0.7695 | 0.9498 | 0.0077 | 130,256 | 89.6 | 0.9982 |
| SVM (RBF) | 22,544 | 0.7252 | 0.9077 | 0.5757 | 0.7046 | 0.7238 | 0.8350 | 0.0619 | 16,165 | 8.1 | 0.9812 |
| LSTM | 22,535 | 0.7573 | 0.9150 | 0.6323 | 0.7478 | 0.7569 | 0.8846 | 0.0115 | 87,174 | 433.1 | 0.9863 |

## Per-class detail & confusion matrices
See `metrics.json` (`per_class`, `confusion_matrix`) and `plots/confusion_matrix_*.png`.

## Notes

- **svm_rbf**: trained on a seeded stratified subsample; ROC-AUC from decision_function
- **lstm**: PyTorch; evaluated on windowed KDDTest+ (22535/22544 records); see sequence_metadata.json / DEVELOPMENT_NOTES 10.3
