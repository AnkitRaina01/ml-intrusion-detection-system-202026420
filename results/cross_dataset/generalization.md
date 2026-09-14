# Phase 4 — cross-dataset generalisation (CIC-IDS2017 ↔ UNSW-NB15)

_Generated 2026-09-09T12:04:39Z · seed 42 · binary task_

> **Within-dataset** rows: trained and tested on the same dataset's own splits (12-feature common schema). **Cross-dataset** rows: the model is trained on the *source* dataset and evaluated on the *other* dataset's held-out test set, standardised with the source's training statistics. No adaptation / fine-tuning. Every number is a real prediction score from `scripts/cross_dataset_experiment.py`.

Common feature space: `dur_s, fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes, fwd_pkt_len_mean, bwd_pkt_len_mean, flow_bytes_per_s, flow_pkts_per_s, fwd_pkts_per_s, bwd_pkts_per_s, down_up_ratio` (12 features). See `common_schema.md` for the exact per-dataset construction and its limitations.

Data: CIC-IDS2017 train 102,571 / test 30,000 (seeded 200,000-row sub-sample); UNSW-NB15 train 58,924 / test 82,332.

## Logistic Regression

| evaluation | accuracy | precision (atk) | recall (atk) | F1 (atk) | macro-F1 | ROC-AUC |
|---|--:|--:|--:|--:|--:|--:|
| within · CIC-IDS2017 | 0.7153 | 0.6283 | 0.4391 | 0.5169 | 0.6575 | 0.6524 |
| within · UNSW-NB15 | 0.7191 | 0.6645 | 0.9893 | 0.7950 | 0.6745 | 0.7760 |
| cross · CIC-IDS2017 → UNSW-NB15 | 0.3955 | 0.1211 | 0.0156 | 0.0277 | 0.2946 | 0.5201 |
| cross · UNSW-NB15 → CIC-IDS2017 | 0.4097 | 0.2962 | 0.5096 | 0.3746 | 0.4078 | 0.4681 |

Generalisation gap (within − cross, macro-F1): CIC-IDS2017 source = **+0.3630**, UNSW-NB15 source = **+0.2667**.

## Random Forest

| evaluation | accuracy | precision (atk) | recall (atk) | F1 (atk) | macro-F1 | ROC-AUC |
|---|--:|--:|--:|--:|--:|--:|
| within · CIC-IDS2017 | 0.9386 | 0.8561 | 0.9894 | 0.9179 | 0.9345 | 0.9930 |
| within · UNSW-NB15 | 0.8564 | 0.8108 | 0.9641 | 0.8808 | 0.8501 | 0.9626 |
| cross · CIC-IDS2017 → UNSW-NB15 | 0.4498 | 1.0000 | 0.0007 | 0.0015 | 0.3109 | 0.3620 |
| cross · UNSW-NB15 → CIC-IDS2017 | 0.3779 | 0.2859 | 0.5297 | 0.3714 | 0.3778 | 0.3816 |

Generalisation gap (within − cross, macro-F1): CIC-IDS2017 source = **+0.6236**, UNSW-NB15 source = **+0.4723**.

## SVM (RBF)

| evaluation | accuracy | precision (atk) | recall (atk) | F1 (atk) | macro-F1 | ROC-AUC |
|---|--:|--:|--:|--:|--:|--:|
| within · CIC-IDS2017 | 0.8331 | 0.7309 | 0.8211 | 0.7734 | 0.8206 | 0.9181 |
| within · UNSW-NB15 | 0.7384 | 0.6808 | 0.9884 | 0.8062 | 0.7019 | 0.6584 |
| cross · CIC-IDS2017 → UNSW-NB15 | 0.4460 | 0.4541 | 0.0306 | 0.0573 | 0.3325 | 0.3843 |
| cross · UNSW-NB15 → CIC-IDS2017 | 0.3984 | 0.3487 | 0.8458 | 0.4938 | 0.3762 | 0.4584 |

Generalisation gap (within − cross, macro-F1): CIC-IDS2017 source = **+0.4881**, UNSW-NB15 source = **+0.3257**.

## Caveats

- 12-feature LOSSY alignment of two different flow-exporter pipelines (CICFlowMeter vs Argus/Bro); see common_schema.md
- cross-dataset = target test data standardised with the SOURCE dataset's training mean/std (standard domain-shift protocol)
- no fine-tuning / adaptation - this measures raw transfer
