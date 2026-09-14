# Dataset statistics — CIC-IDS2017

_Generated 2026-09-09T09:19:59+00:00 — seed=42_

> Figures below are computed by `scripts/prepare_datasets.py --dataset cicids2017` from the real dataset files. Nothing here is hand-entered.

> **Sub-sampling:** seeded (seed=42) label-stratified sub-sample to ~200,000 of 2,830,743 rows; classes with < 12,000 rows kept in full (memory-driven, reproducible)

## 1. Samples

| set | rows |
|---|---:|
| raw train pool | 170,000 |
| raw held-out test | 30,000 |
| grand total | 189,910 |

## 2. Cleaning

- missing values (train pool / test): 264 / 56
- exact full-row duplicates in train pool: 10090 (dropped: 10090)
- feature-only duplicates in train pool: 10101 — rows with an identical 77-feature vector but a different label; kept in the training pool (dropping them changes the class balance)

## 3. Features

- before preprocessing: **77**
- after one-hot encoding + variance filter: **69**
- after selection: **20**
- constant feature(s) dropped: ['num__Bwd PSH Flags', 'num__Bwd URG Flags', 'num__Fwd Avg Bytes/Bulk', 'num__Fwd Avg Packets/Bulk', 'num__Fwd Avg Bulk Rate', 'num__Bwd Avg Bytes/Bulk', 'num__Bwd Avg Packets/Bulk', 'num__Bwd Avg Bulk Rate']
- selection method: top-20 by mutual_info_classif (fit on train only, seed=42)

### Selected features

```
num__Destination Port
num__Flow Duration
num__Total Length of Fwd Packets
num__Total Length of Bwd Packets
num__Fwd Packet Length Max
num__Bwd Packet Length Max
num__Bwd Packet Length Mean
num__Flow IAT Max
num__Fwd IAT Total
num__Fwd IAT Max
num__Max Packet Length
num__Packet Length Mean
num__Packet Length Std
num__Packet Length Variance
num__Average Packet Size
num__Avg Bwd Segment Size
num__Subflow Fwd Bytes
num__Subflow Bwd Bytes
num__Init_Win_bytes_forward
num__Init_Win_bytes_backward
```

### Top-20 features by mutual information (train only)

| feature | MI |
|---|---:|
| num__Packet Length Std | 0.418641 |
| num__Packet Length Variance | 0.418205 |
| num__Destination Port | 0.402302 |
| num__Average Packet Size | 0.390796 |
| num__Packet Length Mean | 0.378396 |
| num__Max Packet Length | 0.369956 |
| num__Avg Bwd Segment Size | 0.364985 |
| num__Bwd Packet Length Mean | 0.363590 |
| num__Total Length of Bwd Packets | 0.363579 |
| num__Subflow Bwd Bytes | 0.361799 |
| num__Init_Win_bytes_forward | 0.349086 |
| num__Init_Win_bytes_backward | 0.338432 |
| num__Bwd Packet Length Max | 0.338075 |
| num__Subflow Fwd Bytes | 0.337789 |
| num__Total Length of Fwd Packets | 0.337362 |
| num__Fwd IAT Max | 0.323636 |
| num__Fwd Packet Length Max | 0.314900 |
| num__Flow IAT Max | 0.302797 |
| num__Fwd IAT Total | 0.294629 |
| num__Flow Duration | 0.293964 |

## 4. Train / validation / test partitions

Split mode: **predefined** — no official split exists; loader makes a seeded 85/15 stratified split of the sub-sample into a train pool + held-out test set, then prepare_dataset carves train/validation from the pool  
Stratified on: attack_category (8-class)

| partition | rows | % of total |
|---|---:|---:|
| train | 131,424 | 69.20% |
| validation | 28,486 | 15.00% |
| test | 30,000 | 15.80% |

## 5. Class distribution — binary (0 = normal/benign, 1 = attack)

### train

| class | count | percent |
|---|---:|---:|
| 0 | 89,005 | 67.72% |
| 1 | 42,419 | 32.28% |

### validation

| class | count | percent |
|---|---:|---:|
| 0 | 19,292 | 67.72% |
| 1 | 9,194 | 32.28% |

### test

| class | count | percent |
|---|---:|---:|
| 0 | 19,592 | 65.31% |
| 1 | 10,408 | 34.69% |

## 6. Class distribution — attack_category (8-class)

### train

| class | count | percent |
|---|---:|---:|
| benign | 89,005 | 67.72% |
| botnet | 1,366 | 1.04% |
| brute_force | 6,482 | 4.93% |
| ddos | 5,139 | 3.91% |
| dos | 21,808 | 16.59% |
| infiltration | 25 | 0.02% |
| probe | 6,102 | 4.64% |
| web_attack | 1,497 | 1.14% |

### validation

| class | count | percent |
|---|---:|---:|
| benign | 19,292 | 67.72% |
| botnet | 296 | 1.04% |
| brute_force | 1,405 | 4.93% |
| ddos | 1,114 | 3.91% |
| dos | 4,727 | 16.59% |
| infiltration | 6 | 0.02% |
| probe | 1,322 | 4.64% |
| web_attack | 324 | 1.14% |

### test

| class | count | percent |
|---|---:|---:|
| benign | 19,592 | 65.31% |
| botnet | 295 | 0.98% |
| brute_force | 2,075 | 6.92% |
| ddos | 1,104 | 3.68% |
| dos | 5,232 | 17.44% |
| infiltration | 5 | 0.02% |
| probe | 1,370 | 4.57% |
| web_attack | 327 | 1.09% |

## 7. Class balancing

- primary target: **binary**
- method: **smote** (SMOTE k_neighbors = 5)
- note: CIC-IDS2017 is benign-heavy; SMOTE is applied to the training partition only. class_weight='balanced' is the alternative used by the model scripts.

### train — before balancing

| class | count | percent |
|---|---:|---:|
| 0 | 89,005 | 67.72% |
| 1 | 42,419 | 32.28% |

### train — after balancing

| class | count | percent |
|---|---:|---:|
| 0 | 89,005 | 50.00% |
| 1 | 89,005 | 50.00% |

