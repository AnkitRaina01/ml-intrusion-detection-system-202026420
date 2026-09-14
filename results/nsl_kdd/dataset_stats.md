# Dataset statistics — nsl_kdd

_Generated 2026-09-09T12:09:10+00:00 — seed=42_

> Figures below are computed by `scripts/prepare_nsl_kdd.py` from the real NSL-KDD files. Nothing here is hand-entered.

## 1. Samples

| set | rows |
|---|---:|
| raw KDDTrain+ pool | 125,973 |
| raw KDDTest+ | 22,544 |
| grand total | 148,517 |

## 2. Cleaning

- missing values (train pool / test): 0 / 0
- exact full-row duplicates in train pool: 0 (dropped: 0)
- feature-only duplicates in train pool: 16 — rows with an identical 41-feature vector but a different label; kept in the training pool (dropping them changes the class balance)

## 3. Features

- before preprocessing: **41**
- after one-hot encoding + variance filter: **121**
- after selection: **20**
- constant feature(s) dropped: ['num__num_outbound_cmds']
- selection method: top-20 by mutual_info_classif (fit on train only, seed=42)

### Selected features

```
num__src_bytes
num__dst_bytes
num__logged_in
num__count
num__serror_rate
num__srv_serror_rate
num__same_srv_rate
num__diff_srv_rate
num__dst_host_count
num__dst_host_srv_count
num__dst_host_same_srv_rate
num__dst_host_diff_srv_rate
num__dst_host_same_src_port_rate
num__dst_host_srv_diff_host_rate
num__dst_host_serror_rate
num__dst_host_srv_serror_rate
cat__service_http
cat__service_private
cat__flag_S0
cat__flag_SF
```

### Top-20 features by mutual information (train only)

| feature | MI |
|---|---:|
| num__src_bytes | 0.566951 |
| num__dst_bytes | 0.441671 |
| num__diff_srv_rate | 0.360889 |
| num__same_srv_rate | 0.357633 |
| num__dst_host_srv_count | 0.332286 |
| cat__flag_SF | 0.325324 |
| num__dst_host_same_srv_rate | 0.307241 |
| num__logged_in | 0.290139 |
| num__dst_host_diff_srv_rate | 0.287137 |
| num__dst_host_serror_rate | 0.285459 |
| num__dst_host_srv_serror_rate | 0.283387 |
| num__serror_rate | 0.277671 |
| num__srv_serror_rate | 0.268272 |
| num__count | 0.265309 |
| cat__flag_S0 | 0.256750 |
| num__dst_host_srv_diff_host_rate | 0.188161 |
| cat__service_http | 0.184631 |
| num__dst_host_count | 0.138065 |
| num__dst_host_same_src_port_rate | 0.134180 |
| cat__service_private | 0.117210 |

## 4. Train / validation / test partitions

Split mode: **predefined** — KDDTrain+ -> stratified train/val; KDDTest+ kept as held-out test (contains novel attack types by NSL-KDD design)  
Stratified on: attack_category (5-class)

| partition | rows | % of total |
|---|---:|---:|
| train | 103,695 | 69.82% |
| validation | 22,278 | 15.00% |
| test | 22,544 | 15.18% |

## 5. Class distribution — binary (0 = normal/benign, 1 = attack)

### train

| class | count | percent |
|---|---:|---:|
| 0 | 55,433 | 53.46% |
| 1 | 48,262 | 46.54% |

### validation

| class | count | percent |
|---|---:|---:|
| 0 | 11,910 | 53.46% |
| 1 | 10,368 | 46.54% |

### test

| class | count | percent |
|---|---:|---:|
| 0 | 9,711 | 43.08% |
| 1 | 12,833 | 56.92% |

## 6. Class distribution — attack_category (5-class)

### train

| class | count | percent |
|---|---:|---:|
| dos | 37,805 | 36.46% |
| normal | 55,433 | 53.46% |
| probe | 9,595 | 9.25% |
| r2l | 819 | 0.79% |
| u2r | 43 | 0.04% |

### validation

| class | count | percent |
|---|---:|---:|
| dos | 8,122 | 36.46% |
| normal | 11,910 | 53.46% |
| probe | 2,061 | 9.25% |
| r2l | 176 | 0.79% |
| u2r | 9 | 0.04% |

### test

| class | count | percent |
|---|---:|---:|
| dos | 7,460 | 33.09% |
| normal | 9,711 | 43.08% |
| probe | 2,421 | 10.74% |
| r2l | 2,885 | 12.80% |
| u2r | 67 | 0.30% |

## 7. Class balancing

- primary target: **binary**
- method: **smote** (SMOTE k_neighbors = 5)
- note: NSL-KDD binary target is only mildly imbalanced (~53/47); class_weight is an equally valid alternative.

### train — before balancing

| class | count | percent |
|---|---:|---:|
| 0 | 55,433 | 53.46% |
| 1 | 48,262 | 46.54% |

### train — after balancing

| class | count | percent |
|---|---:|---:|
| 0 | 55,433 | 50.00% |
| 1 | 55,433 | 50.00% |

