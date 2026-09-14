# Dataset statistics — UNSW-NB15

_Generated 2026-09-09T12:11:29+00:00 — seed=42_

> Figures below are computed by `scripts/prepare_datasets.py --dataset unsw_nb15` from the real dataset files. Nothing here is hand-entered.

## 1. Samples

| set | rows |
|---|---:|
| raw train pool | 107,740 |
| raw held-out test | 82,332 |
| grand total | 190,072 |

## 2. Cleaning

- missing values (train pool / test): 0 / 0
- exact full-row duplicates in train pool: 0 (dropped: 0)
- feature-only duplicates in train pool: 6700 — rows with an identical 42-feature vector but a different label; kept in the training pool (dropping them changes the class balance)
- exact full-row duplicates removed by the loader **before** this pipeline: 67,601

## 3. Features

- before preprocessing: **42**
- after one-hot encoding + variance filter: **191**
- after selection: **20**
- constant feature(s) dropped: []
- selection method: top-20 by mutual_info_classif (fit on train only, seed=42)

### Selected features

```
num__dur
num__spkts
num__dpkts
num__sbytes
num__dbytes
num__rate
num__sttl
num__dttl
num__sload
num__dload
num__sinpkt
num__dinpkt
num__sjit
num__djit
num__tcprtt
num__synack
num__ackdat
num__smean
num__dmean
num__ct_state_ttl
```

### Top-20 features by mutual information (train only)

| feature | MI |
|---|---:|
| num__sbytes | 0.457488 |
| num__dbytes | 0.415948 |
| num__sttl | 0.403638 |
| num__dttl | 0.394097 |
| num__ct_state_ttl | 0.387387 |
| num__dmean | 0.327811 |
| num__rate | 0.312370 |
| num__smean | 0.303676 |
| num__dinpkt | 0.302574 |
| num__synack | 0.285260 |
| num__tcprtt | 0.284099 |
| num__dur | 0.283264 |
| num__ackdat | 0.281366 |
| num__dpkts | 0.272423 |
| num__dload | 0.265660 |
| num__sload | 0.263635 |
| num__sinpkt | 0.221690 |
| num__sjit | 0.199054 |
| num__djit | 0.173799 |
| num__spkts | 0.156747 |

## 4. Train / validation / test partitions

Split mode: **predefined** — official partitioned split respected: training-set.csv = train pool (exact duplicate rows removed), testing-set.csv = held-out test; prepare_dataset carves a stratified validation slice from the pool  
Stratified on: attack_category (10-class)

| partition | rows | % of total |
|---|---:|---:|
| train | 79,229 | 41.68% |
| validation | 28,511 | 15.00% |
| test | 82,332 | 43.32% |

## 5. Class distribution — binary (0 = normal/benign, 1 = attack)

### train

| class | count | percent |
|---|---:|---:|
| 0 | 38,158 | 48.16% |
| 1 | 41,071 | 51.84% |

### validation

| class | count | percent |
|---|---:|---:|
| 0 | 13,732 | 48.16% |
| 1 | 14,779 | 51.84% |

### test

| class | count | percent |
|---|---:|---:|
| 0 | 37,000 | 44.94% |
| 1 | 45,332 | 55.06% |

## 6. Class distribution — attack_category (10-class)

### train

| class | count | percent |
|---|---:|---:|
| analysis | 1,172 | 1.48% |
| backdoor | 1,129 | 1.43% |
| dos | 2,799 | 3.53% |
| exploits | 14,593 | 18.42% |
| fuzzers | 11,876 | 14.99% |
| generic | 3,075 | 3.88% |
| normal | 38,158 | 48.16% |
| reconnaissance | 5,532 | 6.98% |
| shellcode | 802 | 1.01% |
| worms | 93 | 0.12% |

### validation

| class | count | percent |
|---|---:|---:|
| analysis | 422 | 1.48% |
| backdoor | 406 | 1.42% |
| dos | 1,007 | 3.53% |
| exploits | 5,251 | 18.42% |
| fuzzers | 4,274 | 14.99% |
| generic | 1,106 | 3.88% |
| normal | 13,732 | 48.16% |
| reconnaissance | 1,990 | 6.98% |
| shellcode | 289 | 1.01% |
| worms | 34 | 0.12% |

### test

| class | count | percent |
|---|---:|---:|
| analysis | 677 | 0.82% |
| backdoor | 583 | 0.71% |
| dos | 4,089 | 4.97% |
| exploits | 11,132 | 13.52% |
| fuzzers | 6,062 | 7.36% |
| generic | 18,871 | 22.92% |
| normal | 37,000 | 44.94% |
| reconnaissance | 3,496 | 4.25% |
| shellcode | 378 | 0.46% |
| worms | 44 | 0.05% |

## 7. Class balancing

- primary target: **binary**
- method: **smote** (SMOTE k_neighbors = 5)
- note: UNSW-NB15 is attack-majority (~68% attack in the training CSV); SMOTE is applied to the training partition only. The 'worms' class is tiny (130 train rows) - SMOTE k_neighbors is reduced accordingly.

### train — before balancing

| class | count | percent |
|---|---:|---:|
| 0 | 38,158 | 48.16% |
| 1 | 41,071 | 51.84% |

### train — after balancing

| class | count | percent |
|---|---:|---:|
| 0 | 41,071 | 50.00% |
| 1 | 41,071 | 50.00% |

