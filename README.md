# Machine-Learning Network Intrusion Detection: A Leakage-Controlled, Cross-Dataset Evaluation

MSc Computer Science project — School of Computing, University of Leeds.
Author: **Ankita Raina** — GitHub identity `AnkitRaina01`

A machine-learning network intrusion detection pipeline built and evaluated
across three public benchmark datasets, served through a REST API and a
monitoring dashboard, with end-to-end detection cost measured rather than
assumed.

The emphasis is **verification over headline accuracy**: the pipeline is
demonstrably free of data leakage, reproducible from a pinned dependency
specification, and every reported number is traceable to a machine-generated
artefact under `results/`.

---

## Overview

Machine-learning intrusion detection routinely reports accuracies above 99 %,
yet operational adoption lags far behind. Three documented causes are addressed
here within one instrumented implementation:

1. **Data leakage** inflates reported results and is usually invisible in a
   published method, because the order of fitting relative to partitioning is
   rarely stated.
2. **Models are evaluated only on the dataset they were trained on**, although
   deployment necessarily involves a different network.
3. **The engineering cost of serving a model is rarely measured** — published
   inference times exclude the preprocessing that deployment requires.

The resulting negative findings are reported as results, not minimised.

| Headline finding | Evidence |
|---|---|
| Cross-dataset transfer collapses uniformly. A random forest scoring macro-F1 **0.9345** within CIC-IDS2017 drops to **0.3109** on UNSW-NB15, detecting **33 of 45,332** attacks at perfect precision — a recall collapse, not degraded discrimination. | `results/cross_dataset/metrics.json` |
| The best within-dataset model degrades the most under transfer. | `results/cross_dataset/metrics.json` |
| End-to-end detection latency exceeds model inference time by roughly **three orders of magnitude** (15.88 ms vs 0.007677 ms per record): preprocessing dominates, not the classifier. | `results/*/metrics.json`, `results/replay/*.json` |
| Freedom from data leakage is evidenced by six inspectable properties of the fitted artefacts, not asserted. | `VALIDATION_REPORT.md` §4 |

---

## Features

- One leakage-safe preprocessing pipeline applied identically to three
  structurally dissimilar datasets, with no dataset-specific transformation
  logic.
- Four model families trained under an identical selection protocol, with the
  held-out partition evaluated exactly once.
- A cross-dataset transfer experiment over a documented twelve-feature common
  flow schema.
- A stateless REST prediction service that reuses the *same* fitted artefacts
  produced during training — training/serving equivalence verified to bit-level
  equality.
- A monitoring dashboard driven entirely by real generated artefacts.
- Streamed **replay-based** detection with graded threat levels, threshold
  alerting, and measured per-record latency and throughput.
- 48 automated tests, including explicit regressions for the leakage controls
  and every recognised failure path.

### Scope and exclusions

Stated plainly, because they bear on how the results should be read:

- **No cloud deployment.** Developed and evaluated entirely on a single local
  machine. The prediction service is stateless and would *facilitate*
  containerised deployment, but none was performed.
- **No capture of traffic from an operational network.** Every record processed
  originates from a published benchmark dataset file. The streamed mode is
  **replay of historical network-flow records**, labelled *Replay / Simulation*
  throughout, with synthetic timestamps anchored to a fixed past date so they
  cannot be mistaken for capture times.
- **No packet-level feature extraction** — the pipeline consumes pre-computed
  flow and connection records.
- **Binary classification only.** Multi-class attack-category labels are derived
  and retained in the artefacts, but no multi-class model was trained.
- **The LSTM covers NSL-KDD only**, over file-order windows rather than genuine
  temporal sequences.
- **No security hardening**: no authentication, transport encryption or rate
  limiting. **No alert delivery** to email, syslog or a SIEM.

---

## System Architecture

Four layers, connected only through persisted artefacts:

```
Data layer         NSL-KDD · CIC-IDS2017 · UNSW-NB15
                         ↓  src/data_loaders.py  (one bundle contract)
Training pipeline  clean → split → fit transformers (TRAIN only)
                         → feature selection → model selection → evaluation
                         ↓
Persisted          artifacts/<ds>/preprocessor.joblib
artefacts          models/<ds>/*.joblib | lstm.pt
                   results/<ds>/*.json
                         ↓
Serving layer      Prediction API · Replay engine · Monitoring dashboard
                   (load artefacts; never refit)
```

The serving layer consumes **exactly** the fitted preprocessor produced during
training, eliminating training–serving skew by construction. Verified by
`python -m src.inference`.

---

## Datasets

| Dataset | Records | Features | Partitioning |
|---|--:|--:|---|
| NSL-KDD | 125,973 train / 22,544 test | 41 (3 categorical) | Published `KDDTrain+` / `KDDTest+`; the test file contains **17 attack types absent from training** |
| CIC-IDS2017 | 2,830,743 total | 77 (0 categorical) | No official split; seeded stratified 85/15 of a sub-sample |
| UNSW-NB15 | 175,341 train / 82,332 test | 42 (3 categorical) | Official partitions respected (≈42/15/43, not 70/15/15) |

**CIC-IDS2017 is sub-sampled to 200,000 records** (~7.1 % of the corpus) — a
memory constraint of the 3.8 GiB development machine, not a design preference,
and reproducible from the recorded seed.

Datasets are not redistributed here; `scripts/download_datasets.py` fetches them
and asserts each row count against the published value.

---

## Machine Learning Pipeline

Applied identically to all three datasets:

1. **Clean** — drop identifiers, resolve duplicate column names, normalise
   non-finite values to a missing marker.
2. **Deduplicate** — exact duplicate rows removed from the **training pool
   only**; evaluation partitions are never modified.
3. **Partition** — train / validation / held-out test, stratified on the coarse
   attack category.
4. **Fit on TRAIN only** — median/mode imputation → standardisation and one-hot
   encoding (unseen categories ignored) → zero-variance filter.
5. **Select features** — top 20 by mutual information, computed on the training
   partition alone.
6. **Balance** — `class_weight='balanced'`. SMOTE is implemented and its
   balanced arrays generated, but **was not used to produce any reported
   result**.

### Leakage prevention

All fitted transformations live in a single composite object, fitted exactly
once on the training partition and thereafter only applied. Six properties are
verified, each of which leakage would have altered — most directly, the fitted
scaler's recorded sample count equals the training partition size **exactly**
(103,695 for NSL-KDD), a value no fitting that included evaluation data could
produce. All six are enforced by automated regression tests.

---

## Models

| Model | Datasets | Notes |
|---|---|---|
| Logistic Regression | all three | Interpretable linear baseline |
| Random Forest | all three | Strongest on every dataset |
| SVM (RBF kernel) | all three | Trained on a 20,000-record subsample; **no calibrated probability** |
| LSTM (PyTorch) | **NSL-KDD only** | Windows of 10 consecutive records in **file order** — NSL-KDD has no timestamps, so this is a context window over tabular records, **not temporal modelling** |

Hyperparameters chosen by exhaustive grid search on the validation partition;
the held-out partition evaluated exactly once, by a separate process; no refit
on train + validation; all randomness seeded.

---

## Experimental Evaluation

Held-out test results (macro-F1 is the primary metric):

| Dataset | Model | Accuracy | Precision | Recall | Macro-F1 | ROC-AUC |
|---|---|--:|--:|--:|--:|--:|
| NSL-KDD | Random Forest | 0.7706 | 0.9696 | 0.6164 | **0.7695** | 0.9498 |
| NSL-KDD | LSTM | 0.7573 | 0.9150 | 0.6323 | 0.7569 | 0.8846 |
| CIC-IDS2017 | Random Forest | 0.9978 | 0.9962 | 0.9975 | **0.9976** | 0.9998 |
| UNSW-NB15 | Random Forest | 0.8608 | 0.8098 | 0.9766 | **0.8541** | 0.9769 |

All ten model/dataset combinations are in `results/<dataset>/metrics.json`.

### Cross-dataset generalisation

Both datasets projected onto a twelve-feature common flow schema, so within- and
cross-dataset figures are directly comparable. Macro-F1:

| Model | Within CIC | CIC → UNSW | Gap |
|---|--:|--:|--:|
| Logistic Regression | 0.6575 | 0.2946 | +0.3630 |
| Random Forest | 0.9345 | **0.3109** | **+0.6236** |
| SVM (RBF) | 0.8206 | 0.3325 | +0.4881 |

The random forest trained on CIC-IDS2017 and applied to UNSW-NB15 achieves
attack precision **1.0000** and attack recall **0.000728** — 33 of 45,332
attacks detected, zero false positives. It has effectively become a constant
predictor of the benign class.

---

## Replay-Based Detection

> **This is streamed replay of historical network-flow records, not live
> network traffic.** Timestamps are synthetic (a fixed 2020 anchor plus a
> constant per-record interval), the mode is labelled *Replay / Simulation* in
> every interface, and every saved result records
> `"is_live_network_traffic": false`.

Records are read sequentially in file order through the complete serving path:
validation → the production preprocessor → feature selection → model →
prediction → detection event → alert.

**35,000 records across five configurations, 0 errors.**

| Dataset | Model | Records | Agreement | End-to-end (ms) | Throughput (rec/s) |
|---|---|--:|--:|--:|--:|
| NSL-KDD | Random Forest | 12,000 | 0.7718 | 15.8815 | 62.6 |
| NSL-KDD | Logistic Regression | 4,000 | 0.7185 | 15.0105 | 66.2 |
| NSL-KDD | SVM (RBF) | 4,000 | 0.7278 | 17.8571 | 55.7 |
| CIC-IDS2017 | Random Forest | 3,000 | 0.9990 | 33.8377 | 29.4 |
| UNSW-NB15 | Random Forest | 12,000 | 0.9732 | 19.4547 | 51.1 |

Alerts require both an attack prediction and a confidence reaching a
configurable threshold (default 0.90), with graded threat levels. Because the
SVM exposes no calibrated probability, **every** one of its attack predictions
becomes an alert with unknown severity — a training-time configuration decision
invisible in offline metrics that disables triage once the model is served.

Alerts are surfaced in the dashboard, the CLI and the saved results only.

---

## REST API

Starlette + Uvicorn, four endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Service description and expected feature columns |
| `/health` | GET | Readiness; 200 when a model is loadable, 503 otherwise |
| `/models` | GET | Available dataset/model pairs with their exact ordered feature contract |
| `/predict` | POST | Prediction for one record (named object or ordered array) |

Predictions are served from the same fitted artefacts used in training. Every
response carries the artefacts that produced it, the confidence basis, any input
coercion warnings and a correlation identifier. Errors are typed (400, 404, 405,
422, 500) with no disclosure of internal detail.

Reference with real captured responses: [`docs/API.md`](docs/API.md).
Ready-to-use payloads: [`examples/`](examples/).

---

## Dashboard

Streamlit, five views over the project's own generated artefacts: system status,
dataset and model information, detection replay, model performance with
confusion matrices, and model comparison with latency.

Every displayed value originates from a real artefact; where one is missing the
view reports its absence rather than rendering a placeholder.

---

## Testing and Validation

| Command | Outcome |
|---|---|
| `pytest tests/ -q` | **48 passed** |
| `python scripts/test_api.py` | **24/24 checks** against a live server subprocess |
| `python scripts/test_dashboard.py` | **36/36 checks** including render and headless-serve |
| `python -m src.inference` | **PASS** — training/inference parity bit-exact, max \|Δ\| = 0.0 |

The suite includes explicit regression tests for the leakage controls — a change
that fitted the scaler on more than the training partition fails an assertion —
and covers malformed input, missing features, non-finite values, unavailable
model and preprocessing failure.

Detailed reports: [`VALIDATION_REPORT.md`](VALIDATION_REPORT.md),
[`REPRODUCIBILITY_REPORT.md`](REPRODUCIBILITY_REPORT.md),
[`REALTIME_VALIDATION.md`](REALTIME_VALIDATION.md).

---

## Installation

Requires Python 3.12 (developed on 3.12.3, Linux).

```bash
git clone <REPOSITORY-URL>
cd <project-directory>

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-lock.txt
```

`requirements-lock.txt` pins all 69 packages to exact versions and is the
reproduction source. A clean-environment rebuild from it reproduced all four
models' held-out metrics exactly.

---

## Usage

```bash
# 1. Acquire datasets (row counts asserted against published values)
python scripts/download_datasets.py

# 2. Prepare a dataset: clean, partition, fit, select features
python scripts/prepare_datasets.py --dataset nsl_kdd

# 3. Train with validation-driven model selection
python scripts/train_models.py --dataset nsl_kdd

# 4. Evaluate once on the held-out partition
python scripts/evaluate_models.py --dataset nsl_kdd

# 5. Cross-dataset transfer experiment
python scripts/cross_dataset_experiment.py

# 6. Replay-based streamed detection (NOT live traffic)
python scripts/run_replay.py --dataset nsl_kdd --model random_forest

# Serving
python scripts/serve_api.py        # prediction API
python scripts/run_dashboard.py    # monitoring dashboard

# Verification
pytest tests/ -q
python -m src.inference            # training/inference parity
```

For CIC-IDS2017 and UNSW-NB15 substitute `--dataset cicids2017` or
`--dataset unsw_nb15`, and use `scripts/train_eval_datasets.py`.

---

## Project Structure

```
src/
  data_loaders.py          three dataset adapters → one bundle contract
  preprocessing.py         leakage-safe pipeline and feature selection
  models.py                model construction and validation-driven selection
  evaluation.py            metrics, latency measurement, plots
  inference.py             artefact loading; training/inference parity check
  api.py                   prediction service
  monitoring_dashboard.py  five-view dashboard
  replay_detection.py      streamed replay detection, events, alerting
scripts/                   preparation, training, evaluation, replay, harnesses
tests/                     48 automated tests
configs/                   hyperparameter grids and methodology notes
artifacts/<dataset>/       fitted preprocessor and metadata sidecars
models/<dataset>/          trained models and selection records
results/<dataset>/         metrics, dataset statistics, plots
results/cross_dataset/     transfer experiment
results/replay/            streamed detection results
docs/, examples/           API reference and request payloads
dissertation/              MSc project report and its build system
```

Some modules under `src/` are inherited from the open-source project this work
builds on (see [License](#license)) and are **not used** by the current
pipeline. They are retained unmodified so the boundary between inherited and
contributed code stays verifiable. The active modules are those listed above.

---

## Limitations

- No cloud deployment; no measurement under network transit or horizontal
  scaling.
- Detection demonstrated on replayed records only; latency figures exclude
  capture, flow assembly, feature derivation, burst behaviour and backpressure.
- Packet-to-flow feature extraction is not implemented.
- CIC-IDS2017 results rest on a 200,000-record sub-sample.
- The LSTM covers NSL-KDD only, over file-order windows.
- Held-out partitions retain duplicate records (26,387 in UNSW-NB15) because
  deduplication is confined to training pools by policy.
- Throughput (29–66 records/s on one core) is one to two orders of magnitude
  below the flow rates of a busy enterprise link.
- Alerts have no external delivery channel; the API is not hardened; model
  artefacts are deserialised without integrity verification.
- Each configuration was evaluated on a single seed, without confidence
  intervals.

---

## Future Work

- **Domain adaptation** to address the cross-dataset collapse — importance
  weighting, adversarial adaptation, or fine-tuning on a small labelled sample
  from the target environment.
- **Multiple seeds and confidence intervals**, so small differences between
  models can be interpreted.
- **Optimising the transformation path**, which the latency measurement
  identifies as the real cost — not the classifier.
- **Multi-class attack-category classification** (labels are already derived).
- **Containerised deployment**, with model artefact size addressed first (the
  UNSW-NB15 forest is 141 MB).
- **Packet-to-flow feature extraction**, which would require authorisation from
  a network owner before any traffic was observed.
- **Alert delivery integration**, probability calibration for the SVM, and
  security hardening.

---

## License

Released under the MIT Licence — see [`LICENSE`](LICENSE).

This project builds on an existing MIT-licensed open-source project, whose
copyright notice is retained in `LICENSE` as the licence requires. The
inherited modules under `src/` are kept **byte-for-byte unmodified** so the
boundary between inherited and contributed work is verifiable:

```bash
git diff --stat 624f925 -- src/data_preprocessing.py src/feature_selection.py \
    src/model_training.py src/explainability_realtime.py src/dashboard.py \
    src/realtime_detection.py src/shap_explainability.py src/config.py \
    src/utils.py src/__init__.py
# empty output = unmodified since the inherited baseline
```

Provenance is documented in [`DEVELOPMENT_NOTES.md`](DEVELOPMENT_NOTES.md) and
in the project report.

### Datasets

NSL-KDD, CIC-IDS2017 and UNSW-NB15 are used under their published academic
terms and are not redistributed here. They are cited to their originating
publications in the project report.
