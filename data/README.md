# Data Directory

Datasets are **not** committed to this repository. They are downloaded on
demand by `scripts/download_datasets.py`, which asserts each corpus's row count
against its published value before use.

```
data/
├── raw/                 downloaded corpora (git-ignored)
│   ├── nsl_kdd/         KDDTrain+.txt, KDDTest+.txt
│   ├── cicids2017/      MachineLearningCVE daily flow CSVs
│   └── unsw_nb15/       UNSW_NB15_training-set.csv, _testing-set.csv
├── processed/           prepared arrays and label maps (git-ignored)
└── replay_samples/      small committed fixture for the replay demonstration
```

`replay_samples/` holds a 3,000-record CIC-IDS2017 fixture retaining every
coarse attack class, committed so the streamed-replay demonstration is
reproducible without a download.

Datasets are used under their published academic terms and are not
redistributed here. See the project [README](../README.md).
