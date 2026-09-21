# ecg-ml

ECG heartbeat classification experiments.

## Project structure

```
ecg-ml/
├── data/            # Datasets (not tracked by git)
│   └── heartbeat/   # Kaggle "shayanfazeli/heartbeat" dataset
├── notebooks/       # Jupyter notebooks
└── requirements.txt
```

## Data

Datasets live in `data/`, which is ignored by git (only `data/.gitkeep` is committed).

The [Heartbeat dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat) is downloaded with `kagglehub` from `notebooks/test.ipynb` into `data/heartbeat/`:

- `mitbih_train.csv`, `mitbih_test.csv` — MIT-BIH Arrhythmia
- `ptbdb_normal.csv`, `ptbdb_abnormal.csv` — PTB Diagnostic ECG

## Setup

Create and activate a virtual environment (`.venv/` is ignored by git):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

When adding new packages, update the requirements file:

```bash
pip freeze > requirements.txt
```
