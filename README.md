# AdaptSynth

Experiments for *AdaptSynth: A Framework for Structural-Aware Synthetic Data Generation in Cybersecurity*.

The pipeline: split → data preparation → feature ranking with five techniques and voting → train synthetic-data generators on all vs. voted features → compare synthetic with real data (distribution/relationship similarity, TSTR/TRTR with XGBoost).

## Main notebook

[adaptsynth_pipeline.ipynb](adaptsynth_pipeline.ipynb) runs any dataset with any generator. Set `DATASET` and `GENERATORS` in the Settings cell.

| `DATASET` | Data |
|---|---|
| `iot23` | IoT-23 Scenario 1 (CTU-IoT-Malware-1-1, Mirai) |
| `sdn` | SDN DDoS dataset (chronological split) |
| `guide` | Microsoft GUIDE (all-categorical, 40k-row sample) |
| `cic_ids2017` | CIC-IDS2017 (sampled per class) |

| `GENERATORS` | Model |
|---|---|
| `ctabgan_plus` | CTAB-GAN+ (official code) |
| `ctgan`, `tvae` | SDV CTGAN / TVAE |
| `smote` | SMOTE / SMOTENC / SMOTEN |

New models are added by registering a class in [src/adaptsynth/generators.py](src/adaptsynth/generators.py) (see the last section of the notebook).

The older notebooks ([iot23_scenario1.ipynb](iot23_scenario1.ipynb), [machinelearning_pipeline.ipynb](machinelearning_pipeline.ipynb)) are the earlier CTGAN-only experiments.

## Code layout

* `src/adaptsynth/` – datasets, prep, feature voting, generators, evaluation
* `src/results_log.py`, `src/build_results_workbook.py` – append runs to `paper_results/experiment_log.csv` and build `paper_results/AdaptSynth_results.xlsx` (paper tables next to measured numbers)
* `results*/` – outputs of earlier runs

## Setup

```bash
python3.11 -m venv .venv-server && source .venv-server/bin/activate
pip install -r requirements.txt            # PyTorch build: see https://pytorch.org for your CUDA version
python -m ipykernel install --user --name adaptsynth-server --display-name "AdaptSynth Server (Python 3.11)"

# CTAB-GAN+ (official code) plus our one-line NumPy 2 fix
git clone https://github.com/Team-TUD/CTAB-GAN-Plus third_party/CTAB-GAN-Plus
git -C third_party/CTAB-GAN-Plus apply ../ctabgan_plus_numpy2.patch
```

Datasets are not in the repo. `data/README.md` lists where each one comes from and the Kaggle links; most can be fetched with `kagglehub`.
