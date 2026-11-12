"""Append experiment results to one shared log (paper_results/experiment_log.csv).

Every notebook run adds rows here instead of overwriting, so nothing is lost between
runs. build_results_workbook.py turns this log into the Excel workbook for the paper.

Usage from a notebook:
    from results_log import log_comparison, log_result
    log_comparison(results, dataset="CIC-IDS2017", generator="CTGAN", epochs=EPOCHS)
    log_result(dataset="SDN Anomaly", generator="GVAE", seed=42, tstr_f1_macro=0.86, tstr_roc_auc=0.91)
"""
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "paper_results" / "experiment_log.csv"

# One row = one generator fit (one dataset, method, feature set, seed).
COLUMNS = [
    "timestamp", "dataset", "generator", "feature_set", "ablation", "config_tag", "seed",
    "n_features", "epochs", "classifier",
    # Utility (paper Table II)
    "tstr_f1_macro", "tstr_roc_auc", "trtr_f1_macro", "trtr_roc_auc",
    # Fidelity (paper Table III + sdmetrics scores the notebooks already compute)
    "dist_mse", "dist_similarity", "sim_numerical", "sim_categorical", "sim_relationships",
    # Privacy (paper Table V)
    "mia_advantage_pct",
    # Cost
    "train_seconds", "peak_gpu_mem_gb",
    "source", "notes",
]

# Rows sharing these values are re-runs of the same experiment; only the latest one is used.
RUN_KEY = ["dataset", "generator", "feature_set", "ablation", "config_tag", "classifier", "seed"]


def log_result(**fields):
    """Append one row. Unknown field names raise so typos don't silently vanish."""
    unknown = set(fields) - set(COLUMNS)
    if unknown:
        raise ValueError(f"Unknown result fields: {sorted(unknown)}. Allowed: {COLUMNS}")
    fields.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))
    _append(pd.DataFrame([fields]))


def log_comparison(results, dataset, generator="CTGAN", classifier="RandomForest",
                   epochs=None, trtr_f1=None, config_tag="", source="", notes=""):
    """Log the `results` table produced by the feature-selection notebooks
    (columns: experiment, seed, columns_trained_on, numerical_cols, categorical_cols,
    relationships, TSTR_macroF1, train_seconds)."""
    ts = datetime.now().isoformat(timespec="seconds")
    rows = pd.DataFrame({
        "timestamp": ts,
        "dataset": dataset,
        "generator": generator,
        "feature_set": results["experiment"].str.extract(r"^(\w+)")[0]
                       .map({"baseline": "all features", "selected": "voted top-N"})
                       .fillna(results["experiment"]),
        "ablation": "",
        "config_tag": config_tag,
        "seed": results["seed"],
        "n_features": results["columns_trained_on"] - 1,  # minus the label column
        "epochs": epochs,
        "classifier": classifier,
        "tstr_f1_macro": results["TSTR_macroF1"],
        "trtr_f1_macro": trtr_f1,
        "sim_numerical": results["numerical_cols"],
        "sim_categorical": results["categorical_cols"],
        "sim_relationships": results["relationships"],
        "train_seconds": results["train_seconds"],
        "source": source,
        "notes": notes,
    })
    _append(rows)
    print(f"Logged {len(rows)} row(s) for {dataset} to {LOG_PATH.relative_to(ROOT)}")


def load_log():
    if not LOG_PATH.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(LOG_PATH, keep_default_na=False, na_values=[""])
    for c in ["feature_set", "ablation", "config_tag", "source", "notes"]:
        df[c] = df[c].fillna("")
    return df


def _append(rows):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = rows.reindex(columns=COLUMNS)
    for c in ["feature_set", "ablation", "config_tag", "source", "notes"]:
        rows[c] = rows[c].fillna("")
    rows.to_csv(LOG_PATH, mode="a", header=not LOG_PATH.exists(), index=False)
