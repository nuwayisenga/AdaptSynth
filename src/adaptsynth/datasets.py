"""Dataset loaders. Each loader returns a `Dataset` whose target column is called `label`.

Loaders only do dataset-specific cleaning (drop identifiers, pick columns, sample rows).
Everything generic (split, imputation, correlation filter, ...) lives in prep.py so all
datasets go through exactly the same steps.

To add a dataset: write `load_xxx()` returning a Dataset and add it to LOADERS.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


@dataclass
class Dataset:
    key: str                       # short id used in the notebook, e.g. "iot23"
    name: str                      # name used in the paper / results workbook
    df: pd.DataFrame               # RangeIndex, includes the `label` column
    categorical: list              # categorical columns, INCLUDING "label"
    n_select: int                  # default number of features to keep after voting
    split: str = "stratified"      # "stratified" | "time" | "group"
    split_key: pd.Series = None    # time stamp / group id per row; never used as a feature
    notes: str = ""
    info: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- IoT-23
def load_iot23():
    """CTU-IoT-Malware-Capture-1-1 (Scenario 1, Mirai). Zeek conn.log, tab separated."""
    path = DATA / "CTU-IoT-Malware-1-1.conn.log.labeled"
    with open(path) as f:
        for line in f:
            if line.startswith("#fields"):
                columns = line.rstrip("\n").split("\t")[1:]
                break
    raw = pd.read_csv(path, sep="\t", comment="#", header=None, names=columns, na_values=["-", "(empty)"])
    # identifiers/artifacts, and the finer-grained label (would leak the target)
    drop = [c for c in raw.columns if c in ("ts", "uid", "id.orig_h", "id.resp_h") or "detailed" in c]
    df = raw.drop(columns=drop)
    categorical = [c for c in ["proto", "service", "conn_state", "history", "label"] if c in df]
    numerical = [c for c in df.columns if c not in categorical]
    df[numerical] = df[numerical].apply(pd.to_numeric, errors="coerce").fillna(0)   # "no reply" -> 0
    df[categorical] = df[categorical].fillna("none").astype(str)
    return Dataset("iot23", "IoT-23", df.reset_index(drop=True), categorical, n_select=6,
                   notes="CTU-IoT-Malware-1-1 (Scenario 1). Random stratified split.")


# ----------------------------------------------------------------------------- CIC-IDS2017
def load_cic_ids2017(max_rows_per_class=2000, seed=42, chunk_size=100_000):
    """CIC-IDS2017 MachineLearningCVE CSVs. Samples up to `max_rows_per_class` rows per attack
    label (None = every row, needs a lot of RAM). This changes class proportions."""
    files = sorted((DATA / "MachineLearningCVE").glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSVs in {DATA / 'MachineLearningCVE'}")
    rng = np.random.default_rng(seed)
    reservoirs, source_counts, all_chunks = {}, {}, []
    for path in files:
        for chunk in pd.read_csv(path, encoding="cp1252", chunksize=chunk_size, low_memory=False):
            chunk.columns = chunk.columns.str.strip()
            chunk = chunk.rename(columns={"Label": "label"}).dropna(subset=["label"])
            chunk["label"] = chunk["label"].astype(str).str.strip()
            for label, group in chunk.groupby("label", sort=False):
                source_counts[label] = source_counts.get(label, 0) + len(group)
                if max_rows_per_class is None:
                    all_chunks.append(group)
                else:
                    group = group.copy()
                    group["_priority"] = rng.random(len(group))
                    prev = reservoirs.get(label)
                    pool = pd.concat([prev, group], ignore_index=True) if prev is not None else group
                    reservoirs[label] = pool.nsmallest(max_rows_per_class, "_priority")
    raw = (pd.concat(all_chunks, ignore_index=True) if max_rows_per_class is None else
           pd.concat(reservoirs.values(), ignore_index=True).drop(columns="_priority"))
    df = raw.drop(columns=[c for c in ["Flow ID", "Source IP", "Destination IP", "Timestamp"] if c in raw])
    for c in df.columns.drop("label"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).drop_duplicates().reset_index(drop=True)
    return Dataset("cic_ids2017", "CIC-IDS2017", df, ["Destination Port", "label"], n_select=10,
                   notes=f"Up to {max_rows_per_class} rows per attack class (class proportions are NOT the real ones).",
                   info={"source_counts": source_counts, "max_rows_per_class": max_rows_per_class})


# ----------------------------------------------------------------------------- SDN
def load_sdn():
    """DDoS SDN dataset (Ahuja et al.). 104k rows. Split chronologically on `dt`: a random split puts
    near-copies of the same flow (counters that grow over time) on both sides and inflates scores."""
    df = pd.read_csv(DATA / "sdn_anomaly" / "dataset_sdn.csv")
    key = df["dt"].copy()                                   # time index: used for the split only
    df = df.drop(columns=["dt"])
    df["label"] = df["label"].map({0: "Benign", 1: "DDoS"})
    categorical = ["src", "dst", "Protocol", "label"]
    df[categorical] = df[categorical].astype(str)
    return Dataset("sdn", "SDN Anomaly", df, categorical, n_select=6, split="time", split_key=key,
                   notes="Chronological 70/30 split on `dt`. src/dst IPs kept as categorical features "
                         "(they are also the nodes for a graph model such as GVAE).")


# ----------------------------------------------------------------------------- Microsoft GUIDE
GUIDE_COLUMNS = ["OrgId", "DetectorId", "AlertTitle", "Category", "EntityType", "EvidenceRole",
                 "OSFamily", "OSVersion", "CountryCode", "State", "City", "ApplicationName"]


def load_guide(n_rows=40_000, max_categories=30, seed=42):
    """Microsoft GUIDE (security incident triage). The full train file has 9.5M rows, so we draw a
    random sample (cached on disk), keep low-cardinality alert attributes, and collapse each column's
    rare values into "other" so generators can learn them. Split is grouped by IncidentId."""
    folder = DATA / "microsoft_guide"
    cache = folder / f"guide_sample_{n_rows}_seed{seed}.csv"
    if not cache.exists():
        rng = np.random.default_rng(seed)
        frac = min(1.0, n_rows * 1.2 / 9_516_837)
        parts = [c.sample(frac=frac, random_state=int(rng.integers(1_000_000_000)))
                 for c in pd.read_csv(folder / "GUIDE_Train.csv", usecols=GUIDE_COLUMNS + ["IncidentGrade", "IncidentId"],
                                      chunksize=500_000, dtype=str)]
        s = pd.concat(parts, ignore_index=True).dropna(subset=["IncidentGrade"])
        s.sample(n=min(n_rows, len(s)), random_state=seed).to_csv(cache, index=False)
    s = pd.read_csv(cache, dtype=str).reset_index(drop=True)
    key = s["IncidentId"].copy()
    df = s[GUIDE_COLUMNS].fillna("missing")
    for c in GUIDE_COLUMNS:                                  # keep the most frequent values only
        top = df[c].value_counts().index[:max_categories]
        df[c] = df[c].where(df[c].isin(top), "other")
    df["label"] = s["IncidentGrade"]
    return Dataset("guide", "Microsoft GUIDE", df, list(df.columns), n_select=6, split="group", split_key=key,
                   notes=f"{len(df)}-row random sample of 9.5M alerts; every column is categorical (top "
                         f"{max_categories} values + 'other'); split grouped by IncidentId.")


LOADERS = {"iot23": load_iot23, "cic_ids2017": load_cic_ids2017, "sdn": load_sdn, "guide": load_guide}


def load(key, **kwargs):
    if key not in LOADERS:
        raise KeyError(f"Unknown dataset {key!r}. Choose from {sorted(LOADERS)}")
    return LOADERS[key](**kwargs)
