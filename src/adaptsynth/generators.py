"""Synthetic-data generators behind one interface.

    syn, stats = fit_and_sample("ctabgan_plus", train, cols, categorical, seed=42, epochs=200)

`cols` is the list of columns to train on and MUST end with "label". `syn` has the same columns and the
same number of rows as `train`. `stats` = {"train_seconds", "peak_gpu_mem_gb"}.

ADDING A NEW MODEL (e.g. TabDDPM or GVAE): write a class with `fit(data, categorical, seed, epochs)` and
`sample(n)`, decorate it with @register("key", "Display name") and it shows up in the notebook.
"""
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import skew

REGISTRY, DISPLAY = {}, {}
CTABGAN_DIR = Path(__file__).resolve().parents[2] / "third_party" / "CTAB-GAN-Plus"


def register(key, display):
    def deco(cls):
        REGISTRY[key], DISPLAY[key] = cls, display
        return cls
    return deco


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------- SDV: CTGAN, TVAE
class _SDVGenerator:
    synthesizer_name = None

    def fit(self, data, categorical, seed, epochs):
        from sdv.metadata import Metadata
        import sdv.single_table as st
        meta = Metadata.detect_from_dataframes({"flows": data})
        for c in data.columns:
            meta.update_column(table_name="flows", column_name=c,
                               sdtype="categorical" if c in categorical else "numerical")
        seed_everything(seed)
        self.model = getattr(st, self.synthesizer_name)(meta, epochs=epochs, batch_size=500, verbose=False)
        self.model.fit({"flows": data})
        return self

    def sample(self, n):
        return self.model.sample(n)["flows"]


@register("ctgan", "CTGAN")
class CTGAN(_SDVGenerator):
    synthesizer_name = "CTGANSynthesizer"


@register("tvae", "TVAE")
class TVAE(_SDVGenerator):
    synthesizer_name = "TVAESynthesizer"


# ----------------------------------------------------------------------------- CTAB-GAN+
@register("ctabgan_plus", "CTAB-GAN+")
class CTABGANPlus:
    """Official CTAB-GAN+ code (third_party/CTAB-GAN-Plus). We drive its DataPrep + synthesizer directly
    because its `CTABGAN` wrapper reads a CSV and does its own hidden 80/20 split, which would train on
    less data than the other generators.

    Column types are detected from the training data, following the CTAB-GAN+ paper:
      * categorical  - the columns we are told are categorical (incl. the label)
      * log          - heavy-tailed numerics (skew > 3) with min <= 0   -> log transform
      * mixed        - numerics where one value (e.g. 0) covers >= 20% of rows -> point mass + continuous part
      * integer      - numerics whose training values are all whole numbers -> rounded after sampling
    The label is passed as the target so CTAB-GAN+'s auxiliary classifier ("+") is used.
    """

    def fit(self, data, categorical, seed, epochs, batch_size=500):
        if str(CTABGAN_DIR) not in sys.path:
            sys.path.insert(0, str(CTABGAN_DIR))
        from model.pipeline.data_preparation import DataPrep
        from model.synthesizer.ctabgan_synthesizer import CTABGANSynthesizer

        data = data.copy()
        num = [c for c in data.columns if c not in categorical]
        self.log_cols = [c for c in num if data[c].min() <= 0 and skew(data[c]) > 3]
        mixed = {}
        for c in num:
            top_value, share = data[c].value_counts().index[0], data[c].value_counts(normalize=True).iloc[0]
            if share >= 0.2:
                mixed[c] = [float(top_value)]
        integer = [c for c in num if (data[c] % 1 == 0).all()]
        seed_everything(seed)
        # {"": None} = "no problem type": DataPrep then skips its own train/test split
        self.prep = DataPrep(data, list(categorical), self.log_cols, mixed, [], [], integer, {"": None}, 0.0)
        self.model = CTABGANSynthesizer(epochs=epochs, batch_size=batch_size)
        self.model.fit(train_data=self.prep.df, categorical=self.prep.column_types["categorical"],
                       mixed=self.prep.column_types["mixed"], general=self.prep.column_types["general"],
                       non_categorical=self.prep.column_types["non_categorical"],
                       type={"Classification": "label"})
        return self

    def sample(self, n):
        return self.prep.inverse_prep(self.model.sample(n))


# ----------------------------------------------------------------------------- SMOTE
@register("smote", "SMOTE")
class SMOTE:
    """Interpolation baseline. To get a purely synthetic table (like the other models) we ask SMOTE to
    create as many NEW rows per class as the class has real rows, then keep only the new rows. The result
    has the same size and class proportions as the training table. Uses SMOTENC when categoricals exist."""

    def fit(self, data, categorical, seed, epochs):
        self.data, self.categorical, self.seed = data.copy(), list(categorical), seed
        return self

    def sample(self, n):
        from imblearn.over_sampling import SMOTE as _SMOTE, SMOTEN, SMOTENC
        X, y = self.data.drop(columns="label"), self.data["label"]
        counts = y.value_counts()
        k = int(max(1, min(5, counts.min() - 1)))
        strategy = {c: int(2 * m) for c, m in counts.items()}          # existing + as many new rows
        cat_idx = [i for i, c in enumerate(X.columns) if c in self.categorical]
        if len(cat_idx) == X.shape[1]:                                   # all categorical (e.g. GUIDE): SMOTEN
            sm = SMOTEN(sampling_strategy=strategy, k_neighbors=k, random_state=self.seed)
        elif cat_idx:
            sm = SMOTENC(categorical_features=cat_idx, sampling_strategy=strategy, k_neighbors=k,
                         random_state=self.seed)
        else:
            sm = _SMOTE(sampling_strategy=strategy, k_neighbors=k, random_state=self.seed)
        X_res, y_res = sm.fit_resample(X, y)
        new = X_res.iloc[len(X):].copy()                                  # appended rows = synthetic
        new["label"] = y_res.iloc[len(X):].to_numpy()
        for c in X.columns:                                               # keep whole-number columns whole
            if c not in self.categorical and (self.data[c] % 1 == 0).all():
                new[c] = new[c].round()
        return new.sample(n=n, replace=len(new) < n, random_state=self.seed).reset_index(drop=True)[self.data.columns]


# ----------------------------------------------------------------------------- entry point
def fit_and_sample(key, train, cols, categorical, seed=42, epochs=200, n_rows=None):
    if key not in REGISTRY:
        raise KeyError(f"Unknown generator {key!r}. Registered: {sorted(REGISTRY)}")
    assert cols[-1] == "label", "label must be the last column"
    data = train[cols].reset_index(drop=True)
    gen = REGISTRY[key]()
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    gen.fit(data, [c for c in categorical if c in cols], seed, epochs)
    if cuda:
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0                                   # fit time only (sampling excluded)
    syn = gen.sample(n_rows or len(data))
    syn = syn[cols].copy()
    for c in cols:                                                       # same dtypes as the real table
        if c in categorical:
            syn[c] = syn[c].astype(str)
        else:
            syn[c] = pd.to_numeric(syn[c], errors="coerce")
    syn = syn.fillna(data.median(numeric_only=True)).reset_index(drop=True)
    return syn, {"train_seconds": seconds,
                 "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if cuda else np.nan}
