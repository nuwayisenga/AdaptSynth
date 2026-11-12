"""Split + data preparation, following Section IV-B of the paper.

Every statistic (medians, correlations, IQR fences, scaling) is learned on TRAIN only and then
applied to test, so no test information leaks into the pipeline.

Paper step                                         -> what this module does
(1) drop duplicated / >0.95-correlated features    -> drop_duplicate_columns + correlation filter (always on)
(2) one-hot (nominal) / label (ordinal) encoding   -> NOT done here: CTGAN/TVAE/CTAB-GAN+ model categoricals
                                                      natively and the classifier ordinal-encodes them
(3) IQR outlier removal, then z-score scaling      -> optional (remove_outliers / zscore), OFF by default
(4) median / mode imputation                       -> always on
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class Prepared:
    train: pd.DataFrame
    test: pd.DataFrame
    features: list        # candidate features (no label)
    numerical: list       # numeric features
    categorical: list     # categorical columns INCLUDING "label"
    info: dict            # what was dropped / removed, for the report


def split(ds, test_size=0.3, seed=42):
    """70/30 split. Mode comes from the dataset: stratified random, chronological, or grouped."""
    df = ds.df
    if ds.split == "time":
        order = ds.split_key.sort_values(kind="stable").index
        n_train = int(len(df) * (1 - test_size))
        tr_idx, te_idx = order[:n_train], order[n_train:]
    elif ds.split == "group":
        tr_i, te_i = next(GroupShuffleSplit(1, test_size=test_size, random_state=seed)
                          .split(df, groups=ds.split_key))
        tr_idx, te_idx = df.index[tr_i], df.index[te_i]
    else:
        tr_idx, te_idx = train_test_split(df.index, test_size=test_size, stratify=df["label"], random_state=seed)
    return df.loc[tr_idx].reset_index(drop=True), df.loc[te_idx].reset_index(drop=True)


def prepare(ds, train, test, corr_threshold=0.95, remove_outliers=False, zscore=False):
    train, test = train.copy(), test.copy()
    cats = [c for c in ds.categorical if c in train]
    feats = [c for c in train.columns if c != "label"]
    num = [c for c in feats if c not in cats]
    info = {}

    # numeric hygiene: bad values -> NaN, keep magnitudes float32-safe for the models
    limit = np.finfo(np.float32).max / 100
    for d in (train, test):
        d[num] = d[num].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).clip(-limit, limit)

    # (1a) constant and duplicated columns
    info["constant"] = [c for c in feats if train[c].nunique(dropna=True) <= 1]
    feats = [c for c in feats if c not in info["constant"]]
    dup = train[feats].T.duplicated()
    info["duplicate_columns"] = dup.index[dup].tolist()
    feats = [c for c in feats if c not in info["duplicate_columns"]]

    # (4) imputation: median for numeric, mode for categorical (train statistics)
    num = [c for c in feats if c not in cats]
    medians = train[num].median()
    for d in (train, test):
        d[num] = d[num].fillna(medians)
    for c in [c for c in feats if c in cats]:
        mode = train[c].mode(dropna=True)
        fill = mode.iloc[0] if len(mode) else "missing"
        train[c], test[c] = train[c].fillna(fill), test[c].fillna(fill)

    # (1b) drop one of each pair of numeric features with |correlation| > threshold
    corr = train[num].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), 1))
    info["correlated"] = [c for c in upper.columns if (upper[c] > corr_threshold).any()]
    feats = [c for c in feats if c not in info["correlated"]]
    num = [c for c in feats if c not in cats]

    # (3a) optional IQR outlier removal on TRAIN rows only (test is never filtered)
    info["outliers_removed"] = None
    if remove_outliers and num:
        q1, q3 = train[num].quantile(.25), train[num].quantile(.75)
        iqr = q3 - q1
        outlier = ((train[num] < q1 - 1.5 * iqr) | (train[num] > q3 + 1.5 * iqr)).any(axis=1)
        info["outliers_removed"] = outlier.groupby(train["label"]).mean().round(3).to_dict()
        train = train.loc[~outlier].reset_index(drop=True)
    # (3b) optional z-score scaling of numeric features
    if zscore and num:
        scaler = StandardScaler().fit(train[num])
        train[num], test[num] = scaler.transform(train[num]), scaler.transform(test[num])

    for c in cats:
        train[c], test[c] = train[c].astype(str), test[c].astype(str)
    keep = feats + ["label"]
    info["rows"] = {"train": len(train), "test": len(test)}
    return Prepared(train[keep], test[keep], feats, num, [c for c in cats if c in keep], info)
