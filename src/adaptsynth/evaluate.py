"""Compare synthetic vs real: (A) column distributions, (B) column relationships, (C) TSTR / TRTR utility."""
import numpy as np
import pandas as pd
from sdmetrics.column_pairs import ContingencySimilarity, CorrelationSimilarity
from sdmetrics.single_column import KSComplement, TVComplement
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from xgboost import XGBClassifier


def similarity(real, syn, cols, categorical):
    """Average similarity (0-1, higher = more similar) over `cols`.
    A: per column  - KSComplement (numeric) / TVComplement (categorical)
    B: per pair    - CorrelationSimilarity (numeric pairs) / ContingencySimilarity (categorical pairs)"""
    num = [c for c in cols if c not in categorical]
    cat = [c for c in cols if c in categorical]
    per_col = {c: KSComplement.compute(real[c], syn[c]) for c in num}
    per_col.update({c: TVComplement.compute(real[c], syn[c]) for c in cat})
    def safe(metric, a, b):
        # a constant column (common for an undertrained generator) makes sdmetrics raise; score it as NaN and skip
        try:
            return metric.compute(real[[a, b]], syn[[a, b]])
        except Exception:
            return np.nan

    pairs = []
    for i, a in enumerate(num):
        for b in num[i + 1:]:
            pairs.append(safe(CorrelationSimilarity, a, b))
    for i, a in enumerate(cat):
        for b in cat[i + 1:]:
            pairs.append(safe(ContingencySimilarity, a, b))
    return {"sim_numerical": np.mean([per_col[c] for c in num]) if num else np.nan,
            "sim_categorical": np.mean([per_col[c] for c in cat]) if cat else np.nan,
            "sim_relationships": np.nanmean(pairs) if pairs else np.nan}, per_col


def utility(train_df, test_df, cols, categorical, all_classes, seed=42):
    """Train XGBoost on `train_df`, score on the REAL `test_df`. Returns macro-F1 and macro ROC-AUC.
    Pass synthetic data as train_df for TSTR, real train data for TRTR.
    A class the training table never contains cannot be predicted, so it scores F1 = 0 (and AUC = 0.5)."""
    feats = [c for c in cols if c != "label"]
    cat_f = [c for c in feats if c in categorical]
    Xtr, Xte = train_df[feats].copy(), test_df[feats].copy()
    if cat_f:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        Xtr[cat_f] = enc.fit_transform(Xtr[cat_f]); Xte[cat_f] = enc.transform(Xte[cat_f])
    le = LabelEncoder().fit(train_df["label"])
    y_te = test_df["label"].to_numpy()
    classes = list(all_classes)
    proba = np.zeros((len(Xte), len(classes)))
    if len(le.classes_) < 2:                                   # synthetic data has a single class
        pred = np.full(len(Xte), le.classes_[0]); proba[:, classes.index(le.classes_[0])] = 1.0
    else:
        clf = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, tree_method="hist",
                            n_jobs=-1, random_state=seed, verbosity=0)
        clf.fit(Xtr, le.transform(train_df["label"]))
        p = clf.predict_proba(Xte)
        for j, c in enumerate(le.classes_):
            proba[:, classes.index(c)] = p[:, j]
        pred = np.array(classes)[proba.argmax(axis=1)]
    f1 = f1_score(y_te, pred, labels=classes, average="macro", zero_division=0)
    aucs = [roc_auc_score(y_te == c, proba[:, k]) for k, c in enumerate(classes)
            if (y_te == c).any() and (y_te != c).any()]
    return {"f1_macro": f1, "roc_auc": float(np.mean(aucs)) if aucs else np.nan}
