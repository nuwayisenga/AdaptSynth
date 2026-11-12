"""Rank features with five techniques (3 filter, 1 wrapper, 1 embedded), then vote.
Only ever sees the TRAIN split."""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, f_classif, mutual_info_classif
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder

METHODS = ["anova_f (filter)", "chi2_cramersV (filter)", "mutual_info (filter)",
           "rfe (wrapper)", "rf_importance (embedded)"]


def rank_features(prep, n_select, seed=42, rfe_step=1):
    """Return a DataFrame of raw scores: one row per feature, one column per technique (higher = better)."""
    train, feats = prep.train, prep.features
    cat_f = [c for c in feats if c in prep.categorical]
    num_f = [c for c in feats if c in prep.numerical]
    y = train["label"]
    X = train[feats].copy()
    if cat_f:
        X[cat_f] = OrdinalEncoder().fit_transform(X[cat_f])          # integer codes, ranking only
    discrete = [c in cat_f for c in feats]

    # filter 1: ANOVA F (numeric only; NaN for categorical)
    anova = (pd.Series(f_classif(X[num_f], y)[0], index=num_f).reindex(feats) if num_f
             else pd.Series(np.nan, index=feats))

    # filter 2: chi-square -> Cramer's V (numeric features cut into 10 quantile bins first)
    binned = X.copy()
    if num_f:
        binned[num_f] = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile",
                                         subsample=None).fit_transform(X[num_f])

    def cramers_v(f):
        table = pd.crosstab(binned[f], y)
        if min(table.shape) < 2:
            return 0.0
        chi2 = chi2_contingency(table, correction=False)[0]
        return np.sqrt(chi2 / (len(y) * (min(table.shape) - 1)))
    chi = pd.Series({f: cramers_v(f) for f in feats})

    # filter 3: mutual information
    mi = pd.Series(mutual_info_classif(X, y, discrete_features=discrete, random_state=seed), index=feats)

    # wrapper: RFE with a random forest (ranking_ 1 = survivor; negate so higher = better)
    rfe = RFE(RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1),
              n_features_to_select=n_select, step=rfe_step).fit(X, y)
    rfe_score = pd.Series(-rfe.ranking_.astype(float), index=feats)

    # embedded: random-forest impurity importance
    forest = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1).fit(X, y)
    embedded = pd.Series(forest.feature_importances_, index=feats)

    return pd.DataFrame(dict(zip(METHODS, [anova, chi, mi, rfe_score, embedded])))


def vote(scores, n_select):
    """Each technique gives one vote to each of its top `n_select` features. Sort by votes, break ties
    with average rank (lower = better). Returns (vote_table, selected_features)."""
    ranks = scores.rank(ascending=False, method="first")       # each technique hands out exactly n votes
    ranks["rfe (wrapper)"] = scores["rfe (wrapper)"].rank(ascending=False, method="min")  # RFE survivors tie at 1
    votes = (ranks <= n_select).astype(int)
    table = votes.add_prefix("vote_").join(ranks.add_prefix("rank_"))
    table["votes"] = votes.sum(axis=1)
    table["avg_rank"] = ranks.mean(axis=1)
    table = table.sort_values(["votes", "avg_rank"], ascending=[False, True])
    table["selected"] = [i < n_select for i in range(len(table))]
    return table, table.index[:n_select].tolist()
