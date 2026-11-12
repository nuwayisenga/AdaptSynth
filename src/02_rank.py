"""Rank features on train only: mutual information, ANOVA F, RF permutation importance."""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from common import SEED, CATEGORICAL

train = pd.read_csv("results/train.csv", dtype={c: str for c in CATEGORICAL})
y = train.pop("label")
X = train.copy()
for c in CATEGORICAL:
    if c in X:
        X[c] = X[c].astype(int)
is_cat = [c in CATEGORICAL for c in X.columns]

mi = pd.Series(mutual_info_classif(X, y, discrete_features=is_cat, random_state=SEED), X.columns)
anova = pd.Series(f_classif(X, y)[0], X.columns)

Xa, Xv, ya, yv = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED).fit(Xa, ya)
print("inner val accuracy:", rf.score(Xv, yv))
perm = pd.Series(permutation_importance(rf, Xv, yv, n_repeats=5, random_state=SEED,
                                        n_jobs=-1).importances_mean, X.columns)

for name, s in [("mutual_info", mi), ("anova_f", anova), ("perm_importance", perm)]:
    s.sort_values(ascending=False).rename("score").to_csv(f"results/rank_{name}.csv")
    print(name, list(s.sort_values(ascending=False).index[:10]))
