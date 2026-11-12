"""Score synthetic data vs held-out real test data with sdmetrics; build comparison table."""
import json, sys
import numpy as np, pandas as pd
from sdv.metadata import Metadata
from sdmetrics.single_column import KSComplement, TVComplement
from sdmetrics.column_pairs import CorrelationSimilarity, ContingencySimilarity
from common import SEED, CATEGORICAL

seed = int(sys.argv[1]) if len(sys.argv) > 1 else SEED
test = pd.read_csv("results/test.csv", dtype={c: str for c in CATEGORICAL})
selected = pd.read_csv("results/selected_features.csv", header=None)[0].tolist() + ["label"]
times = json.load(open(f"results/runtime_seed{seed}.json"))

def score(real, syn, cols):
    num = [c for c in cols if c not in CATEGORICAL]
    cat = [c for c in cols if c in CATEGORICAL]
    ks = [KSComplement.compute(real[c], syn[c]) for c in num]
    tv = [TVComplement.compute(real[c], syn[c]) for c in cat]
    rel = []
    for i, a in enumerate(num):
        for b in num[i + 1:]:
            rel.append(CorrelationSimilarity.compute(real[[a, b]], syn[[a, b]]))
    for i, a in enumerate(cat):
        for b in cat[i + 1:]:
            rel.append(ContingencySimilarity.compute(real[[a, b]], syn[[a, b]]))
    return dict(numerical=np.nanmean(ks), categorical=np.nanmean(tv),
                relationship=np.nanmean(rel), n_cols=len(cols))

rows = []
for name in ["baseline", "selected"]:
    syn = pd.read_csv(f"results/synth_{name}_seed{seed}.csv", dtype={c: str for c in CATEGORICAL})
    for scope, cols in [("own columns", list(syn.columns)), ("same selected columns", selected)]:
        rows.append(dict(experiment=name, scored_on=scope, seed=seed,
                         runtime_s=round(times[name]), **score(test, syn, cols)))
out = pd.DataFrame(rows).round(3)
out.to_csv(f"results/comparison_seed{seed}.csv", index=False)
print(out.to_string())
