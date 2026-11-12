"""Train CTGAN on all features (baseline) and on the voted top-N features; sample synthetic rows."""
import json, sys, time
import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer
from common import SEED, CATEGORICAL

seed = int(sys.argv[1]) if len(sys.argv) > 1 else SEED
EPOCHS = 100
train = pd.read_csv("results/train.csv", dtype={c: str for c in CATEGORICAL})
selected = pd.read_csv("results/selected_features.csv", header=None)[0].tolist()
runs = {"baseline": [c for c in train.columns],
        "selected": selected + ["label"]}

times = {}
for name, cols in runs.items():
    data = train[cols]
    meta = Metadata.detect_from_dataframes({"iot": data})
    for c in cols:
        meta.update_column(column_name=c, table_name="iot",
                           sdtype="categorical" if c in CATEGORICAL else "numerical")
    import torch; torch.manual_seed(seed)
    synth = CTGANSynthesizer(meta, epochs=EPOCHS, verbose=False)
    t0 = time.time()
    synth.fit({"iot": data})
    times[name] = time.time() - t0
    synth.sample(len(data))["iot"].to_csv(f"results/synth_{name}_seed{seed}.csv", index=False)
    print(name, len(cols), "cols", round(times[name]), "s", flush=True)
json.dump(times, open(f"results/runtime_seed{seed}.json", "w"))
