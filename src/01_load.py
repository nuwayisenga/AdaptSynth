"""Load CIC Device Type CSVs, label by folder, clean, subsample, split."""
import glob, os
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
ROOT = "data/CSV files/CIC Device Type"
N_ROWS = 30000
DROP = ["Unnamed: 0", "ip_dst_new", "most_freq_d_ip"]  # index + IP identifiers
CATEGORICAL = ["L4_tcp", "L4_udp", "L7_http", "L7_https", "port_class_src",
               "port_class_dst", "protocol", "most_freq_prot", "label"]

frames = []
for label in ["Audio", "Camera", "Home Automation"]:
    for f in glob.glob(f"{ROOT}/{label}/**/*.csv", recursive=True):
        d = pd.read_csv(f)
        d["label"] = label
        frames.append(d)
df = pd.concat(frames, ignore_index=True)
df.columns = df.columns.str.strip()
df = df.drop(columns=DROP).drop_duplicates()
print("after dedupe:", df.shape, df.label.value_counts().to_dict())
print("missing:", df.isna().sum()[lambda s: s > 0].to_dict())
df = df.dropna()

if len(df) > N_ROWS:
    df, _ = train_test_split(df, train_size=N_ROWS, stratify=df.label, random_state=SEED)
for c in CATEGORICAL:
    df[c] = df[c].astype(str)
train, test = train_test_split(df, test_size=0.3, stratify=df.label, random_state=SEED)
train.to_csv("results/train.csv", index=False)
test.to_csv("results/test.csv", index=False)
print("train", train.shape, "test", test.shape)
print(df.nunique().sort_values().to_string())
