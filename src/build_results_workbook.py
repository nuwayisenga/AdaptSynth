"""Build paper_results/AdaptSynth_results.xlsx from paper_results/experiment_log.csv.

Each paper table gets a sheet with the numbers currently in the paper (grey) next to
our measured numbers (blue), so it is easy to see what to update.

    .venv-server/bin/python src/build_results_workbook.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from results_log import COLUMNS, LOG_PATH, ROOT, RUN_KEY, load_log  # noqa: E402

OUT_PATH = ROOT / "paper_results" / "AdaptSynth_results.xlsx"
DATASETS = ["SDN Anomaly", "IoT-23", "Kaggle Cyber", "CIC-IDS2017"]
METHODS = ["GVAE", "SMOTE", "TVAE", "CTAB-GAN+", "TabDDPM", "AdaptSynth"]

# ---- Numbers as printed in "AdaptSynth (44).pdf" -------------------------------------
# Table II: (TRTR F1, TRTR ROC, TSTR F1, TSTR AUC)
PAPER_T2 = {
    "SDN Anomaly": {"GVAE": (.88, .94, .86, .91), "SMOTE": (.84, .90, .80, .87), "TVAE": (.82, .82, .77, .80),
                    "CTAB-GAN+": (.80, .79, .74, .77), "TabDDPM": (.81, .78, .74, .75), "AdaptSynth": (None, None, .86, .92)},
    "IoT-23": {"GVAE": (.88, .90, .85, .85), "SMOTE": (.84, .85, .80, .81), "TVAE": (.82, .83, .75, .78),
               "CTAB-GAN+": (.82, .83, .76, .78), "TabDDPM": (.81, .82, .76, .82), "AdaptSynth": (None, None, .85, .85)},
    "Kaggle Cyber": {"GVAE": (.82, .84, .78, .80), "SMOTE": (.75, .77, .69, .70), "TVAE": (.82, .84, .77, .79),
                     "CTAB-GAN+": (.87, .90, .83, .84), "TabDDPM": (.81, .84, .80, .82), "AdaptSynth": (None, None, .86, .87)},
    "CIC-IDS2017": {"GVAE": (.83, .86, .79, .79), "SMOTE": (.82, .85, .75, .77), "TVAE": (.86, .85, .79, .80),
                    "CTAB-GAN+": (.84, .84, .81, .80), "TabDDPM": (.90, .93, .84, .86), "AdaptSynth": (None, None, .89, .91)},
}
# Table III: (MSE, similarity)
PAPER_T3 = {
    "SDN Anomaly": {"GVAE": (.041, .93), "TabDDPM": (.058, .89), "CTAB-GAN+": (.062, .88), "TVAE": (.065, .86),
                    "SMOTE": (.089, .81), "AdaptSynth": (.038, .94)},
    "IoT-23": {"GVAE": (.039, .94), "TabDDPM": (.065, .86), "CTAB-GAN+": (.071, .85), "TVAE": (.075, .83),
               "SMOTE": (.095, .83), "AdaptSynth": (.037, .95)},
    "Kaggle Cyber": {"CTAB-GAN+": (.045, .91), "TabDDPM": (.052, .89), "TVAE": (.058, .88), "GVAE": (.068, .86),
                     "SMOTE": (.098, .76), "AdaptSynth": (.042, .92)},
    "CIC-IDS2017": {"TabDDPM": (.032, .92), "TVAE": (.048, .90), "GVAE": (.051, .91), "CTAB-GAN+": (.078, .88),
                    "SMOTE": (.092, .82), "AdaptSynth": (.029, .94)},
}
# Table IV: (component, dataset, F1 drop %, ROC drop %)
PAPER_T4 = [("w/o Graph Awareness (G)", "SDN Anomaly", -3.4, -2.8),
            ("w/o Graph Awareness (G)", "Kaggle Cyber", -0.2, -0.1),
            ("w/o Categorical Ratio (C)", "Kaggle Cyber", -4.1, -3.5),
            ("w/o Categorical Ratio (C)", "CIC-IDS2017", -0.5, -0.3),
            ("w/o RL Policy (Random)", "Global Avg.", -3.8, -3.1)]
# Table V: (MIA advantage %, risk)
PAPER_T5 = {"TabDDPM": (8.4, "Low"), "TVAE": (14.2, "Low-Med"), "GVAE": (19.7, "Medium"),
            "CTAB-GAN+": (22.4, "Med-High"), "SMOTE": (44.1, "Critical")}

# Notebook output folders that existed before the log; imported once so they aren't lost.
LEGACY_DIRS = {"results_iot23": "IoT-23", "results_machinelearning": "CIC-IDS2017"}


def backfill_legacy(log):
    """Import comparison.csv from the older output folders if not already in the log."""
    from results_log import log_comparison
    for folder, dataset in LEGACY_DIRS.items():
        path = ROOT / folder / "comparison.csv"
        tag = f"{folder}/comparison.csv (backfilled)"
        if not path.exists() or (log["source"] == tag).any():
            continue
        cfg_path = ROOT / folder / "run_config.json"
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        log_comparison(pd.read_csv(path), dataset=dataset, epochs=cfg.get("epochs", 200),
                       trtr_f1=cfg.get("real_train_macroF1"), source=tag,
                       notes="Imported from notebook output that existed before the run log.")
    return load_log()


def latest_runs(log):
    """Keep the most recent row for each experiment+seed so re-runs replace old numbers."""
    return log.sort_values("timestamp").drop_duplicates(RUN_KEY, keep="last")


def method_label(row):
    label = row["generator"]
    if row["feature_set"] and row["feature_set"] != "all features":
        label += f" [{row['feature_set']}]"
    elif row["generator"] not in METHODS and row["feature_set"]:
        label += f" [{row['feature_set']}]"
    if row["config_tag"]:
        label += f" ({row['config_tag']})"
    return label


def aggregate(runs, metrics):
    runs = runs[runs["ablation"] == ""].copy()
    if runs.empty:
        return pd.DataFrame(columns=["dataset", "method", "n_runs", "classifier"])
    runs["method"] = runs.apply(method_label, axis=1)
    g = runs.groupby(["dataset", "method"])
    out = g[metrics].mean().add_suffix(" (mean)")
    stds = g[metrics].std().add_suffix(" (std)")
    out = out.join(stds)
    out["n_runs"] = g.size()
    out["classifier"] = g["classifier"].agg(lambda s: ", ".join(sorted(set(map(str, s)))))
    return out.reset_index()


def merge_with_paper(paper_rows, ours, paper_cols):
    paper = pd.DataFrame(paper_rows, columns=["dataset", "method"] + paper_cols)
    df = paper.merge(ours, on=["dataset", "method"], how="outer")
    order = {d: i for i, d in enumerate(DATASETS)}
    df["_d"] = df["dataset"].map(order).fillna(len(DATASETS))
    df["_m"] = df["method"].map({m: i for i, m in enumerate(METHODS)}).fillna(len(METHODS))
    return df.sort_values(["_d", "_m", "method"]).drop(columns=["_d", "_m"])


def table2(runs):
    rows = [(d, m, *v) for d, ms in PAPER_T2.items() for m, v in ms.items()]
    ours = aggregate(runs, ["tstr_f1_macro", "tstr_roc_auc", "trtr_f1_macro", "trtr_roc_auc"])
    df = merge_with_paper(rows, ours, ["Paper TRTR F1", "Paper TRTR ROC", "Paper TSTR F1", "Paper TSTR AUC"])
    df["Δ TSTR F1 (ours − paper)"] = df.get("tstr_f1_macro (mean)") - df["Paper TSTR F1"]
    return df


def table3(runs):
    rows = [(d, m, *v) for d, ms in PAPER_T3.items() for m, v in ms.items()]
    ours = aggregate(runs, ["dist_mse", "dist_similarity", "sim_numerical", "sim_categorical", "sim_relationships"])
    return merge_with_paper(rows, ours, ["Paper MSE ↓", "Paper Similarity ↑"])


def table4(runs):
    """Ablation: % drop vs. the full AdaptSynth run on the same dataset."""
    full = runs[(runs["generator"] == "AdaptSynth") & (runs["ablation"] == "")]
    full = full.groupby("dataset")[["tstr_f1_macro", "tstr_roc_auc"]].mean()
    abl = runs[runs["ablation"] != ""].groupby(["ablation", "dataset"])[["tstr_f1_macro", "tstr_roc_auc"]].mean()
    ours = []
    for (comp, ds), r in abl.iterrows():
        base = full.loc[ds] if ds in full.index else None
        pct = (lambda k: None if base is None or pd.isna(base[k]) else 100 * (r[k] - base[k]) / base[k])
        ours.append((comp, ds, pct("tstr_f1_macro"), pct("tstr_roc_auc")))
    paper = pd.DataFrame(PAPER_T4, columns=["component", "dataset", "Paper F1 drop %", "Paper ROC drop %"])
    ours = pd.DataFrame(ours, columns=["component", "dataset", "Ours F1 drop %", "Ours ROC drop %"])
    return paper.merge(ours, on=["component", "dataset"], how="outer")


def table5(runs):
    paper = pd.DataFrame([(m, *v) for m, v in PAPER_T5.items()],
                         columns=["method", "Paper MIA advantage %", "Paper risk"])
    r = runs.dropna(subset=["mia_advantage_pct"])
    ours = r.groupby("generator")["mia_advantage_pct"].agg(["mean", "std", "count"]).reset_index()
    ours.columns = ["method", "Ours MIA advantage % (mean)", "(std)", "n_runs"]
    return paper.merge(ours, on="method", how="outer")


def feature_selection(runs):
    """The CTGAN all-features vs voted-top-N experiment the notebooks currently run."""
    r = runs[runs["feature_set"].isin(["all features", "voted top-N"])]
    metrics = ["n_features", "tstr_f1_macro", "trtr_f1_macro", "sim_numerical", "sim_categorical",
               "sim_relationships", "train_seconds"]
    g = r.groupby(["dataset", "generator", "feature_set", "classifier", "config_tag"])
    out = g[metrics].mean().join(g["tstr_f1_macro"].std().rename("tstr_f1_macro (std)"))
    out["n_seeds"] = g.size()
    return out.reset_index()


README = [
    ("What this is", "Results for the AdaptSynth paper, generated from paper_results/experiment_log.csv. "
                     "Do not edit this workbook by hand: it is overwritten every time it is rebuilt."),
    ("How to add results", "At the end of a notebook run: log_comparison(...) or log_result(...) from src/results_log.py. "
                           "Then rebuild: .venv-server/bin/python src/build_results_workbook.py"),
    ("Re-runs", "Rows with the same dataset/generator/feature_set/ablation/config_tag/classifier/seed are re-runs; "
                "only the newest is used in the tables. The Run log sheet keeps every row."),
    ("Colours", "Grey columns = numbers currently printed in AdaptSynth (44).pdf. Blue columns = our measured numbers."),
    ("Blank 'ours' cells", "No experiment has been logged for that dataset/method yet."),
    ("Watch out", "The paper's Table II uses XGBoost; the current notebooks use RandomForest and CTGAN. "
                  "The 'classifier' column shows which was used, so compare like with like before copying numbers into the paper."),
]

GREY = PatternFill("solid", fgColor="E7E6E6")
BLUE = PatternFill("solid", fgColor="DDEBF7")
HEAD = PatternFill("solid", fgColor="1F3864")


def style(ws, df):
    for j, col in enumerate(df.columns, 1):
        cell = ws.cell(row=1, column=j)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = HEAD
        cell.alignment = Alignment(wrap_text=True, vertical="center")
        fill = GREY if str(col).startswith("Paper") else BLUE if (
            str(col).startswith("Ours") or "(mean)" in str(col) or "(std)" in str(col) or str(col).startswith("Δ")) else None
        width = max(10, min(40, max([len(str(col))] + [len(f"{v:.3f}" if isinstance(v, float) else str(v))
                                                         for v in df[col].head(200)]) + 2))
        ws.column_dimensions[get_column_letter(j)].width = width if j > 2 else max(width, 16)
        for i in range(2, len(df) + 2):
            c = ws.cell(row=i, column=j)
            if fill:
                c.fill = fill
            if isinstance(c.value, float):
                c.number_format = "0.000"
    ws.row_dimensions[1].height = 32
    ws.freeze_panes = "C2"
    ws.auto_filter.ref = ws.dimensions


def build():
    log = backfill_legacy(load_log())
    runs = latest_runs(log)
    sheets = {
        "README": pd.DataFrame(README, columns=["Topic", "Details"]),
        "Table II Utility": table2(runs),
        "Table III Fidelity": table3(runs),
        "Table IV Ablation": table4(runs),
        "Table V Privacy (MIA)": table5(runs),
        "Feature selection": feature_selection(runs),
        "Run log (all)": log.reindex(columns=COLUMNS).sort_values("timestamp"),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name, index=False)
            style(xw.sheets[name], df)
        xw.sheets["README"].column_dimensions["B"].width = 120
    print(f"Wrote {OUT_PATH.relative_to(ROOT)} ({len(log)} logged runs, {len(runs)} unique)")
    return OUT_PATH


if __name__ == "__main__":
    build()
