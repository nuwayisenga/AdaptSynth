"""Vote across the three rankings and pick the final top-N features."""
import pandas as pd
from common import N

names = ["mutual_info", "anova_f", "perm_importance"]
ranks = pd.DataFrame({n: pd.read_csv(f"results/rank_{n}.csv", index_col=0)["score"]
                      .rank(ascending=False) for n in names})
votes = pd.DataFrame({n: (ranks[n] <= N).astype(int) for n in names})
table = votes.add_prefix("vote_").join(ranks.add_prefix("rank_"))
table["votes"] = votes.sum(axis=1)
table["avg_rank"] = ranks.mean(axis=1)
table = table.sort_values(["votes", "avg_rank"], ascending=[False, True])
table["selected"] = [i < N for i in range(len(table))]
table.to_csv("results/voting_table.csv")
print(table.head(15).round(2).to_string())
table.index[:N].to_series().to_csv("results/selected_features.csv", index=False, header=False)
