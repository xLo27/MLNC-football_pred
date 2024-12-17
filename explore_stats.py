import pandas
import numpy as np

df = pandas.read_csv("data/features_17_24_full_ema_span=30.csv")
# print all column dtypes to file
with open("data/column_dtypes.txt", "w") as f:
    for col in df.columns:
        f.write(f"{col}: {df[col].dtype}\n")
# Example DataFrames
import pandas as pd

# First DataFrame: Matches
df_matches = pd.DataFrame(
    {"HomeTeam": ["TeamA", "TeamB", "TeamC"], "AwayTeam": ["TeamC", "TeamA", "TeamB"]}
)

# Second DataFrame: ELO ratings
df_elo = pd.DataFrame({"Team": ["TeamA", "TeamB", "TeamC"], "ELO": [1500, 1450, 1520]})

# Create a dictionary for quick lookup of ELO scores
elo_dict = df_elo.set_index("Team")["ELO"].to_dict()

# Map the ELO scores to the matches DataFrame
df_matches["home_elo"] = df_matches["HomeTeam"].map(elo_dict)
df_matches["away_elo"] = df_matches["AwayTeam"].map(elo_dict)

print(df_matches)

df_models = pd.read_csv(
    "models/model_results_tuning=make_scorer(f1_score, response_method='predict', average=weighted)_oversampled=True.csv"
)
df_filtered = df_models[["Classifier", "Accuracy", "precision", "recall", "f1-score"]]
# filter for classifiers with "tuned" in name
df_filtered = df_filtered[df_filtered["Classifier"].str.contains("tuned")]
df_filtered["Classifier"] = df_filtered["Classifier"].str.replace("_tuned", "")
print("\nWeighted by F1\n")
print(df_filtered.sort_values(by="f1-score", ascending=False).to_string(index=False))
print("\nWeighted by Accuracy\n")
print(df_filtered.sort_values(by="Accuracy", ascending=False).to_string(index=False))
df_sorted_accuracy = df_filtered.sort_values(by="Accuracy", ascending=False)
for _, row in df_sorted_accuracy.iterrows():
    # print columns separated by "&"
    latex_str = row["Classifier"]
    # round each column to 2 decimal places and covert to percent
    for column in ["Accuracy", "precision", "recall", "f1-score"]:
        latex_str += f" & {round(100*row[column],2)}" + r"\%"
    latex_str += r"\\ \hline"
    print(latex_str)
