import pandas as pd
import os
from PlayerEloMatchUpdated import calculate_player_elo


def process_multiple_files(files, cutoff_date):
    all_data = []

    for filename in files:
        if filename.endswith(".csv"):
            print(f"Reading file: {filename}")
            data = pd.read_csv(os.path.join("data", filename))
            new_columns = data.iloc[0]
            data = data[1:]
            data.columns = new_columns
            data = data.loc[:, ~data.columns.duplicated()].copy()
            data["Date"] = pd.to_datetime(
                data["Date"], format="%Y-%m-%d", errors="coerce"
            )
            data = data[data["Date"] <= cutoff_date]
            print(f"Number of rows in {filename} after date filter: {len(data)}")
            all_data.append(data)

    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"Total combined rows: {len(combined_data)}")
    return combined_data


data = process_multiple_files(
    ["2122_players.csv", "2223_players.csv", "2324_players.csv"], "2023-12-31"
)


calculate_player_elo(data)

ranked_players = pd.read_csv("ranked_players.csv")

team_position_elo = (
    ranked_players.groupby(["Team", "Position"])["Elo"].mean().reset_index()
)

team_avg_elo = team_position_elo.pivot(index="Team", columns="Position", values="Elo")

team_avg_elo = team_avg_elo.reset_index()

if "Pos." in team_avg_elo.columns:
    team_avg_elo = team_avg_elo.drop(columns=["Pos."])

team_avg_elo = team_avg_elo.loc[:, ~team_avg_elo.columns.str.contains("Column_")]

team_avg_elo = team_avg_elo.dropna(subset=["Team"])
team_avg_elo = team_avg_elo[~team_avg_elo["Team"].str.contains("Team|Column", na=False)]

weights = {"FW": 1.5, "MF": 1.0, "DF": 1.0, "GK": 1.0}

team_avg_elo["Weighted Elo"] = (
    team_avg_elo["FW"].fillna(0) * weights["FW"]
    + team_avg_elo["MF"].fillna(0) * weights["MF"]
    + team_avg_elo["DF"].fillna(0) * weights["DF"]
    + team_avg_elo["GK"].fillna(0) * weights["GK"]
)

sorted_teams = team_avg_elo.sort_values(by="Weighted Elo", ascending=False)

sorted_teams.to_csv("team_elo.csv", index=False)

print("saved")
