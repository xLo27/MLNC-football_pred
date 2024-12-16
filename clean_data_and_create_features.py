import pandas as pd
from datetime import datetime
import numpy as np
import os
from model_training import get_cv_score
import matplotlib.pyplot as plt


def drop_duplicate_columns(data: pd.DataFrame) -> pd.DataFrame:
    cols = data.columns
    duplicate_cols = []
    for i in range(len(cols)):
        for j in range(i):
            if data[cols[i]].dtype == data[cols[j]].dtype:
                if data[cols[i]].equals(data[cols[j]]):
                    duplicate_cols.append(cols[i])
                    break
    data = data.drop(columns=duplicate_cols)
    return data


def clean_data(data: pd.DataFrame) -> pd.DataFrame:

    print("Original Length:", len(data), len(data.columns))

    data = data.dropna(how="all")
    print("After dropping NaN:", len(data), len(data.columns))
    data = data.drop_duplicates()
    print("After dropping duplicate rows:", len(data), len(data.columns))
    data = drop_duplicate_columns(data)
    print("After dropping duplicate columns:", len(data), len(data.columns))

    data["HomeGame"] = data["Unnamed: 13"].isna()
    data = data.drop(columns=["Unnamed: 13", "Match Report", "Comp", "Rk"])
    print("After dropping columns:", len(data), len(data.columns))

    data["Date"] = pd.to_datetime(data["Date"])
    data = data.drop(columns=["Result"])

    data["FTR"] = data.apply(
        lambda row: (
            "D"
            if row["GD"] == 0
            else ("H" if (not (row["GD"] > 0) ^ row["HomeGame"]) else "A")
        ),
        axis=1,
    )
    name_map = {"1/3.1": "Carries_1/3", "Att": "Att_passes", "Att.1": "Att_passes_"}
    opta_cols = [
        "xG",
        "npxG",
        "xGD",
        "npxGD",
        "xAG",
        "xA",
        "G-xG",
        "np:G-xG",
        "A-xAG",
        "npxG/Sh",
    ]
    # TODO: Figure out if should include these
    data = data.drop(columns=opta_cols)

    # print to file ambiguous named columns (sorted)
    # lines = []
    # with open("ambiguous_columns.txt", "w") as f:
    #     for column in data.columns:
    #         if "." in column:
    #             lines.append(column)
    #     lines.sort()
    #     for line in lines:
    #         f.write(line + "\n")

    return data.copy()


def create_ema_features(data: pd.DataFrame, span):
    non_feature_cols = ["Date", "HomeGame", "FTR", "Team", "Opp"]
    # df_ema_features = data[non_feature_cols].copy().sort_values(by=["Date"])
    # feature_names = data.drop(columns=non_feature_cols).columns
    # ema_features = []
    # for feature_name in feature_names:
    #     feature_ema = data.groupby("Team")[feature_name].transform(
    #         lambda row: row.ewm(span=span, min_periods=2).mean().shift(1)
    #     )
    #     ema_features.append(pd.Series(feature_ema, name=feature_name))
    # df_ema_features = pd.concat(
    #     [df_ema_features, pd.concat(ema_features, axis=1)], axis=1
    # )
    # return df_ema_features
    df = data
    feature_names = df.drop(columns=non_feature_cols).columns
    ema_features = []
    for feature_name in feature_names:
        feature_ema = data.groupby("Team")[feature_name].transform(
            lambda row: row.ewm(span=span, min_periods=2).mean().shift(1)
        )
        df["f_" + feature_name] = feature_ema
    return df.copy()


def restructure_data(data: pd.DataFrame):
    """
    Merge the data such that each row represents features for both the home and away team in a particular match.
    """
    non_feature_cols = ["Date", "FTR", "HomeTeam", "AwayTeam"]
    unwanted_cols = ["Opp_home", "Opp_away", "HomeGame_home", "HomeGame_away"]
    data_merged = (
        data.query("HomeGame == True")
        .rename(columns={"Team": "HomeTeam"})
        .pipe(
            pd.merge,
            data.query("HomeGame == False").rename(columns={"Team": "AwayTeam"}),
            left_on=[
                "Date",
                "FTR",
                "HomeTeam",
                "Opp",
            ],
            right_on=[
                "Date",
                "FTR",
                "Opp",
                "AwayTeam",
            ],
            suffixes=("_home", "_away"),
        )
    )
    data_merged = data_merged.drop(columns=unwanted_cols).dropna()
    # data_merged = data_merged.rename(
    #     columns={
    #         col: "f_" + col
    #         for col in data_merged.columns
    #         if col not in non_feature_cols
    #     }
    # )
    return data_merged


def optimize_ema(data: pd.DataFrame):
    scores = []
    best_score = np.float16("inf")
    best_span = 0
    spans = range(1, 500, 10)
    for i, span in enumerate(spans):
        if i % 10 == 0:
            print(f"Optimizing span {span}")
        data_features = create_ema_features(data, span)
        data_restructured = restructure_data(data_features)
        score = get_cv_score(data_restructured)
        scores.append(score)
        if score * -1 < best_score:
            best_score = score * -1
            best_span = span
    print(f"Best span: {best_span}, Best score: {best_score}")
    # plot plt graph of scores

    plt.plot(spans, -1 * pd.Series(scores))
    plt.xlabel("Span")
    plt.ylabel("Log Loss")
    plt.show()


def add_seasonal_features(df: pd.DataFrame):
    team_stats = {}
    season_threshold = 60  # Gap in days to detect new season

    # Function to reset team stats at the start of a new season
    def reset_team_stats():
        return {
            "GoalsScored": 0,
            "GoalsConceded": 0,
            "Points": 0,
            "Form": ["M"] * 5,
            "WinStreak3": 0,
            "WinStreak5": 0,
            "LossStreak3": 0,
            "LossStreak5": 0,
        }

    # Function to calculate win/loss streaks
    def update_streaks(form):
        """Calculate win/loss streaks based on the last 5 matches."""
        win_streak_3 = int(form[:3] == ["W", "W", "W"])
        win_streak_5 = int(form[:5] == ["W", "W", "W", "W", "W"])
        loss_streak_3 = int(form[:3] == ["L", "L", "L"])
        loss_streak_5 = int(form[:5] == ["L", "L", "L", "L", "L"])
        return win_streak_3, win_streak_5, loss_streak_3, loss_streak_5

    # Initialize matchweek counter
    matchweek = 1
    last_date = df["Date"].iloc[0]

    df = df.sort_values(by="Date", ascending=True)
    new_features = []
    # Iterate over each row in the dataset
    for _, row in df.iterrows():
        home_team = row["HomeTeam"]
        away_team = row["AwayTeam"]
        fthg, ftag = row["GF_home"], row["GF_away"]
        ftr = row["FTR"]
        date = row["Date"]

        # Detect new season if the date gap exceeds the threshold
        if (date - last_date).days > season_threshold:
            team_stats.clear()
            matchweek = 1

        # Ensure both teams have initialized stats
        if home_team not in team_stats:
            team_stats[home_team] = reset_team_stats()
        if away_team not in team_stats:
            team_stats[away_team] = reset_team_stats()

        # Get current stats
        home = team_stats[home_team]
        away = team_stats[away_team]

        # Calculate Goal Differences
        htgd = home["GoalsScored"] - home["GoalsConceded"]
        atgd = away["GoalsScored"] - away["GoalsConceded"]

        # Form Points Calculation
        form_points = {"W": 3, "D": 1, "L": 0}
        ht_form_pts = sum([form_points.get(x, 0) for x in home["Form"]])
        at_form_pts = sum([form_points.get(x, 0) for x in away["Form"]])

        # Update streaks
        (
            home["WinStreak3"],
            home["WinStreak5"],
            home["LossStreak3"],
            home["LossStreak5"],
        ) = update_streaks(home["Form"])
        (
            away["WinStreak3"],
            away["WinStreak5"],
            away["LossStreak3"],
            away["LossStreak5"],
        ) = update_streaks(away["Form"])

        cum_stats = {
            "HTGS": home["GoalsScored"],
            "ATGS": away["GoalsScored"],
            "HTGC": home["GoalsConceded"],
            "ATGC": away["GoalsConceded"],
            "HTP": home["Points"],
            "ATP": away["Points"],
            "MatchWeek": matchweek,
        }
        win_loss_streak_stats = {
            "HTWinStreak3": home["WinStreak3"],
            "HTWinStreak5": home["WinStreak5"],
            "HTLossStreak3": home["LossStreak3"],
            "HTLossStreak5": home["LossStreak5"],
            "ATWinStreak3": away["WinStreak3"],
            "ATWinStreak5": away["WinStreak5"],
            "ATLossStreak3": away["LossStreak3"],
            "ATLossStreak5": away["LossStreak5"],
        }
        # hm1 to atformpts
        recent_form_stats = {
            "HM1": home["Form"][0],
            "HM2": home["Form"][1],
            "HM3": home["Form"][2],
            "HM4": home["Form"][3],
            "HM5": home["Form"][4],
            "AM1": away["Form"][0],
            "AM2": away["Form"][1],
            "AM3": away["Form"][2],
            "AM4": away["Form"][3],
            "AM5": away["Form"][4],
            "HTFormPtsStr": "".join(home["Form"]),
            "ATFormPtsStr": "".join(away["Form"]),
            "HTFormPts": ht_form_pts,
            "ATFormPts": at_form_pts,
        }
        goal_point_diff_stats = {
            "HTGD": htgd,
            "ATGD": atgd,
            "DiffPts": home["Points"] - away["Points"],
            "DiffFormPts": ht_form_pts - at_form_pts,
        }

        row_stats = (
            cum_stats
            | win_loss_streak_stats
            | recent_form_stats
            | goal_point_diff_stats
        )
        row_stats = {f"f_{k}": v for k, v in row_stats.items()}
        new_features.append(row_stats)

        # Update stats after the match
        home["GoalsScored"] += fthg
        home["GoalsConceded"] += ftag
        away["GoalsScored"] += ftag
        away["GoalsConceded"] += fthg

        # Update points and form
        if ftr == "H":
            home["Points"] += 3
            home["Form"] = (["W"] + home["Form"])[:5]
            away["Form"] = (["L"] + away["Form"])[:5]
        elif ftr == "A":
            away["Points"] += 3
            home["Form"] = (["L"] + home["Form"])[:5]
            away["Form"] = (["W"] + away["Form"])[:5]
        else:
            home["Points"] += 1
            away["Points"] += 1
            home["Form"] = (["D"] + home["Form"])[:5]
            away["Form"] = (["D"] + away["Form"])[:5]

        # Update matchweek and last_date
        matchweek += 1
        last_date = date
    new_features = pd.DataFrame(new_features, index=df.index)
    df = pd.concat([df, new_features], axis=1)
    return df.copy()


def add_team_elo(df: pd.DataFrame):
    # read data/team_elo.csv
    df_elo = pd.read_csv("data/team_elo.csv")
    # add a column to df for home team elo and away team elo
    for elo_column in [col for col in df_elo.columns if col != "Team"]:
        # Create a dictionary for quick lookup for the current ELO column
        elo_dict = df_elo.set_index("Team")[elo_column].to_dict()
        # Add corresponding home and away ELO columns to df_matches
        df[f"f_{elo_column.lower()}_home"] = df["HomeTeam"].map(elo_dict)
        df[f"f_{elo_column.lower()}_away"] = df["AwayTeam"].map(elo_dict)
    return df.copy()


# Load the CSV file
if __name__ == "__main__":
    data_dir = "data"
    data_files = filter(
        lambda x: x.endswith("_games.csv"),
        os.listdir(data_dir),
    )
    raw_data = pd.concat(
        pd.read_csv(os.path.join(data_dir, file), encoding="latin1")
        for file in data_files
    )
    raw_data = raw_data.reset_index(drop=True)
    data_cleaned = clean_data(raw_data).dropna()
    print("Data cleaned shape:", data_cleaned.shape)
    # TODO: move this somewhere else
    # optimize_ema(data_cleaned)
    ema_span = 50
    data_features = create_ema_features(data_cleaned, ema_span)
    print("Data ema features shape:", data_features.shape)
    data_restructured = restructure_data(data_features)
    print("Data restructured shape:", data_restructured.shape)
    data_with_stats = add_seasonal_features(data_restructured)
    print("Data shape after adding seasonal features:", data_with_stats.shape)
    data_final = add_team_elo(data_with_stats)
    print("Data shape after adding team elo:", data_final.shape)
    print("ok")
    data_final.to_csv(f"data/features_17_24_ema_span={ema_span}.csv", index=False)
