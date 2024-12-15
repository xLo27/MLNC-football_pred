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
            if row["GF"] == row["GA"]
            else (
                "H"
                if (
                    row["GF" if row["HomeGame"] else "GA"]
                    > row["GA" if row["HomeGame"] else "GF"]
                )
                else "A"
            )
        ),
        axis=1,
    )
    name_map = {"1/3.1": "Carries_1/3", "Att": "Att_passes", "Att.1": "Att_passes_"}
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
    df_ema_features = data[non_feature_cols].copy().sort_values(by=["Date"])
    feature_names = data.drop(columns=non_feature_cols).columns
    ema_features = []
    for feature_name in feature_names:
        feature_ema = data.groupby("Team")[feature_name].transform(
            lambda row: row.ewm(span=span, min_periods=2).mean().shift(1)
        )
        ema_features.append(pd.Series(feature_ema, name=feature_name))
    df_ema_features = pd.concat(
        [df_ema_features, pd.concat(ema_features, axis=1)], axis=1
    )

    return df_ema_features.copy()


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
    data_merged = data_merged.rename(
        columns={
            col: "f_" + col
            for col in data_merged.columns
            if col not in non_feature_cols
        }
    )
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
    data_cleaned = clean_data(raw_data)
    print(data_cleaned["FTR"].value_counts())
    optimize_ema(data_cleaned)
    ema_span = 10
    data_features = create_ema_features(data_cleaned, ema_span)
    data_restructured = restructure_data(data_features)
    data_restructured.to_csv(
        f"data/ema_features_17_24_span={ema_span}.csv", index=False
    )
    print(data_restructured.shape)
