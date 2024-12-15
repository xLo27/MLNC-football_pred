import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
import os
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler


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

    # scaler = StandardScaler()
    # X = pd.DataFrame(scaler.fit_transform(data), columns=X.columns)
    # label_encoder = LabelEncoder()
    # data["FTR"] = label_encoder.fit_transform(data["FTR"])
    # print(data["FTR"].value_counts())

    data = data.dropna(how="all")
    print("After dropping NaN:", len(data), len(data.columns))
    data = data.drop_duplicates()
    print("After dropping duplicate rows:", len(data), len(data.columns))
    data = drop_duplicate_columns(data)
    print("After dropping duplicate columns:", len(data), len(data.columns))
    data["HomeGame"] = ~data["Unnamed: 13"].isna()
    data = data.drop(columns=["Unnamed: 13", "Match Report", "Comp", "Rk"])
    print("After dropping columns:", len(data), len(data.columns))

    data["Date"] = pd.to_datetime(data["Date"])
    # rename GA and GF based on whether homegame = true or false

    data["GoalsAway"] = data.apply(
        lambda row: row["GA"] if row["HomeGame"] else row["GF"], axis=1
    )
    data["GoalsHome"] = data.apply(
        lambda row: row["GF"] if row["HomeGame"] else row["GA"], axis=1
    )

    data = data.drop(columns=["Result", "GA", "GF"])
    # add FTR column, with H/D/A based on GoalsHome and GoalsAway
    data["FTR"] = data.apply(
        lambda row: (
            "H"
            if row["GoalsHome"] > row["GoalsAway"]
            else "A" if row["GoalsHome"] < row["GoalsAway"] else "D"
        ),
        axis=1,
    )
    # print to file ambiguous named columns (sorted)
    lines = []
    with open("ambiguous_columns.txt", "w") as f:
        for column in data.columns:
            if "." in column:
                lines.append(column)
        lines.sort()
        for line in lines:
            f.write(line + "\n")
    return data.copy()


def create_ema_features(data: pd.DataFrame, span):
    non_feature_cols = ["Date", "HomeGame", "FTR", "Team", "Opp"]
    df_ema_features = data[non_feature_cols].copy().sort_values(by=["Date"])
    feature_names = data.drop(columns=non_feature_cols).columns
    ema_features = []
    for feature_name in feature_names:
        feature_ema = data.groupby("Team")[feature_name].transform(
            lambda row: row.ewm(span=span, min_periods=2).mean()
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


def optimize_ema_span(data: pd.DataFrame):
    pass


# Load the CSV file
# file_path = "C:/Users/super/Documents/UCL/CS/workspace/serious/MLNC-CW/transformed2.csv"
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
    data_features = create_ema_features(data_cleaned, 5)
    data_restructured = restructure_data(data_features)

    with open("restructured_dtypes.txt", "w") as f:
        f.write(str(data_restructured.dtypes.to_string()))
        f.write(data_restructured.head().to_string())

    print(data_restructured)
    data_restructured.to_csv("data/ema_features_17_24.csv", index=False)
