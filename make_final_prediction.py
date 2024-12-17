import pandas as pd
import pickle
from clean_data_and_create_features import (
    get_all_season_data,
    transform_data_full,
    clean_data,
)
from train_xgb_classifier import prepare_data


def restructure_test_data(df):
    new_data = []
    team_name_remapping = {
        "AFC Bournemouth": "Bournemouth",
        "Man City": "Manchester City",
        "Nottingham Forest": "Nott'ham Forest",
        "Newcastle": "Newcastle Utd",
        "Spurs": "Tottenham",
        "Man Utd": "Manchester Utd",
    }
    for _, row in df.iterrows():
        new_data.append(
            {
                "Date": pd.to_datetime(row["Date"]),
                "Team": team_name_remapping.get(row["HomeTeam"], row["HomeTeam"]),
                "Opp": team_name_remapping.get(row["AwayTeam"], row["AwayTeam"]),
                "Unnamed: 13": "@",
            }
        )
        new_data.append(
            {
                "Date": pd.to_datetime(row["Date"]),
                "Team": team_name_remapping.get(row["AwayTeam"], row["AwayTeam"]),
                "Opp": team_name_remapping.get(row["HomeTeam"], row["HomeTeam"]),
                "Unnamed: 13": "",
            }
        )
    return pd.DataFrame(new_data)


def make_final_prediction(model):
    raw_data = get_all_season_data("data")
    print(raw_data["Unnamed: 13"].value_counts())
    test_data = pd.read_csv("submission/sample-submission.csv")
    test_data = restructure_test_data(test_data)
    # Align smaller_df to match larger_df's columns
    test_data = test_data.reindex(columns=raw_data.columns)
    test_data = test_data.fillna(0)
    # Append the smaller DataFrame to the larger one
    df = pd.concat([raw_data, test_data], ignore_index=True)
    print(df.shape)
    df_features = transform_data_full(df)
    print(df_features.shape)
    X, y, encoder = prepare_data(df_features)
    print(X.shape, y.shape)
    y_pred = model.predict(X)
    y_pred = encoder.inverse_transform(y_pred)
    df_features["FTR"] = y_pred
    df_prediction = df_features[df_features["Date"] > "2025-01-01"][
        ["Date", "HomeTeam", "AwayTeam", "FTR"]
    ].reset_index(drop=True)
    df_prediction.to_csv("submission/final_submission.csv")
    return df_prediction


if __name__ == "__main__":
    saved_model_path = "models/xgb_model_best"
    model = pickle.load(open(saved_model_path, "rb"))
    prediction = make_final_prediction(model)
    print(prediction)
