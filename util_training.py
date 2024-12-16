from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import pandas as pd
from sklearn.linear_model import LogisticRegression
import xgboost as xgb


def get_cv_score(estimator, features: pd.DataFrame):
    label_encoder = LabelEncoder()
    features["FTR"] = label_encoder.fit_transform(features["FTR"])
    y = features["FTR"]
    # Prepare the data
    X = features[[col for col in features.columns if col.startswith("f_")]]
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    avg_score = cross_val_score(
        estimator, X, y, scoring="neg_log_loss", cv=TimeSeriesSplit(4)
    ).mean()
    return avg_score
