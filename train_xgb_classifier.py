import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import pickle
import matplotlib.pyplot as plt
from model_training import get_cv_indices_forward_chain


def train_classifier():
    # Load the CSV file
    file_path = (
        "C:/Users/super/Documents/UCL/CS/workspace/serious/MLNC-CW/transformed2.csv"
    )
    data = pd.read_csv(file_path, encoding="latin1")
    data.dropna(inplace=True)
    label_encoder = LabelEncoder()
    data["FTR"] = label_encoder.fit_transform(data["FTR"])
    y = data["FTR"]
    # Prepare the data
    X = data
    X = X.select_dtypes(exclude=["object"]).copy()
    X = X.drop(columns=["FTR", "FTHG", "FTAG", "HTHG", "HTAG"])
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    print(y.value_counts())

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    save_path = "models/best_xgb_params.pkl"
    best_params = None
    # check if save path exists
    try:
        with open(save_path, "rb") as f:
            best_params = pickle.load(f)
    except FileNotFoundError:
        xgb_c = xgb.XGBClassifier(
            n_estimators=100, objective="multi:softmax", random_state=1000
        )
        # Define search space
        search_space = {
            "max_depth": Integer(3, 10),
            "learning_rate": Real(0.01, 0.3, prior="log-uniform"),
            "subsample": Real(0.5, 1.0),
            "colsample_bytree": Real(0.5, 1.0),
        }
        # Perform Bayesian optimization
        bayes_search = BayesSearchCV(
            estimator=xgb_c,
            search_spaces=search_space,
            n_iter=25,
            cv=3,
            n_jobs=-1,
            verbose=2,
        )
        bayes_search.fit(X_train, y_train)

        # Print best parameters
        print(f"Best parameters: {bayes_search.best_params_}")
        print(f"Best score: {bayes_search.best_score_}")
        best_params = bayes_search.best_params_
        # Save the best parameters to a file
        with open(save_path, "wb") as f:
            pickle.dump(best_params, f)

    # Make predictions
    xgb_best = xgb.XGBClassifier(
        n_estimators=100, objective="multi:softmax", random_state=42, **best_params
    )
    xgb_best.fit(X_train, y_train)
    pred = xgb_best.predict(X_test)
    print(pd.DataFrame(pred).value_counts())
    error_rate = np.sum(pred != y_test) / y_test.shape[0]
    print("Test error using softmax = {}".format(error_rate))

    # Evaluate the model
    accuracy = accuracy_score(y_test, pred)
    print(f"Accuracy: {accuracy}")

    ax = xgb.plot_importance(xgb_best, importance_type="weight")
    plt.show()


def train_full(file_path):
    # Load the CSV file
    data = pd.read_csv(file_path, encoding="latin1").sort_values(
        by="Date", ascending=True
    )
    print(data["FTR"].value_counts())
    label_encoder = LabelEncoder()
    data["FTR"] = label_encoder.fit_transform(data["FTR"])
    y = data["FTR"]
    # Prepare the data
    X = data[[col for col in data.columns if col.startswith("f_")]]
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    print(y.value_counts())

    # Split the data into training and testing sets
    # split 80-20 first 80% for training, 20% for testing
    X_train, X_test = X[: int(0.8 * len(X))], X[int(0.8 * len(X)) :]
    y_train, y_test = y[: int(0.8 * len(y))], y[int(0.8 * len(y)) :]

    save_path = "models/full_data_params.pkl"
    best_params = None
    # check if save path exists
    try:
        with open(save_path, "rb") as f:
            best_params = pickle.load(f)
    except FileNotFoundError:
        xgb_c = xgb.XGBClassifier(
            n_estimators=100, objective="multi:softmax", random_state=1000
        )
        # Define search space
        search_space = {
            "max_depth": Integer(3, 10),
            "learning_rate": Real(0.01, 0.3, prior="log-uniform"),
            "subsample": Real(0.5, 1.0),
            "colsample_bytree": Real(0.5, 1.0),
        }
        # Perform Bayesian optimization
        bayes_search = BayesSearchCV(
            estimator=xgb_c,
            search_spaces=search_space,
            n_iter=25,
            cv=get_cv_indices_forward_chain(X_train, 3),
            n_jobs=-1,
            verbose=2,
        )
        bayes_search.fit(X_train, y_train)

        # Print best parameters
        print(f"Best parameters: {bayes_search.best_params_}")
        print(f"Best score: {bayes_search.best_score_}")
        best_params = bayes_search.best_params_
        # Save the best parameters to a file
        with open(save_path, "wb") as f:
            pickle.dump(best_params, f)

    # Make predictions
    xgb_best = xgb.XGBClassifier(
        n_estimators=100, objective="multi:softmax", random_state=42, **best_params
    )
    xgb_best.fit(X_train, y_train)
    pred = xgb_best.predict(X_test)
    print(pd.DataFrame(pred).value_counts())
    error_rate = np.sum(pred != y_test) / y_test.shape[0]
    print("Test error using softmax = {}".format(error_rate))

    # Evaluate the model
    accuracy = accuracy_score(y_test, pred)
    print(f"Accuracy: {accuracy}")

    ax = xgb.plot_importance(xgb_best, importance_type="weight", max_num_features=20)
    plt.show()


train_full(
    "C:/Users/super/Documents/UCL/CS/workspace/serious/MLNC-CW/data/ema_features_17_24_span=10.csv"
)
