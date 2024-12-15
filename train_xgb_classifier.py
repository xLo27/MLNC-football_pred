import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, train_test_split
import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import pickle
import matplotlib.pyplot as plt


def train_full(file_path):
    # Load the CSV file
    data = pd.read_csv(file_path, encoding="latin1").sort_values(
        by="Date", ascending=True
    )
    data["FTR"] = data["FTR"].astype("category")
    label_encoder = LabelEncoder()
    data["FTR"] = label_encoder.fit_transform(data["FTR"])
    y = data["FTR"]
    # Prepare the data
    X = data[[col for col in data.columns if col.startswith("f_")]]
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

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
            n_estimators=100,
            objective="multi:softmax",
            random_state=1000,
            enable_categorical=True,
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
            cv=TimeSeriesSplit(n_splits=4),
            n_jobs=-1,
            verbose=1,
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
        n_estimators=100,
        objective="multi:softmax",
        random_state=999,
        enable_categorical=True,
        **best_params,
    )
    print(y_train.value_counts())
    xgb_best.fit(X_train, y_train)
    pred = xgb_best.predict(X_test)
    print("predictions")
    print(pd.DataFrame(pred).value_counts())
    print(pd.DataFrame(y_test).value_counts())
    # error_rate = np.sum(pred != y_test) / y_test.shape[0]
    # print("Test error using softmax = {}".format(error_rate))

    # Evaluate the model
    accuracy = accuracy_score(y_test, pred)
    print(f"Accuracy: {accuracy}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
    ax = xgb.plot_importance(xgb_best, importance_type="weight", max_num_features=20)
    plt.show()


train_full(
    "C:/Users/super/Documents/UCL/CS/workspace/serious/MLNC-CW/data/ema_features_17_24_span=30.csv"
)
