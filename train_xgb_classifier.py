import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from collections import Counter
import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import pickle
import matplotlib.pyplot as plt


def prepare_data(data: pd.DataFrame):
    data["FTR"] = data["FTR"].astype("category")
    label_encoder = LabelEncoder()
    label_encoder.fit(data["FTR"])
    data["FTR"] = label_encoder.transform(data["FTR"])
    y = data["FTR"]
    # Prepare the data
    object_columns = [
        "f_HM1",
        "f_HM2",
        "f_HM3",
        "f_HM4",
        "f_HM5",
        "f_AM1",
        "f_AM2",
        "f_AM3",
        "f_AM4",
        "f_AM5",
        "f_HTFormPtsStr",
        "f_ATFormPtsStr",
    ]
    # elo_cols = [col for col in data.columns if col.startswith("f_el")]
    # pi_cols = [col for col in data.columns if col.startswith("f_pi")]
    # X = all non object columns
    data = data.drop(columns=object_columns)
    X = data[[col for col in data.columns if col.startswith("f_")]]
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X, y, label_encoder


def train_full(file_path):
    # Load the CSV file
    data = pd.read_csv(file_path).sort_values(by="Date", ascending=True)
    print(data.shape)
    X, y, encoder = prepare_data(data)

    # print("Resampled dataset shape %s" % Counter(y_res))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    # Resample the training data
    # oversampler = SMOTE(k_neighbors=10, random_state=1000)
    oversampler = RandomOverSampler(random_state=1000)
    # oversampler = ADASYN(random_state=1000)
    X_res, y_res = oversampler.fit_resample(X_train, y_train)
    X_train, y_train = X_res, y_res
    # print shapes
    print("X_TRAIN", X_train.shape)
    print("Y_TRAIN", Counter(y_train))
    print("Y_TEST", Counter(y_test))
    save_path = "models/xgb_params_full.pkl"
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
    # save model
    with open("models/xgb_model_best", "wb") as f:
        pickle.dump(xgb_best, f)
    y_pred = xgb_best.predict(X_test)
    print("predictions")
    print(pd.DataFrame(y_pred).value_counts())
    print(pd.DataFrame(y_test).value_counts())
    # error_rate = np.sum(pred != y_test) / y_test.shape[0]
    # print("Test error using softmax = {}".format(error_rate))

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    report = classification_report(y_test, y_pred)
    print(report)
    ax = xgb.plot_importance(xgb_best, importance_type="weight", max_num_features=20)
    plt.show()


if __name__ == "__main__":
    train_full(
        "C:/Users/super/Documents/UCL/CS/workspace/serious/MLNC-CW/data/features_17_24_full_ema_span=30.csv"
    )
