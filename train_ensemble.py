from collections import Counter
import pickle
import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn import (
    linear_model,
    tree,
    discriminant_analysis,
    naive_bayes,
    ensemble,
    gaussian_process,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.model_selection import (
    TimeSeriesSplit,
    cross_val_score,
    train_test_split,
)
from sklearn.metrics import (
    log_loss,
    classification_report,
    confusion_matrix,
    make_scorer,
    f1_score,
)
from imblearn.over_sampling import RandomOverSampler, SMOTE
from train_xgb_classifier import prepare_data

# def optimize_ema(data: pd.DataFrame):
#     scores = []
#     best_score = np.float16("inf")
#     best_span = 0
#     spans = range(1, 500, 10)
#     for i, span in enumerate(spans):
#         if i % 10 == 0:
#             print(f"Optimizing span {span}")
#         data_features = create_ema_features(data, span)
#         data_restructured = restructure_data(data_features)
#         score = get_cv_score(data_restructured)
#         scores.append(score)
#         if score * -1 < best_score:
#             best_score = score * -1
#             best_span = span
#     print(f"Best span: {best_span}, Best score: {best_score}")
#     # plot plt graph of scores

#     plt.plot(spans, -1 * pd.Series(scores))
#     plt.xlabel("Span")
#     plt.ylabel("Log Loss")
#     plt.show()


def gen_comparison_report(X_train, y_train, X_test, y_test, model_dict):
    reports = []
    for model in model_dict:
        classifier = model_dict[model]["model"]
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        class_report = classification_report(y_test, y_pred, output_dict=True)

        report = class_report["weighted avg"]
        report["Classifier"] = str(model)
        report["Accuracy"] = accuracy_score(y_test, y_pred)
        reports.append(report)

    df_combined = pd.DataFrame(reports)
    print(
        df_combined[
            ["Classifier", "Accuracy", "precision", "recall", "f1-score"]
        ].sort_values(by="f1-score", ascending=False)
    )
    return df_combined


def get_tuned_models_bayes(
    classifier_dict, X_train, y_train, cv_splits=5, scoring="neg_log_loss"
):
    best_models = {}
    for name, model_info in classifier_dict.items():

        model_save_path = f"models/{name}_tuned"
        param_save_path = f"models/{name}_best_params"
        try:
            model = pickle.load(open(model_save_path, "rb"))
            params = pickle.load(open(param_save_path, "rb"))
            print(f"Loaded {name}")
            best_models[name + "_tuned"] = {"model": model, "params": params}
        except FileNotFoundError as e:
            print(f"Tuning {name}")
            model = model_info["model"]()
            search = BayesSearchCV(
                model,
                search_spaces=model_info["search_space"],
                cv=TimeSeriesSplit(cv_splits),
                scoring=scoring,
                n_iter=60,
                n_jobs=-1,
                verbose=1,
            )
            search.fit(X_train, y_train)
            best_models[name + "_tuned"] = {
                "model": search.best_estimator_,
                "params": search.best_params_,
            }
            pickle.dump(search.best_estimator_, open(f"models/{name}_tuned", "wb"))
            pickle.dump(search.best_params_, open(f"models/{name}_best_params", "wb"))
    return best_models


if __name__ == "__main__":
    # Classifier dictionary with search spaces and random_state for consistency
    classifier_dict = {
        "LogisticRegressionCV": {
            "model": lambda: linear_model.LogisticRegressionCV(
                random_state=100, cv=TimeSeriesSplit(4), max_iter=500
            ),
            "search_space": {
                "Cs": Integer(1, 50),  # Number of regularization strengths
            },
        },
        "BernoulliNB": {
            "model": lambda: naive_bayes.BernoulliNB(),
            "search_space": {
                "alpha": Real(0.01, 10.0, prior="log-uniform"),  # Smoothing parameter
                "binarize": Real(0.0, 1.0),
            },
        },
        "GaussianNB": {
            "model": lambda: naive_bayes.GaussianNB(),
            "search_space": {
                "var_smoothing": Real(1e-9, 1e-6, prior="log-uniform")  # Stabilization
            },
        },
        "LinearDiscriminantAnalysis": {
            "model": lambda: discriminant_analysis.LinearDiscriminantAnalysis(),
            "search_space": {
                "tol": Real(1e-6, 1e-2, prior="log-uniform"),  # Tolerance
            },
        },
        "QuadraticDiscriminantAnalysis": {
            "model": lambda: discriminant_analysis.QuadraticDiscriminantAnalysis(),
            "search_space": {"reg_param": Real(0.0, 1.0)},  # Regularization parameter
        },
        "AdaBoostClassifier": {
            "model": lambda: ensemble.AdaBoostClassifier(
                random_state=1000, algorithm="SAMME"
            ),
            "search_space": {
                "n_estimators": Integer(50, 200),
                "learning_rate": Real(0.01, 1.0, prior="log-uniform"),
            },
        },
        "BaggingClassifier": {
            "model": lambda: ensemble.BaggingClassifier(random_state=1000),
            "search_space": {
                "n_estimators": Integer(10, 50),
                "max_samples": Real(0.5, 1.0),
                "max_features": Real(0.5, 1.0),
            },
        },
        "ExtraTreesClassifier": {
            "model": lambda: ensemble.ExtraTreesClassifier(random_state=1000),
            "search_space": {
                "n_estimators": Integer(50, 200),
                "max_depth": Integer(3, 20),
                "min_samples_split": Integer(2, 10),
            },
        },
        "GradientBoostingClassifier": {
            "model": lambda: ensemble.GradientBoostingClassifier(
                random_state=1000, n_iter_no_change=20
            ),
            "search_space": {
                "n_estimators": Integer(50, 200),
                "learning_rate": Real(0.01, 0.3, prior="log-uniform"),
                "max_depth": Integer(3, 10),
                "subsample": Real(0.5, 1.0),
            },
        },
        "RandomForestClassifier": {
            "model": lambda: ensemble.RandomForestClassifier(random_state=1000),
            "search_space": {
                "n_estimators": Integer(50, 200),
                "max_depth": Integer(3, 20),
                "min_samples_split": Integer(2, 10),
            },
        },
        "GaussianProcessClassifier": {
            "model": lambda: gaussian_process.GaussianProcessClassifier(
                random_state=1000
            ),
            "search_space": {
                "max_iter_predict": Integer(50, 200),
            },
        },
        "XGBClassifier": {
            "model": lambda: xgb.XGBClassifier(
                random_state=999,
                enable_categorical=True,
                objective="multi:softmax",
            ),
            "search_space": {
                "n_estimators": Integer(50, 200),
                "learning_rate": Real(0.01, 0.3, prior="log-uniform"),
                "max_depth": Integer(3, 10),
                "subsample": Real(0.5, 1.0),
                "colsample_bytree": Real(0.5, 1.0),
            },
        },
    }
    data = pd.read_csv("data/features_17_24_full_ema_span=30.csv").sort_values(
        by="Date", ascending=True
    )
    X, y, encoder = prepare_data(data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    oversample = True
    f1_scorer = make_scorer(f1_score, average="weighted")
    neg_log_loss_scorer = make_scorer(
        log_loss,
        greater_is_better=False,
    )
    scoring = f1_scorer

    if oversample:
        # oversampler = ADASYN(random_state=1000)
        oversampler = RandomOverSampler(random_state=1000)
        # oversampler = SMOTE(k_neighbors=10, random_state=1000)
        X_res, y_res = oversampler.fit_resample(X_train, y_train)
        X_train, y_train = X_res, y_res
    print("Class distribution after oversampling:", Counter(y_train))
    tuned_model_dict = get_tuned_models_bayes(
        classifier_dict, X_train, y_train, scoring=scoring
    )
    df_report_tuned = gen_comparison_report(
        X_train, y_train, X_test, y_test, tuned_model_dict
    )
    untuned_model_dict = {
        model_name: {"model": classifier_dict[model_name]["model"]()}
        for model_name in classifier_dict
    }
    df_report_untuned = gen_comparison_report(
        X_train, y_train, X_test, y_test, untuned_model_dict
    )
    df_report = pd.concat([df_report_tuned, df_report_untuned])
    df_report.to_csv(
        f"models/model_results_tuning={str(scoring)}_oversampled={oversample}.csv"
    )
