from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import pandas as pd
from sklearn.linear_model import LogisticRegression


def get_cv_indices_forward_chain(data, k_folds):
    """
    This function returns the indices of the training and test sets for each fold of the cross-validation
    :param data: the data to be split
    :param k_folds: the number of folds
    :return: a list of tuples where each tuple contains the indices of the training and test sets for a fold
    """
    n = len(data)
    indices = list(range(n))
    fold_size = n // (k_folds + 1)
    train_test_indices = []
    for i in range(1, k_folds + 1):
        test_indices = indices[i * fold_size : (i + 1) * fold_size]
        train_indices = indices[: i * fold_size]
        train_test_indices.append((train_indices, test_indices))
    return train_test_indices


def get_cv_score(features: pd.DataFrame):
    label_encoder = LabelEncoder()
    features["FTR"] = label_encoder.fit_transform(features["FTR"])
    y = features["FTR"]
    # Prepare the data
    X = features[[col for col in features.columns if col.startswith("f_")]]
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    lr = LogisticRegression(max_iter=500)
    avg_score = cross_val_score(
        lr, X, y, scoring="neg_log_loss", cv=get_cv_indices_forward_chain(X, 3)
    ).mean()
    return avg_score
