import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from skopt import BayesSearchCV
from skopt.space import Real, Integer

# Load the CSV file
file_path = "C:/Users/super/Documents/UCL/CS/workspace/serious/MLNC-CW/transformed2.csv"
data = pd.read_csv(file_path, encoding="latin1")
label_encoder = LabelEncoder()
data["FTR"] = label_encoder.fit_transform(data["FTR"])
print(data["FTR"].value_counts())
# Prepare the data
X = data.drop(columns=["FTR"])
X = data[["HTWinStreak3", "HTWinStreak5", "HTLossStreak3", "HTLossStreak5"]]
y = data["FTR"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

xg_train = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
xg_test = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)
# setup parameters for xgboost
param = {}
# use softmax multi-class classification
param["objective"] = "multi:softmax"
# scale weight of positive examples
param["eta"] = 0.1
param["max_depth"] = 6
param["nthread"] = 4
param["num_class"] = 4


watchlist = [(xg_train, "train"), (xg_test, "test")]
num_round = 5
bst = xgb.train(param, xg_train, num_round, watchlist)

# Make predictions
pred = bst.predict(xg_test)
error_rate = np.sum(pred != y_test) / y_test.shape[0]
print("Test error using softmax = {}".format(error_rate))

# Evaluate the model
accuracy = accuracy_score(y_test, pred)
print(f"Accuracy: {accuracy}")
