import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("titanic.csv")
data.info()

#print(data.isnull().sum())


# Preprocess the data (drop columns that won't be used and handle missing values)
def preprocess_data(df):
    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)
    df["Embarked"].fillna("S", inplace=True)
    df.drop(columns="Embarked", inplace=True)

    # Fill missing ages based on Pclass median
    fill_missing_ages(df)

    # Convert gender to numerical values
    df["Sex"] = df["Sex"].map({"male": 1, "female": 0})

    # Feature Engineering
    df["FamilySize"] = df["SibSP"] + df["Parch"]
    df["IsAlone"] = np.where(df["FamilySize"] == 0, 1, 0)
    df["FareBin"] = pd.qcut(df["Fare"], 4, labels=False)
    df["AgeBin"] = pd.cut(df["Age"], bins=[0, 12, 20, 40, 60, np.inf], labels=False)

    return df

# Fill missing ages
def fill_missing_ages(df):
    age_fill_map = {}
    for pclass in df["Pclass"].unique():
        if pclass not in age_fill_map:
            age_fill_map[pclass] = df[df["Pclass"] == pclass]["Age"].median()
    df["Age"] = df.apply(lambda row: age_fill_map[row["Pclass"]] if pd.isnull(row["Age"]) else row["Age"], axis=1)


data = preprocess_data(data)

# Dropping survived column to learn our model
X = data.drop(columns=["Survived"])

# Target values
y = data["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# ML Preprocessing
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter tuning - KNN
def tune_model(X_train, y_train):
    param_grid = {
        ""
    }
