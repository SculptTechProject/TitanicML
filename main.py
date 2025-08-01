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

# print(data.isnull().sum())


# Preprocess the data (drop columns that won't be used and handle missing values)
def preprocess_data(df):
    # Drop columns we don't use
    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin", "Embarked"])

    # Fill missing values in 'Age' based on Pclass median
    fill_missing_ages(df)

    # Fill missing values in 'Fare' with median
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    # Map 'Pclass' to categorical values
    df["Sex"] = df["Sex"].map({"male": 1, "female": 0}).astype(int)

    # Feature engineering
    df["FamilySize"] = df["SibSp"] + df["Parch"]
    df["IsAlone"] = np.where(df["FamilySize"] == 0, 1, 0)

    # Binning 'Fare' and 'Age'
    df["FareBin"] = pd.qcut(df["Fare"], 4, labels=False)
    df["AgeBin"] = pd.cut(df["Age"], bins=[0, 12, 20, 40, 60, np.inf], labels=False)

    df["FareBin"] = df["FareBin"].astype(int)
    df["AgeBin"] = df["AgeBin"].astype(int)

    return df


# Fill missing ages
def fill_missing_ages(df):
    age_fill_map = {}
    for pclass in df["Pclass"].unique():
        if pclass not in age_fill_map:
            age_fill_map[pclass] = df[df["Pclass"] == pclass]["Age"].median()
    df["Age"] = df.apply(
        lambda row: (
            age_fill_map[row["Pclass"]] if pd.isnull(row["Age"]) else row["Age"]
        ),
        axis=1,
    )


data = preprocess_data(data)

# Dropping survived column to learn our model
X = data.drop(columns=["Survived", "Sex"])

# Target values
y = data["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ML Preprocessing
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Hyperparameter tuning - KNN
def tune_model(X_train, y_train):
    param_grid = {
        "n_neighbors": range(1, 21),
        "metric": ["euclidean", "manhattan", "chebyshev", "minkowski"],
        "weights": ["uniform", "distance"],
    }

    model = KNeighborsClassifier()
    grid_search = GridSearchCV(
        model, param_grid=param_grid, cv=5, scoring="accuracy", verbose=1, n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


best_model = tune_model(X_train, y_train)


# Evaluate the model
def evaluate_model(model, X_test, y_test):
    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)
    conf_matrix = confusion_matrix(y_test, prediction)
    return accuracy, conf_matrix


accuracy, matrix = evaluate_model(best_model, X_test, y_test)

print(f"Best Model Accuracy: {accuracy*100:.2f}%")
print("Confusion Matrix:")
print(matrix)

def plot_model(matrix):
    plt.figure(figsize=(8,6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Not Survived", "Survived"],
                yticklabels=["Not Survived", "Survived"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

plot_model(matrix)