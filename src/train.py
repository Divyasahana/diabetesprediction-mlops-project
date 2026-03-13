import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_data():
    data = pd.read_csv("data/diabetes.csv")
    return data


def preprocess_data(data):
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)


def train_model(X_train, y_train):

    model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return accuracy, precision, recall, f1


def main():

    mlflow.set_experiment("diabetes-prediction")

    with mlflow.start_run():

        data = load_data()

        X_train, X_test, y_train, y_test = preprocess_data(data)

        model = train_model(X_train, y_train)

        accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test)

        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)

        mlflow.log_param("model", "RandomForest")

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        os.makedirs("model", exist_ok=True)
        joblib.dump(model, "model/model.pkl")

        mlflow.sklearn.log_model(model, "model")


if __name__ == "__main__":
    main()
