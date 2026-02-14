import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.data_loader import load_data
from src.preprocess import preprocess_data


def main():
    mlflow.set_experiment("diabetes-baseline")

    with mlflow.start_run(run_name="logreg-baseline"):
        X, y = load_data("data/diabetes.csv")
        X_train, X_test, y_train, y_test = preprocess_data(X, y)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_metric("accuracy", accuracy)

        mlflow.sklearn.log_model(model, "model")

        print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
