import pandas as pd


def validate_dataset(path: str):
    """
    Validate the diabetes dataset before training.

    Checks:
    - Correct column names
    - No missing values
    - Dataset shape

    Args:
        path (str): Path to CSV dataset file
    """
    df = pd.read_csv(path)

    expected_columns = {
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
    }

    # Check column names
    assert set(df.columns) == expected_columns, (
        f"Unexpected columns found: {df.columns}"
    )

    # Check missing values
    assert df.isnull().sum().sum() == 0, "Dataset contains missing values!"

    # Check column count
    assert df.shape[1] == 9, "Dataset should have 9 columns."

    print("✅ Dataset validation passed successfully!")


# Optional: Run directly
if __name__ == "__main__":
    validate_dataset("data/diabetes.csv")
