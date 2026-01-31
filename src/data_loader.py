import pandas as pd
from pathlib import Path
from typing import Tuple

def load_data(data_path: str | Path) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load diabetes dataset from CSV file.

    Args:
        data_path: Path to the CSV dataset.

    Returns:
        X (pd.DataFrame): Feature dataframe
        y (pd.Series): Target variable
    """

# Convert input path to Path object
    data_path = Path(data_path)
 # Read CSV file  
    df = pd.read_csv(data_path)

# Separate features and target column
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    return X, y