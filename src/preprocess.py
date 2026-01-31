from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple
import pandas as pd

def preprocess_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> Tuple:
    """
    Split dataset and apply standard scaling.

    Args:
        X: Feature dataframe
        y: Target labels
        test_size: Fraction of data used for testing
        random_state: Random seed for reproducibility

    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test
    """

# Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

# Initialize standard scaler
    scaler = StandardScaler()

# Scale training and testing features
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test
