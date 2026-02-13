import pytest
from src.data_loader import load_data
from src.preprocess import preprocess_data


def test_preprocess_shapes():
     
    """
    Ensure that train and test splits
    have matching feature and target sizes.
    """
     
    X, y = load_data("data/diabetes.csv")
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]


def test_preprocess_split_consistency():

    """
    Ensure total samples after split equals original dataset.
    """

    X, y = load_data("data/diabetes.csv")
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    total_split = X_train.shape[0] + X_test.shape[0]
    assert total_split == X.shape[0]

    def test_preprocess_not_empty():

     """
     Ensure that train and test splits are not empty.
     """

    X, y = load_data("data/diabetes.csv")
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0

