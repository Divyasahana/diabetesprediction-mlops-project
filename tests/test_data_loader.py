from src.data_loader import load_data


def test_load_data():
    X, y = load_data("data/diabetes.csv")

    assert X.shape[0] > 0
    assert y.shape[0] > 0
    assert "Outcome" not in X.columns
