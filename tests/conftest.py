import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FEATURES = ["f1", "f2", "f3"]


@pytest.fixture()
def model_path(tmp_path):
    """A tiny dummy model so the tests do not need the real dataset or model."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 3)), columns=FEATURES)
    y = (X["f1"] > 0).astype(int)
    scaler = StandardScaler().fit(X)
    model = LogisticRegression().fit(pd.DataFrame(scaler.transform(X), columns=FEATURES), y)
    path = tmp_path / "model.joblib"
    joblib.dump({"model": model, "scaler": scaler, "features": FEATURES,
                 "examples": {"DDoS": {"f1": 3, "f2": 0, "f3": 0}, "BENIGN": {"f1": -3, "f2": 0, "f3": 0}},
                 "metrics": {"F1": 1.0}}, path)
    return path
