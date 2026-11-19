import pytest

from app.predictor import Predictor


def test_predicts_both_classes(model_path):
    p = Predictor(model_path)
    assert p.loaded
    assert p.predict({"f1": 3, "f2": 0, "f3": 0})["label"] == "DDoS"
    assert p.predict({"f1": -3, "f2": 0, "f3": 0})["label"] == "BENIGN"


def test_missing_feature_raises(model_path):
    p = Predictor(model_path)
    with pytest.raises(ValueError):
        p.predict({"f1": 1})


def test_missing_model_file(tmp_path):
    assert Predictor(tmp_path / "nope.joblib").loaded is False
