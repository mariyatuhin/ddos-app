from fastapi.testclient import TestClient

from app.main import app


def test_health_and_predict(model_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
        assert client.get("/").status_code == 200
        assert client.get("/features").json()["features"] == ["f1", "f2", "f3"]

        r = client.post("/predict", json={"features": {"f1": 3, "f2": 0, "f3": 0}})
        assert r.status_code == 200 and r.json()["label"] == "DDoS"

        r = client.post("/predict", json={"features": {"f1": 3}})
        assert r.status_code == 422


def test_no_model_gives_503(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    with TestClient(app) as client:
        assert client.get("/health").json()["model_loaded"] is False
        assert client.post("/predict", json={"features": {}}).status_code == 503
