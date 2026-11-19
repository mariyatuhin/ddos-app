"""Loads the trained model bundle and turns raw flow features into a prediction."""
from pathlib import Path

import joblib
import pandas as pd


class Predictor:
    def __init__(self, path):
        self.path = Path(path)
        self.loaded = False
        self.features = []
        self.examples = {}
        self.metrics = {}
        if self.path.exists():
            bundle = joblib.load(self.path)
            self.model = bundle["model"]
            self.scaler = bundle["scaler"]
            self.features = list(bundle["features"])
            self.examples = bundle.get("examples", {})
            self.metrics = bundle.get("metrics", {})
            self.loaded = True

    def predict(self, values: dict) -> dict:
        missing = [f for f in self.features if f not in values]
        if missing:
            raise ValueError(f"Missing features: {', '.join(missing)}")

        row = pd.DataFrame([[values[f] for f in self.features]], columns=self.features)
        scaled = pd.DataFrame(self.scaler.transform(row), columns=self.features)
        p_benign, p_ddos = self.model.predict_proba(scaled)[0]
        return {
            "label": "DDoS" if p_ddos >= 0.5 else "BENIGN",
            "prob_ddos": round(float(p_ddos), 4),
            "prob_benign": round(float(p_benign), 4),
        }
