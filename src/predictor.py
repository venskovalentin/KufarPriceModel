import json
import os

import joblib
import numpy as np
import pandas as pd


class Predictor:
    """Load a trained pipeline and run inference on raw Kufar-style rows."""

    def __init__(self, model_path="models/xgb_pipeline.pkl", meta_path="models/model_meta.json"):
        self.model_path = model_path
        self.meta_path = meta_path

        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        else:
            self.model = None
            print(f"Model not found at {self.model_path}")

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Predict prices for a batch of rows."""
        if self.model is None:
            raise ValueError("Model is not loaded")

        preds = self.model.predict(data)

        meta = self.get_meta()
        if meta.get("log_target", False):
            preds = np.expm1(preds)

        return np.asarray(preds)

    def predict_single(self, row: dict) -> dict:
        """Return ``{price, lower, upper}`` for a single row.

        The interval is a quick ``price ± MAE`` approximation — good enough for
        a UI hint, not a calibrated confidence interval.
        """
        df = pd.DataFrame([row])
        price = float(self.predict(df)[0])

        meta = self.get_meta()
        mae = meta.get("metrics", {}).get("mae", price * 0.1)

        return {
            "price": round(price, 2),
            "lower": round(price - mae, 2),
            "upper": round(price + mae, 2),
        }

    def get_meta(self) -> dict:
        """Read ``model_meta.json`` from disk on every call.

        Reading on demand means the GUI sees fresh metrics after a retrain
        without having to be restarted.
        """
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"error": "Metadata file not found"}
