import os
import json
import pandas as pd
import joblib
import numpy as np  # Добавлен импорт numpy


class Predictor:
    def __init__(self, model_path="models/xgb_pipeline.pkl", meta_path="models/model_meta.json"):
        self.model_path = model_path
        self.meta_path = meta_path

        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        else:
            self.model = None
            print(f"Модель не найдена по пути {self.model_path}")

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError(
            )

        preds = self.model.predict(data)

        meta = self.get_meta()
        if meta.get("log_target", False):
            preds = np.expm1(preds)

        return np.asarray(preds)

    def predict_single(self, row: dict) -> dict:

        df = pd.DataFrame([row])

        price = float(self.predict(df)[0])

        meta = self.get_meta()
        mae = meta.get("metrics", {}).get("mae", price * 0.1)

        lower = price - mae
        upper = price + mae

        return {
            "price": round(price, 2),
            "lower": round(lower, 2),
            "upper": round(upper, 2),
        }

    def get_meta(self) -> dict:
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"error": "Файл метаданных не найден"}