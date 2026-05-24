import os
import json
import pandas as pd
import joblib


class Predictor:
    def __init__(self, model_path="models/xgb_pipeline.pkl", meta_path="models/model_meta.json"):
        self.model_path = model_path
        self.meta_path = meta_path

        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        else:
            self.model = None
            print(f"Модель не найдена по пути {self.model_path}")

    def predict(self, data: pd.DataFrame):
        if self.model is None:
            raise ValueError("Модель не загружена. Сначала обучите её с помощью retrainer.py")

        return self.model.predict(data)

    def get_meta(self) -> dict:
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"error": "Файл метаданных не найден"}