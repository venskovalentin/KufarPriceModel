import os
import glob
import json
import joblib
import mlflow
import mlflow.sklearn

from datetime import datetime

import numpy as np
import pandas as pd

from xgboost import XGBRegressor

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

from src.preprocessing import FeatureExtractor


def retrainer(folder_path=r'..\data\raw', log_target=False):

    files = glob.glob(os.path.join(folder_path, '*.csv'))

    if not files:
        raise FileNotFoundError(
            f'CSV файлы не найдены в {folder_path}'
        )

    latest_file = max(files, key=os.path.getctime)

    print(f"Загружаем файл: {latest_file}")

    df = pd.read_csv(latest_file)

    df = df[df["price_byn"] > 30]

    filename = os.path.basename(latest_file)
    time_str = filename.split("_")

    access_time = pd.Timestamp(
        year=int(time_str[0][0:4]),
        month=int(time_str[0][4:6]),
        day=int(time_str[0][6:8]),
        hour=int(time_str[1][0:2]),
        minute=int(time_str[1][2:4]),
        tz='Europe/Moscow'
    )

    target = "price_byn"

    X = df.drop(columns=[target])
    y = df[target]

    if log_target:
        y = np.log1p(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    xgb_pipeline = Pipeline([
        ('extractor', FeatureExtractor(
            access_time,
            GBM_XGB=True
        )),

        ('regressor', XGBRegressor(
            n_estimators=1179,
            learning_rate=0.020428215357261675,
            max_depth=9,
            min_child_weight=9,
            subsample=0.7243929286862649,
            colsample_bytree=0.7433862914177091,
            reg_alpha=1.654490124263246,
            reg_lambda=1.5720251525742128,

            objective='reg:absoluteerror',
            random_state=42,
            enable_categorical=True,
            verbosity=0,
            n_jobs=-1
        ))
    ])

    mlflow.set_experiment("kufar-notebook")

    with mlflow.start_run():

        xgb_pipeline.fit(X_train, y_train)

        y_pred = xgb_pipeline.predict(X_test)

        if log_target:
            y_test = np.expm1(y_test)
            y_pred = np.expm1(y_pred)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f"R²: {r2:.4f}")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")

        mlflow.log_param("log_target", log_target)
        mlflow.log_param("n_estimators", 1179)
        mlflow.log_param("learning_rate", 0.020428215357261675)
        mlflow.log_param("max_depth", 9)
        mlflow.log_param("min_child_weight", 9)
        mlflow.log_param("subsample", 0.7243929286862649)
        mlflow.log_param("colsample_bytree", 0.7433862914177091)
        mlflow.log_param("reg_alpha", 1.654490124263246)
        mlflow.log_param("reg_lambda", 1.5720251525742128)


        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)

        os.makedirs("models", exist_ok=True)

        model_path = "models/xgb_pipeline.pkl"

        joblib.dump(xgb_pipeline, model_path)

        mlflow.sklearn.log_model(
            sk_model=xgb_pipeline,
            artifact_path="model"
        )

        meta = {
            "trained_at": datetime.now().isoformat(),
            "n_samples": len(X_train),
            "log_target": log_target,
            "metrics": {
                "r2": r2,
                "mae": mae,
                "rmse": rmse
            }
        }

        with open("models/model_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        mlflow.log_artifact("models/model_meta.json")

    return xgb_pipeline