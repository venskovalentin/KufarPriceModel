import pandas as pd
import numpy as np
import glob
import os
import warnings
import logging
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
import sys

sys.path.append('..')
from src.preprocessing import FeatureExtractor

folder_path = r'..\data\raw'
files = glob.glob(folder_path + '/*csv')
latest_file = max(files, key=os.path.getctime)

print(f"Загружаем файл: {latest_file}")

df = pd.read_csv(latest_file)
df = df[df["price_byn"] > 30]

time_str = latest_file[12:].split(sep="_")

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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

xgb_pipeline = Pipeline([
    ('extractor', FeatureExtractor(access_time, GBM_XGB=True)),
    ('regressor', XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=7,
        objective='reg:absoluteerror',
        random_state=42,
        enable_categorical=True,
        verbosity=0  # тишина
    ))
])

xgb_pipeline.fit(X_train, y_train)


y_pred = xgb_pipeline.predict(X_test)

print(f"R²: {r2_score(y_test, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")