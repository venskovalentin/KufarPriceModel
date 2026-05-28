# KufarPriceModel — Dynamic Pricing Optimizer для рынка б/у ноутбуков

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/UI-PyQt6-41cd52.svg)](https://riverbankcomputing.com/software/pyqt/)
[![Model](https://img.shields.io/badge/model-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)

End-to-end pet-проект: парсим объявления о ноутбуках с белорусской площадки
[Kufar](https://www.kufar.by), обучаем градиентный бустинг прогнозировать
справедливую цену и через десктоп-приложение помогаем находить недооценённые
лоты.

> Метрики последней модели (на ~8 000 объявлений): **R² = 0.756, MAE ≈ 396 BYN,
> RMSE ≈ 901 BYN, MAPE=47%, MedAE**, 

---

## Возможности

- **📊 Market Analysis** — фильтры (бренд / процессор / состояние / тип хранилища /
  диапазон цен), гистограмма цен, таблица из 200 свежих лотов.
- **🎯 Price Prediction** — форма ввода характеристик ноутбука, прогноз цены
  с доверительным интервалом, сравнение со «своей ценой».
- **📈 Analytics** — карточки метрик MAE / RMSE / R² / размер обучающей выборки,
  feature importance из XGBoost, дата последнего обучения.
- **🔄 Retrain в один клик** — фоновый поток парсит свежие данные с Kufar,
  переобучает модель, логирует запуск в MLflow и обновляет UI.

## Архитектура

```
KufarPriceModel/
├── src/
│   ├── kufar_parser.py     # Сбор объявлений через API Kufar
│   ├── preprocessing.py    # FeatureExtractor (sklearn-compatible)
│   │                       # — 8 regex-экстракторов из текста объявления
│   ├── retrainer.py        # Тренировка XGBoost + MLflow tracking
│   ├── predictor.py        # Инференс + доверительный интервал
│   ├── market_analyzer.py  # Аналитика рынка + поиск underpriced
│   └── app.py              # PyQt6 GUI (3 вкладки + QThread retrain)
├── notebooks/              # EDA, сравнение моделей, подбор гиперпараметров
├── data/raw/               # CSV-снапшоты рынка (gitignored)
├── models/                 # xgb_pipeline.pkl + model_meta.json (gitignored)
└── mlruns/                 # MLflow file backend (gitignored)
```

Слои разделены по ответственности: сбор → препроцессинг → обучение →
инференс → аналитика → UI. Препроцессор зашит внутрь sklearn-Pipeline,
поэтому train-serve skew исключён.

## Стек

`Python 3.11+` · `XGBoost` · `scikit-learn` · `MLflow` · `PyQt6` · `matplotlib`
· `pandas` · `requests`

В EDA-ноутбуках также сравниваются `CatBoost` / `LightGBM` и подбираются
гиперпараметры через randomsearch

## Установка

```bash
git clone https://github.com/venskovalentin/KufarPriceModel.git
cd KufarPriceModel
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Запуск

### 1. Собрать первый снапшот рынка

```bash
python -m src.kufar_parser
```

Создаст `data/raw/YYYYMMDD_HHMM_price_data.csv` (~10–12 тыс. объявлений за
30–60 секунд). API Kufar доступен из РБ/РФ; из-за границы может понадобиться
VPN.

### 2. Обучить модель

```bash
python -m src.retrainer
```

Запишет `models/xgb_pipeline.pkl` + `models/model_meta.json` и залогирует
запуск в `mlruns/` под именем эксперимента `kufar-notebook`.

### 3. Запустить GUI

```bash
python src/app.py
```

В приложении доступны три вкладки; кнопка **Retrain** на вкладке Analytics
делает шаги 1–2 в фоновом потоке и обновляет интерфейс по завершении.

### 4. (Опционально) MLflow UI

```bash
mlflow ui --backend-store-uri file:///%CD%/mlruns   # Windows
mlflow ui --backend-store-uri file://$PWD/mlruns    # macOS/Linux
```

## Как работает препроцессинг

Ключевая особенность датасета — у части объявлений критичные поля (RAM, ROM,
GPU, процессор) пустые: продавцы их пишут только в свободном тексте `subject`.
`FeatureExtractor` ([src/preprocessing.py](src/preprocessing.py)) восстанавливает
эти поля регулярками: понимает форматы `"16/512"`, `"16ГБ 1000ГБ"`,
`"RTX 3060 Ti Max-Q"`, MacBook'и, Apple-style Mn Pro/Max/Ultra и т.д. Дальше
идут sklearn-овские `OrdinalEncoder` + `SimpleImputer`, всё внутри одного
`Pipeline`.

Целевая переменная обучается на `log1p(price)` (тяжёлый правый хвост), а
функция потерь XGBoost — `reg:absoluteerror`, что даёт устойчивость к выбросам.

## Структура CSV

Парсер сохраняет 19 признаков: `ad_id, subject, price_byn, company_ad,
list_time, condition, brand, processor, rom_volume, rom_type, diagonal, os,
videocard, videocard_brand, region, gaming_laptop, matrix_type,
display_resolution, ram_volume, ram_type, battery_life`.

Имя файла кодирует время сбора (`YYYYMMDD_HHMM`) — оно же используется при
обучении как «время доступа» для расчёта давности объявления.

## ToDo

- [ ] Сбор большего числа признаков, векторизация subject, парсинг и extractoin значений из описания.
- [ ] Перенос инференса в REST-сервис (FastAPI).
- [ ] Дрейф признаков между снапшотами (Evidently).
- [ ] Регистрация модели в MLflow Model Registry со stages.

## Важно

- API Kufar геоблокирован вне РБ/РФ — для парсинга может потребоваться VPN.