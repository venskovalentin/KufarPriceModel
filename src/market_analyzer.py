import glob
import os
import pandas as pd
import predictor

class MarketAnalyzer:
    def __init__(self, data_path=r'..\data\raw'):

        file_type = '/*csv'
        files = glob.glob(data_path + file_type)
        latest_file = max(files, key=os.path.getctime)

    def get_price_distribution(self, filters=None) -> pd.Series:
        # цены с учётом фильтров (бренд, процессор и т.д.)
        pass

    def get_market_stats(self, filters=None) -> dict:
        # {"median": ..., "mean": ..., "count": ..., "min": ..., "max": ...}
        pass

    def find_underpriced(self, predictor: Predictor, threshold=0.85) -> pd.DataFrame:
        # объявления где цена < threshold * predicted_price
        pass