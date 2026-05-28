"""Market-level analytics built on top of the latest CSV snapshot.

Provides the data layer for the "Market Analysis" tab of the GUI: filtering,
basic stats and an underpriced-listings detector that compares actual prices
against the model's prediction.
"""

import glob
import os

import pandas as pd

from predictor import Predictor


class MarketAnalyzer:
    """Load the most recent CSV from ``data_path`` and expose query helpers."""

    def __init__(self, data_path=r'..\data\raw'):
        files = glob.glob(os.path.join(data_path, '*.csv'))
        if not files:
            raise FileNotFoundError(f"No CSV files found in {data_path}")

        latest_file = max(files, key=os.path.getctime)
        self.df = pd.read_csv(latest_file)

    def _apply_filters(self, df: pd.DataFrame, filters: dict) -> pd.DataFrame:
        """Equality filter on the given columns. Empty values are ignored."""
        if not filters:
            return df
        filtered_df = df.copy()
        for key, value in filters.items():
            if key in filtered_df.columns and value:
                filtered_df = filtered_df[filtered_df[key] == value]
        return filtered_df

    def get_price_distribution(self, filters=None) -> pd.Series:
        filtered_df = self._apply_filters(self.df, filters)
        return filtered_df['price_byn'].dropna()

    def get_market_stats(self, filters=None) -> dict:
        dist = self.get_price_distribution(filters)
        if dist.empty:
            return {"median": 0, "mean": 0, "count": 0, "min": 0, "max": 0}

        return {
            "median": round(dist.median(), 2),
            "mean":   round(dist.mean(), 2),
            "count":  len(dist),
            "min":    round(dist.min(), 2),
            "max":    round(dist.max(), 2),
        }

    def find_underpriced(self, predictor: Predictor, threshold=0.85) -> pd.DataFrame:
        """Return listings whose price is below ``threshold * predicted_price``.

        Sorted from the deepest discount upward.
        """
        eval_df = self.df.dropna(subset=['price_byn']).copy()
        eval_df = eval_df[eval_df['price_byn'] > 30]
        if eval_df.empty:
            return pd.DataFrame()

        eval_df['predicted_price'] = predictor.predict(eval_df.drop(columns=['price_byn']))

        underpriced_mask = eval_df['price_byn'] < (eval_df['predicted_price'] * threshold)
        underpriced = eval_df[underpriced_mask].copy()

        underpriced['discount_ratio'] = underpriced['price_byn'] / underpriced['predicted_price']
        underpriced['profit_byn']     = underpriced['predicted_price'] - underpriced['price_byn']

        return underpriced.sort_values('discount_ratio', ascending=True)
