import pandas as pd
import numpy as np
import csv, os, re, json, glob
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer


def extract_gpu(subject: str):
    """Возвращает 0 — встроенная, 1 — дискретная"""
    if pd.isna(subject) or not isinstance(subject, str):
        return float('nan')
    s = subject

    # Встроенная
    if re.search(r'mac\s*book', s, re.IGNORECASE):
        return 0
    if re.search(r'Intel\s+Iris\s+Xe', s, re.IGNORECASE):
        return 0
    if re.search(r'Intel\s+UHD\s+Graphics', s, re.IGNORECASE):
        return 0

    # Дискретная
    if re.search(r'(?:NVIDIA|GeForce|RTX|GTX|MX|Quadro|Arc|Radeon)', s, re.IGNORECASE):
        return 1

    return float('nan')


def extract_gpu_model(subject: str):
    if pd.isna(subject) or not isinstance(subject, str):
        return float('nan')
    s = subject
    if pd.isna(subject):
        return float('nan')

    def find_mem(text: str):
        """
        Берёт VRAM сразу после модели.
        Если следом идёт ЕЩЁ одна память (16ГБ 1000ГБ) — это RAM+SSD, пропускаем.
        Если число > 64 — это точно не VRAM.
        """
        m = re.search(r'(\d+)\s*(?:ГБ|GB)', text, re.IGNORECASE)
        if not m:
            return float('nan')
        if int(m.group(1)) > 64:
            return float('nan')
        tail = text[m.end():m.end() + 25]
        if re.search(r'\d+\s*(?:ГБ|GB|ТБ|TB)', tail, re.IGNORECASE):
            return float('nan')
        return m.group(0)

    def fmt(base: str, mem: str):
        return f"{base} {mem}" if mem else base

    SEP = r'[\s\-]*'  # separator: space, - or nothing

    if re.search(r'mac\s*book', s, re.IGNORECASE):
        return 0
    # Intel
    m = re.search(r'Intel\s+Iris\s+Xe\s+Graphics\s+(G\d+)', s, re.IGNORECASE)
    if m:
        return f"Intel Iris XE Graphics {m.group(1).upper()}"
    if re.search(r'Intel\s+Iris\s+Xe', s, re.IGNORECASE):
        return 'Intel Iris XE Graphics'
    if re.search(r'Intel\s+UHD\s+Graphics', s, re.IGNORECASE):
        return 'Intel UHD Graphics'
    m = re.search(r'(?:Intel\s+)?Arc\s+(A\d{3}M?)', s, re.IGNORECASE)
    if m:
        return fmt(f"Intel Arc {m.group(1).upper()}", find_mem(s[m.end():]))

    # NVIDIA Quadro
    m = re.search(rf'(?:NVIDIA\s+)?Quadro{SEP}RTX{SEP}(\d{{3,5}})\b', s, re.IGNORECASE)
    if m:
        return fmt(f"NVIDIA Quadro RTX {m.group(1)}", find_mem(s[m.end():]))

    m = re.search(rf'(?:NVIDIA\s+)?Quadro{SEP}T{SEP}(\d{{3,4}})\s*(Max-Q)?', s, re.IGNORECASE)
    if m:
        maxq = " Max-Q" if m.group(2) else ""
        return fmt(f"NVIDIA Quadro T{m.group(1)}{maxq}", find_mem(s[m.end():]))

    m = re.search(rf'(?:NVIDIA\s+)?Quadro{SEP}P{SEP}(\d{{3,4}})\b', s, re.IGNORECASE)
    if m:
        return fmt(f"NVIDIA Quadro P{m.group(1)}", find_mem(s[m.end():]))

    m = re.search(rf'(?:NVIDIA\s+)?Quadro{SEP}M{SEP}(\d{{3,4}})\b', s, re.IGNORECASE)
    if m:
        return fmt(f"NVIDIA Quadro M{m.group(1)}", find_mem(s[m.end():]))

    # NVIDIA GeForce RTX / GTX / MX
    m = re.search(
        rf'(?:NVIDIA\s+)?(?:GeForce\s+)?RTX{SEP}(\d{{4}}){SEP}(Ti\b)?\s*(Max-Q)?',
        s, re.IGNORECASE
    )
    if m:
        ti = " Ti" if m.group(2) else ""
        maxq = " Max-Q" if m.group(3) else ""
        return fmt(f"NVIDIA GeForce RTX {m.group(1)}{ti}{maxq}", find_mem(s[m.end():]))

    m = re.search(
        rf'(?:NVIDIA\s+)?(?:GeForce\s+)?GTX{SEP}(\d{{4}}){SEP}(Ti\b)?\s*(Max-Q)?',
        s, re.IGNORECASE
    )
    if m:
        ti = " Ti" if m.group(2) else ""
        maxq = " Max-Q" if m.group(3) else ""
        return fmt(f"NVIDIA GeForce GTX {m.group(1)}{ti}{maxq}", find_mem(s[m.end():]))

    m = re.search(rf'(?:NVIDIA\s+)?(?:GeForce\s+)?MX{SEP}(\d{{3,4}})', s, re.IGNORECASE)
    if m:
        return fmt(f"NVIDIA GeForce MX{m.group(1)}", find_mem(s[m.end():]))

    # AMD Radeon
    m = re.search(r'(?:AMD\s+)?Radeon\s+RX\s+Vega\s+M\s+GL', s, re.IGNORECASE)
    if m:
        return fmt("AMD Radeon RX Vega M GL", find_mem(s[m.end():]))

    m = re.search(rf'(?:AMD\s+)?Radeon\s+Pro{SEP}(\d{{3,4}}[A-Z]*)', s, re.IGNORECASE)
    if m:
        return fmt(f"AMD Radeon Pro {m.group(1)}", find_mem(s[m.end():]))

    m = re.search(rf'(?:AMD\s+)?(?:Radeon\s+)?RX{SEP}(\d{{3,4}}M?)', s, re.IGNORECASE)
    if m:
        return fmt(f"AMD Radeon RX {m.group(1).upper()}", find_mem(s[m.end():]))

    m = re.search(rf'(?:AMD\s+)?Radeon{SEP}(\d{{3,4}}[A-Z]*)', s, re.IGNORECASE)
    if m:
        return fmt(f"AMD Radeon {m.group(1).upper()}", find_mem(s[m.end():]))
    return float('nan')

def extract_ram(subject):
    """Извлекает объём RAM в формате 'X ГБ'"""
    if pd.isna(subject) or not isinstance(subject, str):
        return float('nan')
    s = subject

    # 1. RAM
    m = re.search(r'(\d+)\s*(?:ГБ|GB)\s*(?:RAM|ОЗУ)', s, re.IGNORECASE)
    if m:
        return f"{m.group(1)} ГБ"
    m = re.search(r'(?:RAM|ОЗУ)[\s:\-]*(\d+)\s*(?:ГБ|GB)?', s, re.IGNORECASE)
    if m:
        return f"{m.group(1)} ГБ"

    # 2. Apple-style "16/256" — первое число RAM, второе ROM
    m = re.search(r'\b(\d{1,3})\s*/\s*(\d{2,4})\b', s)
    if m:
        ram, rom = int(m.group(1)), int(m.group(2))
        if ram in {2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128} and rom >= 64:
            return f"{ram} ГБ"

    # 3. "16GB/512GB" — первое RAM
    m = re.search(r'(\d+)\s*(?:ГБ|GB)\s*/\s*\d+\s*(?:ГБ|GB|ТБ|TB)', s, re.IGNORECASE)
    if m and int(m.group(1)) <= 128:
        return f"{m.group(1)} ГБ"

    # 4. "16ГБ 1000ГБ" — два числа подряд, первое RAM
    m = re.search(r'(\d+)\s*(?:ГБ|GB)\s+(\d+)\s*(?:ГБ|GB|ТБ|TB)', s, re.IGNORECASE)
    if m and int(m.group(1)) <= 128:
        return f"{m.group(1)} ГБ"

    return float('nan')


def extract_rom(subject):
    """Извлекает объём ROM в формате 'X ГБ'"""
    if pd.isna(subject) or not isinstance(subject, str):
        return float('nan')
    s = subject

    # 1. ТБ → переводим в ГБ
    m = re.search(r'(\d+)\s*(?:ТБ|TB)', s, re.IGNORECASE)
    if m:
        return f"{int(m.group(1)) * 1000} ГБ"

    # 2. SSD/HDD маркер
    m = re.search(r'(?:SSD|HDD)[\s\-]*(\d+)\s*(?:ГБ|GB)', s, re.IGNORECASE)
    if m:
        return f"{m.group(1)} ГБ"
    m = re.search(r'(\d+)\s*(?:ГБ|GB)\s*(?:SSD|HDD)', s, re.IGNORECASE)
    if m:
        return f"{m.group(1)} ГБ"

    # 3. Apple-style "16/256" — второе число ROM
    m = re.search(r'\b(\d{1,3})\s*/\s*(\d{2,4})\b', s)
    if m:
        ram, rom = int(m.group(1)), int(m.group(2))
        if ram in {2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128} and rom >= 64:
            return f"{rom} ГБ"

    # 4. "16GB/512GB" — второе число ROM
    m = re.search(r'\d+\s*(?:ГБ|GB)\s*/\s*(\d+)\s*(?:ГБ|GB)', s, re.IGNORECASE)
    if m and int(m.group(1)) >= 64:
        return f"{m.group(1)} ГБ"

    # 5. "16ГБ 1000ГБ" — два числа подряд, второе ROM
    m = re.search(r'(\d+)\s*(?:ГБ|GB)\s+(\d+)\s*(?:ГБ|GB)', s, re.IGNORECASE)
    if m and int(m.group(1)) <= 128 and int(m.group(2)) >= 64:
        return f"{m.group(2)} ГБ"

    return float('nan')


def extract_diagonal(subject):
    """Извлекает диагональ в дюймах (float)"""
    if pd.isna(subject) or not isinstance(subject, str):
        return float('nan')
    s = subject

    # 1. С маркером дюймов: 15.6", 17,3'', 13.3 дюйм
    m = re.search(r'(\d{2}[,.]?\d?)\s*(?:[\'"]{1,2}|дюйм|inch)', s, re.IGNORECASE)
    if m:
        val = float(m.group(1).replace(',', '.'))
        if 10 <= val <= 18:
            return val

    # 2. Десятичное число типа 15.6, 17.3, 13.3 — редко false positive
    m = re.search(r'\b(1[0-7][,.]\d)\b', s)
    if m:
        val = float(m.group(1).replace(',', '.'))
        if 10 <= val <= 18:
            return val

    # 3. После MacBook/Pro/Air — целое число 13-17
    m = re.search(r'(?:MacBook|Air|Pro)\s+(\d{2})\b', s, re.IGNORECASE)
    if m:
        val = float(m.group(1))
        if 10 <= val <= 18:
            return val

    return float('nan')


def extract_processor(subject):
    """Извлекает категорию процессора"""
    if pd.isna(subject) or not isinstance(subject, str):
        return float('nan')
    s = subject

    # Apple M1/M2/M3/M4/M5 (Pro/Max/Ultra)
    m = re.search(r'\bM([1-5])\s*(Pro|Max|Ultra)?\b', s)
    # Только если контекст Apple/MacBook
    if m and re.search(r'mac\s*book|apple', s, re.IGNORECASE):
        suffix = f" {m.group(2)}" if m.group(2) else ""
        return f"Apple M{m.group(1)}{suffix}"

    # Intel Core iN
    m = re.search(r'(?:Core\s+)?i([3579])[\s\-]?\d{3,5}[A-Z]{0,2}', s, re.IGNORECASE)
    if m:
        return f"Intel Core i{m.group(1)}"
    m = re.search(r'Core\s+i([3579])', s, re.IGNORECASE)
    if m:
        return f"Intel Core i{m.group(1)}"

    # AMD Ryzen — "Ryzen 5", "R5-7520U"
    m = re.search(r'Ryzen\s+([3579])', s, re.IGNORECASE)
    if m:
        return f"AMD Ryzen {m.group(1)}"
    m = re.search(r'\bR([3579])\s*[\-–]\s*\d{3,5}[A-Z]{0,2}', s, re.IGNORECASE)
    if m:
        return f"AMD Ryzen {m.group(1)}"

    # Intel Pentium / Celeron
    if re.search(r'Pentium', s, re.IGNORECASE):
        return 'Intel Pentium'
    if re.search(r'Celeron', s, re.IGNORECASE):
        return 'Intel Celeron'

    return float('nan')


def extract_brand(subject):
    if pd.isna(subject) or not isinstance(subject, str):
        return float('nan')
    s = subject

    brand_patterns = [
        (r'\bMacBook|Apple\b', 'Apple'),
        (r'\bASUS|Aorus\b', 'Asus'),
        (r'\bLenovo|ThinkPad|IdeaPad|Legion\b', 'Lenovo'),
        (r'\bHP|HewlettPackard|Pavilion|EliteBook|ProBook|Omen|Victus\b', 'HP'),
        (r'\bDell|Inspiron|Latitude|XPS|Alienware|Vostro\b', 'Dell'),
        (r'\bAcer|Aspire|Predator|Nitro|Swift|TravelMate\b', 'Acer'),
        (r'\bMSI\b', 'MSI'),
        (r'\bHuawei|MateBook\b', 'Huawei'),
        (r'\bHonor\s*MagicBook|Honor\b', 'Honor'),
        (r'\bSamsung|Galaxy\s*Book\b', 'Samsung'),
        (r'\bXiaomi|RedmiBook|Redmi\b', 'Xiaomi'),
        (r'\bGigabyte\b', 'Gigabyte'),
        (r'\bRazer\b', 'Razer'),
        (r'\bMicrosoft|Surface\b', 'Microsoft'),
        (r'\bLG\s*Gram|LG\b', 'LG'),
        (r'\bToshiba|Dynabook\b', 'Toshiba'),
        (r'\bSony|VAIO\b', 'Sony'),
        (r'\bChuwi\b', 'Chuwi'),
        (r'\bThunderobot\b', 'Thunderobot'),
        (r'\bMechRevo\b', 'MechRevo'),
        (r'\bMaibenben\b', 'Maibenben'),
        (r'\bDigma\b', 'Digma'),
        (r'\bIRBIS\b', 'IRBIS'),
        (r'\bHaier\b', 'Haier'),
    ]

    for pattern, brand in brand_patterns:
        if re.search(pattern, s, re.IGNORECASE):
            return brand

    return float('nan')

def parse_number(val):
    if pd.isna(val):
        return np.nan
    match = re.search(r'\d+\.?\d*', str(val))
    return float(match.group()) if match else np.nan

def extract_first_number(series: pd.Series, transform=None) -> pd.Series:
    result = series.apply(parse_number)
    return result


class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, access_time):
        self.access_time = access_time

        self.unknown_features = [
            'brand', 'videocard_brand', 'ram_type', 'matrix_type', 'rom_type',
            'processor', 'os'
        ]

        self.mode_features = [
            'display_resolution', 'rom_type'
        ]

        self.num_knn_features = [
            'ram_volume', 'rom_volume', 'diagonal',
            'battery_life'
        ]

        self.resolution_map = {
            '1280 х 800': 'HD',
            '1366 х 768': 'HD',
            '1024 х 600': 'HD',
            '1920 х 1080': 'FHD',
            '1920 х 945': 'FHD',
            '1792 х 768': 'FHD',
            '1920 х 1200': 'FHD',
            '1600 х 900': 'HD+',
            '2560 х 1440': '2K',
            '2560 х 1600': '2K',
            '2304 х 1440': '2K',
            '2880 х 1620': '2K',
            '2880 х 1800': '3K',
            '3200 х 1800': '3K',
            '3840 х 2160': '4K',
            '1440 х 900': 'другое'
        }

        self.battery_map = {
            '1-2 часа': 1.5,
            '2-4 часа': 3,
            '4-6 часов': 5,
            '6-10 часов': 8,
            '10 и более часов': 12,
            '1 час и меньше': 0.5
        }

        self.condition_map = {
            "Б/у": 0,
            "Новое": 1
        }

    def fit(self, X, y=None):
        return self

    def _fill_from_subject(self, X):
        X['videocard'] = X['subject'].apply(extract_gpu)
        X['videocard_brand'] = X['videocard_brand'].fillna(X['subject'].apply(extract_gpu_model))
        X['ram_volume'] = X['ram_volume'].fillna(X['subject'].apply(extract_ram))
        X['rom_volume'] = X['rom_volume'].fillna(X['subject'].apply(extract_rom))
        X['diagonal'] = X['diagonal'].fillna(X['subject'].apply(extract_diagonal))
        X['processor'] = X['processor'].fillna(X['subject'].apply(extract_processor))
        X['brand'] = X['brand'].fillna(X['subject'].apply(extract_brand))
        X["rom_volume"] = extract_first_number(X["rom_volume"])
        X["ram_volume"] = extract_first_number(X["ram_volume"])
        X["diagonal"] = extract_first_number(X["diagonal"])
        return X

    def _fix_videocard(self, X):
        videocard_existance_mask = (
                X['videocard_brand'].notna() &
                ~X['videocard_brand'].isin(['0', 0])
        )

        X.loc[videocard_existance_mask & X['videocard'].isna(), 'videocard'] = 1

        return X

    def _map_categoricals(self, X):
        X['battery_life'] = X['battery_life'].map(self.battery_map)
        X['condition'] = X['condition'].map(self.condition_map)
        X['display_resolution'] = X['display_resolution'].map(self.resolution_map)

        return X

    def _fill_apple(self, X):
        apple_mask = (
                X['brand'].eq('Apple') |
                X['os'].eq('Mac OS')
        )

        X.loc[apple_mask, 'videocard'] = 0
        X.loc[apple_mask, 'videocard_brand'] = 0
        X.loc[apple_mask, 'rom_type'] = 'SSD'
        X.loc[apple_mask, 'matrix_type'] = 'IPS'

        return X

    def _change_time(self, X):
        X["list_time"] = pd.to_datetime(X["list_time"])
        X["timedelta_minutes"] = (self.access_time - X["list_time"]).dt.total_seconds() / 60
        return X

    def _drop_columns(self, X):
        return X.drop(['list_time', 'subject', 'ad_id', 'gaming_laptop'], axis=1)

    def extract(self, X, y=None):
        X = X.copy()

        X = self._fill_from_subject(X)
        X = self._fix_videocard(X)
        X = self._map_categoricals(X)
        X = self._fill_apple(X)
        X = self._change_time(X)
        X = self._drop_columns(X)

        X['company_ad'] = X['company_ad'].astype(int)
        X = X.fillna(np.nan)

        return X

def preprocess(X, access_time, y=None):
    extractor = FeatureExtractor(access_time)
    res = extractor.extract(X)
    return res
