import requests
import pandas as pd
import csv
import time

category_code = 1604 # code of laptops

URL = f"https://api.kufar.by/search-api/v2/search/rendered-paginated?cat={category_code}0&lang=ru&size=20&cur=BYR"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json"
}

def get_param(data: dict, param_name: str):
    ad_parameters = data["ad_parameters"]
    for dict_ in ad_parameters:
        if dict_.get('p', None) == param_name:
            return dict_.get("vl", None)
 # Return the value associated with parameter


def next_page_code(data):
    for dict_ in data["pagination"]["pages"]:
        if dict_['label'] == 'next':
            return dict_['token']
    return None
 # Find token of next page

def parse_ad(ad):
    return {
        'ad_id': ad["ad_id"],
        'subject': ad["subject"],
        'price_byn': float(ad["price_byn"]) / 100,
        'company_ad': ad["company_ad"],
        'list_time': ad["list_time"],
        'condition': get_param(ad, "condition"),
        'brand': get_param(ad, "computers_laptop_brand"),
        'processor': get_param(ad, "computers_laptop_processor"),
        'rom_volume': get_param(ad, "computers_laptop_hdd_volume"),
        'rom_type': get_param(ad, "computers_laptop_hdd_type"),
        'diagonal': get_param(ad, "computers_laptop_diagonal"),
        'os': get_param(ad, "computers_laptop_os"),
        'videocard': get_param(ad, "computers_laptop_videocard"),
        'videocard_brand': get_param(ad, "computers_laptop_videocard_brand"),
        'region': get_param(ad, "region"),
        'gaming_laptop': get_param(ad, "computer_equipment_laptops_gaming"),
        'matrix_type': get_param(ad, "computers_display_matrix_type"),
        'display_resolution': get_param(ad, "computers_laptop_resolution"),
        'ram_volume': get_param(ad, "computer_equipment_laptops_ram"),
        'ram_type': get_param(ad, "computers_ram_type"),
        'battery_life': get_param(ad, "computers_laptop_battery_life"),
    }

    # Create dictionary with params

def fetch_with_retry(url, headers, max_retries=3, base_delay=0.5):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"Попытка {attempt + 1} упала: {e}")
            time.sleep(base_delay * (attempt + 1))

    return None


def parse_all_ads(URL, HEADERS):
    all_ads = []
    counter = 0

    url = URL
    data = fetch_with_retry(url, HEADERS)

    if not data:
        print("Не удалось получить первую страницу")
        return pd.DataFrame()

    while True:
        print(f"processing page {counter}")

        all_ads.extend([parse_ad(ad) for ad in data['ads']])

        next_page = next_page_code(data)
        if not next_page:
            break

        url = f"{URL}&cursor={next_page}"
        data = fetch_with_retry(url, HEADERS)

        if not data:
            print("Ошибка при получении страницы, пропускаем...")
            break

        counter += 1
        time.sleep(0.5)

    return pd.DataFrame(all_ads)

df = parse_all_ads(URL, HEADERS)

from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M")
df.to_csv(f'../data/raw/{timestamp}_price_data.csv', index=False)