# Kufar marketplace scraper.
#
# The API uses cursor-based pagination. We probe a few page sizes at startup
# because the API silently truncates large pages, so the first size that comes
# back full enough is the one we stick with.

import os
import time
from datetime import datetime

import pandas as pd
import requests


CATEGORY_CODE = 1604               # Kufar category id — "laptops".
DEFAULT_SIZES = (200, 100, 50, 20)  # Page sizes to try, largest first.

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
}


def _build_url(size: int, cursor: str | None = None) -> str:
    url = (
        f"https://api.kufar.by/search-api/v2/search/rendered-paginated"
        f"?cat={CATEGORY_CODE}0&lang=ru&size={size}&cur=BYR"
    )
    if cursor:
        url += f"&cursor={cursor}"
    return url


def get_param(ad: dict, param_name: str):
    for d in ad.get("ad_parameters", []):
        if d.get("p") == param_name:
            return d.get("vl")
    return None


def next_page_token(data: dict) -> str | None:
    for d in data.get("pagination", {}).get("pages", []):
        if d.get("label") == "next":
            return d.get("token")
    return None


def parse_ad(ad: dict) -> dict:
    return {
        "ad_id": ad["ad_id"],
        "subject": ad["subject"],
        "price_byn": float(ad["price_byn"]) / 100,
        "company_ad": ad["company_ad"],
        "list_time": ad["list_time"],
        "condition": get_param(ad, "condition"),
        "brand": get_param(ad, "computers_laptop_brand"),
        "processor": get_param(ad, "computers_laptop_processor"),
        "rom_volume": get_param(ad, "computers_laptop_hdd_volume"),
        "rom_type": get_param(ad, "computers_laptop_hdd_type"),
        "diagonal": get_param(ad, "computers_laptop_diagonal"),
        "os": get_param(ad, "computers_laptop_os"),
        "videocard": get_param(ad, "computers_laptop_videocard"),
        "videocard_brand": get_param(ad, "computers_laptop_videocard_brand"),
        "region": get_param(ad, "region"),
        "gaming_laptop": get_param(ad, "computer_equipment_laptops_gaming"),
        "matrix_type": get_param(ad, "computers_display_matrix_type"),
        "display_resolution": get_param(ad, "computers_laptop_resolution"),
        "ram_volume": get_param(ad, "computer_equipment_laptops_ram"),
        "ram_type": get_param(ad, "computers_ram_type"),
        "battery_life": get_param(ad, "computers_laptop_battery_life"),
    }


def _fetch(session: requests.Session, url: str, max_retries: int = 3, base_delay: float = 0.5):
    """GET ``url`` with linear backoff. Returns parsed JSON or ``None``."""
    for attempt in range(max_retries):
        try:
            r = session.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            time.sleep(base_delay * (attempt + 1))
    return None


def _pick_page_size(session: requests.Session) -> int:
    """Probe candidate page sizes and pick the first one the API honors."""
    for size in DEFAULT_SIZES:
        data = _fetch(session, _build_url(size))
        if data and data.get("ads"):
            returned = len(data["ads"])
            if returned >= min(size, 20):
                return size
    raise RuntimeError("Kufar API returned no data for any page size")


def parse_all_ads() -> pd.DataFrame:
    """Iterate through all paginated results and return them as a DataFrame."""
    with requests.Session() as session:
        size = _pick_page_size(session)
        print(f"Using page size = {size}")

        all_ads: list[dict] = []
        cursor = None
        counter = 0

        while True:
            print(f"processing page {counter}")
            data = _fetch(session, _build_url(size, cursor))
            if not data:
                print("Failed to fetch page, aborting pagination")
                break

            ads = data.get("ads", [])
            all_ads.extend(parse_ad(a) for a in ads)

            cursor = next_page_token(data)
            if not cursor:
                break
            counter += 1

    return pd.DataFrame(all_ads)


def fetch_and_save(folder_path: str) -> str:
    """Run a full scrape and persist the result to ``folder_path``.

    The file name encodes the scrape time (``YYYYMMDD_HHMM_price_data.csv``) —
    :func:`retrainer.retrainer` parses this back as the listing ``access_time``.
    """
    df = parse_all_ads()
    if df.empty:
        raise RuntimeError("Parser returned 0 ads — CSV not saved")
    os.makedirs(folder_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = os.path.join(folder_path, f"{timestamp}_price_data.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} ads to {out_path}")
    return out_path


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    fetch_and_save(os.path.join(here, "..", "data", "raw"))
