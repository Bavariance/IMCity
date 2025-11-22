# src/scrape_live_data.py

import os
import time
import json
from pathlib import Path

import pandas as pd
import requests
import pytz
from google.cloud import storage  # pip install google-cloud-storage

# ----------------- Local paths -----------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Canonical 15‑minute time-series lives in data/
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CANONICAL_CSV = DATA_DIR / "eisbach_15min.csv"

# Live artifacts (small, for convenience) live in live_data/
LIVE_DIR = PROJECT_ROOT / "live_data"
LIVE_DIR.mkdir(parents=True, exist_ok=True)

LATEST_JSON = LIVE_DIR / "latest_snapshot.json"

# ----------------- GCS config ------------------

GCS_BUCKET = os.environ.get("GCS_BUCKET")              # e.g. "imcity_live_data"
GCS_PREFIX = os.environ.get("GCS_PREFIX", "live_data")  # folder inside bucket

_storage_client: storage.Client | None = None


def get_storage_client() -> storage.Client:
    """Get a cached Storage client using ADC / VM service account."""
    global _storage_client
    if _storage_client is None:
        _storage_client = storage.Client()   # uses VM service account / ADC
    return _storage_client


def upload_to_gcs(local_path: Path, object_name: str) -> None:
    """Uploads local_path to gs://GCS_BUCKET/object_name (no-op if bucket unset)."""
    if not GCS_BUCKET:
        return

    client = get_storage_client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(object_name)
    blob.upload_from_filename(str(local_path))
    print(f"[GCS] Uploaded {local_path} -> gs://{GCS_BUCKET}/{object_name}")

# ----------------- Scraping logic --------------

HND_BASE = (
    "https://www.hnd.bayern.de/pegel/isar/"
    "muenchen-himmelreichbruecke-16515005/tabelle"
)


def _fetch_table(methode: str) -> pd.DataFrame:
    """
    Fetch the HND HTML table for 'wasserstand' or 'abfluss' and return as DataFrame.

    The table has local timestamps in 'DD.MM.YYYY HH:MM' and values with comma decimal. 
    """
    params: dict[str, str] = {"methode": methode}
    if methode == "abfluss":
        # 15‑minute discharge values
        params["setdiskr"] = "15"

    resp = requests.get(HND_BASE, params=params, timeout=10)
    resp.raise_for_status()

    # First table is the time series; decimal="," parses German comma floats
    tables = pd.read_html(resp.text, decimal=",")
    if not tables:
        raise RuntimeError(f"No tables found for methode={methode}")
    df = tables[0]

    # Expect first column = timestamp, second column = value
    df = df.iloc[:, :2].copy()
    df.columns = ["timestamp_local_raw", "value"]
    return df


def fetch_eisbach_point() -> dict:
    """
    Scrape latest Wasserstand (cm) and Abfluss (m³/s) from HND and return a canonical dict.

    Returns keys:
        - timestamp_local: 'YYYY-MM-DD HH:MM' (Europe/Berlin)
        - timestamp_utc:   'YYYY-MM-DDTHH:MM:SSZ'
        - waterlevel_cm:   float
        - discharge_m3s:   float
        - status_level:    string (e.g. 'Rohdaten')
        - status_flow:     string (e.g. 'Rohdaten')
    """
    level_df = _fetch_table("wasserstand")
    flow_df = _fetch_table("abfluss")

    # HND tables are ordered newest‑first; take the first row
    level_row = level_df.iloc[0]
    flow_row = flow_df.iloc[0]

    # Parse local time (CET/CEST) and convert to UTC ISO8601
    tz_local = pytz.timezone("Europe/Berlin")
    ts_local_parsed = pd.to_datetime(level_row["timestamp_local_raw"], dayfirst=True)
    ts_local = tz_local.localize(ts_local_parsed)
    ts_utc = ts_local.astimezone(pytz.UTC)

    timestamp_local = ts_local.strftime("%Y-%m-%d %H:%M")
    timestamp_utc = ts_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Near-real-time data from HND/GKD is typically 'Rohdaten'
    status_level = "Rohdaten"
    status_flow = "Rohdaten"

    return {
        "timestamp_local": timestamp_local,
        "timestamp_utc": timestamp_utc,
        "waterlevel_cm": float(level_row["value"]),
        "discharge_m3s": float(flow_row["value"]),
        "status_level": status_level,
        "status_flow": status_flow,
    }


def append_canonical_row(path: Path, row: dict) -> None:
    """Append one canonical row to data/eisbach_15min.csv."""
    df_row = pd.DataFrame(
        [
            {
                "timestamp_local": row["timestamp_local"],
                "timestamp_utc": row["timestamp_utc"],
                "waterlevel_cm": row["waterlevel_cm"],
                "discharge_m3s": row["discharge_m3s"],
                "status_level": row["status_level"],
                "status_flow": row["status_flow"],
            }
        ]
    )
    if path.exists():
        df_row.to_csv(path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(path, index=False)


def write_latest_snapshot(row: dict) -> None:
    """Write the latest observation as a small JSON file in live_data/."""
    tmp = LATEST_JSON.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(row, f)
    os.replace(tmp, LATEST_JSON)  # atomic on POSIX


def run_once() -> None:
    """
    Single scrape‑and‑upload pass.

    - Scrape latest values from HND.
    - Append to canonical data/eisbach_15min.csv.
    - Write JSON snapshot.
    - Upload canonical CSV + JSON to GCS.
    """
    row = fetch_eisbach_point()

    # Append to canonical 15‑minute series
    append_canonical_row(CANONICAL_CSV, row)

    # Update snapshot
    write_latest_snapshot(row)

    # Upload canonical CSV + latest snapshot to GCS
    csv_object = f"{GCS_PREFIX}/eisbach_15min.csv"
    json_object = f"{GCS_PREFIX}/latest_snapshot.json"

    upload_to_gcs(CANONICAL_CSV, csv_object)
    upload_to_gcs(LATEST_JSON, json_object)


def main_loop() -> None:
    """Long‑lived loop: scrape + upload every 15 minutes."""
    while True:
        try:
            run_once()
        except Exception as e:
            print(f"[SCRAPER] Error in run_once: {e}")
        time.sleep(15 * 60)


if __name__ == "__main__":
    main_loop()
