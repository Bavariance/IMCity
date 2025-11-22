# src/scrape_live_data.py
import os
import time
import json
import datetime as dt
from pathlib import Path

import pandas as pd
import requests
from google.cloud import storage  # pip install google-cloud-storage

# ----------------- Local paths -----------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "live_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

EISBACH_CSV = DATA_DIR / "eisbach_live.csv"
LATEST_JSON = DATA_DIR / "latest_snapshot.json"

# ----------------- GCS config ------------------

GCS_BUCKET = os.environ.get("GCS_BUCKET")        # e.g. "imcity-live-data"
GCS_PREFIX = os.environ.get("GCS_PREFIX", "live_data")  # folder inside bucket

# Create client lazily (to avoid issues when running without credentials)
_storage_client: storage.Client | None = None


def get_storage_client() -> storage.Client:
    global _storage_client
    if _storage_client is None:
        _storage_client = storage.Client()   # uses GOOGLE_APPLICATION_CREDENTIALS
    return _storage_client


def upload_to_gcs(local_path: Path, object_name: str) -> None:
    """
    Uploads local_path to gs://GCS_BUCKET/object_name.
    No‑op if GCS_BUCKET is not set (useful for local testing).
    """
    if not GCS_BUCKET:
        # Skip upload if bucket not configured
        return

    client = get_storage_client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(object_name)
    # if_generation_match=0 -> only create if object does not exist; drop to allow overwrite
    blob.upload_from_filename(str(local_path))
    print(f"[GCS] Uploaded {local_path} -> gs://{GCS_BUCKET}/{object_name}")


# ----------------- Scraping logic --------------

def fetch_eisbach_point() -> dict:
    """
    Replace this stub with real HTTP calls to HND Bayern / timeanddate, etc.
    Keep total external requests << 1 rps to satisfy IMC rules.
    """
    now = dt.datetime.utcnow()
    # TODO: parse real API responses here
    return {
        "timestamp": now.isoformat(),
        "flow_m3s": 24.3,
        "level_cm": 141.0,
    }


def append_csv(path: Path, row: dict):
    df_row = pd.DataFrame([row])
    if path.exists():
        df_row.to_csv(path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(path, index=False)


def write_latest_snapshot(row: dict):
    tmp = LATEST_JSON.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(row, f)
    os.replace(tmp, LATEST_JSON)  # atomic on POSIX


def run_once():
    """
    Single scrape‑and‑upload pass.
    Call this from cron/systemd, or from main_loop().
    """
    row = fetch_eisbach_point()
    append_csv(EISBACH_CSV, row)
    write_latest_snapshot(row)

    # Upload to GCS: one full CSV and one small JSON
    csv_object   = f"{GCS_PREFIX}/eisbach_live.csv"
    json_object  = f"{GCS_PREFIX}/latest_snapshot.json"

    upload_to_gcs(EISBACH_CSV, csv_object)
    upload_to_gcs(LATEST_JSON, json_object)


def main_loop():
    """
    Long‑lived loop: scrape + upload every 15 minutes.
    Use this if the script runs on a VM/home server.
    """
    while True:
        try:
            run_once()
        except Exception as e:
            print(f"[SCRAPER] Error in run_once: {e}")
        # Sleep until next 15‑minute interval
        time.sleep(15 * 60)


if __name__ == "__main__":
    main_loop()
