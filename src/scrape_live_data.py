# src/scrape_live_data.py

import os
import time
import json
import logging
from pathlib import Path

import pandas as pd
import requests
import pytz
from google.cloud import storage  # pip install google-cloud-storage

# ----------------- Logging setup -----------------

# Simple console logger; systemd will capture stdout/stderr into journal
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eisbach.scraper")

# ----------------- Local paths -----------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJECT_ROOT set to {PROJECT_ROOT}")

# Canonical 15‑minute time-series lives in data/
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CANONICAL_CSV = DATA_DIR / "eisbach_15min.csv"
logger.info(f"Canonical CSV path: {CANONICAL_CSV}")

# Live artifacts (small, for convenience) live in live_data/
LIVE_DIR = PROJECT_ROOT / "live_data"
LIVE_DIR.mkdir(parents=True, exist_ok=True)
LATEST_JSON = LIVE_DIR / "latest_snapshot.json"
logger.info(f"Latest JSON path: {LATEST_JSON}")

# ----------------- GCS config ------------------

GCS_BUCKET = os.environ.get("GCS_BUCKET")              # e.g. "imcity_live_data"
GCS_PREFIX = os.environ.get("GCS_PREFIX", "live_data")  # folder inside bucket

logger.info(f"GCS_BUCKET={GCS_BUCKET!r}, GCS_PREFIX={GCS_PREFIX!r}")

_storage_client: storage.Client | None = None


def get_storage_client() -> storage.Client:
    """Get a cached Storage client using ADC / VM service account."""
    global _storage_client
    if _storage_client is None:
        logger.info("Creating new google.cloud.storage.Client()")
        _storage_client = storage.Client()   # uses VM service account / ADC
        logger.info(
            f"Storage client project: {_storage_client.project!r}"
        )
    return _storage_client


def upload_to_gcs(local_path: Path, object_name: str) -> None:
    """
    Uploads local_path to gs://GCS_BUCKET/object_name.

    Logs detailed information and re-raises on failure so systemd sees errors.
    """
    if not GCS_BUCKET:
        logger.warning(
            "GCS_BUCKET not set; skipping upload of %s as %s",
            local_path,
            object_name,
        )
        return

    if not local_path.exists():
        logger.error("Local file %s does not exist; cannot upload", local_path)
        raise FileNotFoundError(local_path)

    client = get_storage_client()
    logger.info(
        "Uploading %s (%d bytes) to gs://%s/%s",
        local_path,
        local_path.stat().st_size,
        GCS_BUCKET,
        object_name,
    )
    try:
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(object_name)
        blob.upload_from_filename(str(local_path))
        logger.info(
            "[GCS] Uploaded %s -> gs://%s/%s",
            local_path,
            GCS_BUCKET,
            object_name,
        )
    except Exception as e:
        logger.exception(
            "Upload to gs://%s/%s failed: %s", GCS_BUCKET, object_name, e
        )
        raise

# ----------------- Scraping logic --------------

HND_BASE = (
    "https://www.hnd.bayern.de/pegel/isar/"
    "muenchen-himmelreichbruecke-16515005/tabelle"
)
logger.info(f"HND_BASE={HND_BASE}")


def _fetch_table(methode: str) -> pd.DataFrame:
    """
    Fetch the HND HTML table for 'wasserstand' or 'abfluss' and return as DataFrame.
    """
    params: dict[str, str] = {"methode": methode}
    if methode == "abfluss":
        params["setdiskr"] = "15"  # 15‑minute discharge values

    logger.info("Requesting HND table: methode=%s params=%s", methode, params)
    resp = requests.get(HND_BASE, params=params, timeout=15)
    logger.info("HND response: status=%s url=%s", resp.status_code, resp.url)
    resp.raise_for_status()

    tables = pd.read_html(resp.text, decimal=",")
    logger.info("Found %d tables for methode=%s", len(tables), methode)
    if not tables:
        raise RuntimeError(f"No tables found for methode={methode}")

    df = tables[0]
    logger.debug("Raw table head for %s:\n%s", methode, df.head())

    df = df.iloc[:, :2].copy()
    df.columns = ["timestamp_local_raw", "value"]
    return df


def fetch_eisbach_point() -> dict:
    """
    Scrape latest Wasserstand (cm) and Abfluss (m³/s) from HND and return a canonical dict.
    """
    logger.info("Fetching latest Wasserstand/Abfluss from HND")
    level_df = _fetch_table("wasserstand")
    flow_df = _fetch_table("abfluss")

    level_row = level_df.iloc[0]
    flow_row = flow_df.iloc[0]
    logger.info(
        "Latest rows: level=%s, flow=%s",
        level_row.to_dict(),
        flow_row.to_dict(),
    )

    tz_local = pytz.timezone("Europe/Berlin")
    ts_local_parsed = pd.to_datetime(
        level_row["timestamp_local_raw"], dayfirst=True
    )
    ts_local = tz_local.localize(ts_local_parsed)
    ts_utc = ts_local.astimezone(pytz.UTC)

    timestamp_local = ts_local.strftime("%Y-%m-%d %H:%M")
    timestamp_utc = ts_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    status_level = "Rohdaten"
    status_flow = "Rohdaten"

    row = {
        "timestamp_local": timestamp_local,
        "timestamp_utc": timestamp_utc,
        "waterlevel_cm": float(level_row["value"]),
        "discharge_m3s": float(flow_row["value"]),
        "status_level": status_level,
        "status_flow": status_flow,
    }
    logger.info("Canonical row: %s", row)
    return row


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
    mode = "a" if path.exists() else "w"
    logger.info(
        "Writing canonical row to %s (mode=%s, header=%s)",
        path,
        mode,
        not path.exists(),
    )
    df_row.to_csv(path, mode=mode, header=not path.exists(), index=False)


def write_latest_snapshot(row: dict) -> None:
    """Write the latest observation as a small JSON file in live_data/."""
    tmp = LATEST_JSON.with_suffix(".tmp")
    logger.info("Writing latest snapshot to %s (tmp=%s)", LATEST_JSON, tmp)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(row, f)
    os.replace(tmp, LATEST_JSON)


def run_once() -> None:
    """
    Single scrape‑and‑upload pass.
    """
    logger.info("=== run_once() starting ===")
    row = fetch_eisbach_point()

    append_canonical_row(CANONICAL_CSV, row)
    write_latest_snapshot(row)

    csv_object = f"{GCS_PREFIX}/eisbach_15min.csv"
    json_object = f"{GCS_PREFIX}/latest_snapshot.json"

    upload_to_gcs(CANONICAL_CSV, csv_object)
    upload_to_gcs(LATEST_JSON, json_object)
    logger.info("=== run_once() completed successfully ===")


def main_loop() -> None:
    """Long‑lived loop: scrape + upload every 15 minutes."""
    logger.info("Entering main_loop() with 15‑minute interval")
    while True:
        try:
            run_once()
        except Exception as e:
            logger.exception("Error in run_once: %s", e)
        time.sleep(15 * 60)


if __name__ == "__main__":
    main_loop()
