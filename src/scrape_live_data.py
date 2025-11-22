import os
import time
import json
import logging
import datetime as dt
from pathlib import Path

import pandas as pd
import requests
import pytz
from google.cloud import storage  # pip install google-cloud-storage

# ----------------- Logging setup -----------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eisbach.scraper")

# ----------------- Local paths -----------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
logger.info("PROJECT_ROOT set to %s", PROJECT_ROOT)

# Canonical 15‑minute time-series lives in data/
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CANONICAL_CSV = DATA_DIR / "eisbach_15min.csv"
logger.info("Canonical CSV path: %s", CANONICAL_CSV)

# Live artifacts (small, for convenience) live in live_data/
LIVE_DIR = PROJECT_ROOT / "live_data"
LIVE_DIR.mkdir(parents=True, exist_ok=True)
LATEST_JSON = LIVE_DIR / "latest_snapshot.json"
logger.info("Latest JSON path: %s", LATEST_JSON)

# ----------------- GCS config -----------------

GCS_BUCKET = os.environ.get("GCS_BUCKET")              # e.g. "imcity_live_data"
GCS_PREFIX = os.environ.get("GCS_PREFIX", "live_data")  # folder inside bucket

logger.info("GCS_BUCKET=%r, GCS_PREFIX=%r", GCS_BUCKET, GCS_PREFIX)

_storage_client: storage.Client | None = None

# Timezone used by HND / GKD (MEZ/MESZ)
TZ_LOCAL = pytz.timezone("Europe/Berlin")


def get_storage_client() -> storage.Client:
    """Get a cached Storage client using ADC / VM service account."""
    global _storage_client
    if _storage_client is None:
        logger.info("Creating new google.cloud.storage.Client()")
        _storage_client = storage.Client()
        logger.info("Storage client project: %r", _storage_client.project)
    return _storage_client


def upload_to_gcs(local_path: Path, object_name: str) -> None:
    """
    Upload local_path to gs://GCS_BUCKET/object_name.
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
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(object_name)
    blob.upload_from_filename(str(local_path))
    logger.info(
        "[GCS] Uploaded %s -> gs://%s/%s",
        local_path,
        GCS_BUCKET,
        object_name,
    )

# ----------------- Scraping logic --------------


HND_BASE = (
    "https://www.hnd.bayern.de/pegel/isar/"
    "muenchen-himmelreichbruecke-16515005/tabelle"
)
logger.info("HND_BASE=%s", HND_BASE)


def _fetch_full_table(methode: str) -> pd.DataFrame:
    """
    Fetch the *full* 15‑minute HND HTML table for 'wasserstand' or 'abfluss'
    and return a DataFrame with canonical timestamp strings.

    Result columns:
        - timestamp_local: 'YYYY-MM-DD HH:MM'
        - value: float (cm or m³/s)
    """
    params: dict[str, str] = {"methode": methode, "setdiskr": "15"}
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

    # First two columns: timestamp (DD.MM.YYYY HH:MM) and value
    df = df.iloc[:, :2].copy()
    df.columns = ["timestamp_local_raw", "value"]

    # Convert to canonical local timestamp format
    ts = pd.to_datetime(df["timestamp_local_raw"], dayfirst=True)
    df["timestamp_local"] = ts.dt.strftime("%Y-%m-%d %H:%M")

    # Keep only canonical timestamp + value
    df = df[["timestamp_local", "value"]]

    # HND lists newest first; sort ascending so we can append naturally
    df = df.sort_values("timestamp_local").reset_index(drop=True)
    logger.info(
        "Fetched %d rows for %s; oldest=%s newest=%s",
        len(df),
        methode,
        df["timestamp_local"].iloc[0],
        df["timestamp_local"].iloc[-1],
    )
    return df


def fetch_eisbach_table() -> pd.DataFrame:
    """
    Fetch full 15‑minute Wasserstand and Abfluss tables and merge them
    into a single DataFrame.

    Result columns:
        - timestamp_local
        - waterlevel_cm
        - discharge_m3s
        - status_level
        - status_flow
    """
    logger.info("Fetching full 15‑minute Wasserstand + Abfluss tables")
    level_df = _fetch_full_table("wasserstand")
    flow_df = _fetch_full_table("abfluss")

    merged = pd.merge(
        level_df,
        flow_df,
        on="timestamp_local",
        how="outer",
        suffixes=("_level", "_flow"),
    ).sort_values("timestamp_local")

    merged.rename(
        columns={
            "value_level": "waterlevel_cm",
            "value_flow": "discharge_m3s",
        },
        inplace=True,
    )

    # Near-real-time data from HND/GKD is typically 'Rohdaten'
    merged["status_level"] = "Rohdaten"
    merged["status_flow"] = "Rohdaten"

    logger.info("Merged table rows=%d", len(merged))
    return merged


def append_new_rows(path: Path, table: pd.DataFrame) -> dict | None:
    """
    Append all rows from 'table' whose timestamp_local is newer
    than the max timestamp already in 'path'.

    On first run (no existing CSV), the *entire* scraped table is written.
    """
    ts_all = pd.to_datetime(table["timestamp_local"])

    # Determine cutoff based on existing data (if any)
    if path.exists():
        try:
            existing = pd.read_csv(path, usecols=["timestamp_local"])
            if not existing.empty:
                last_ts_existing = pd.to_datetime(
                    existing["timestamp_local"]
                ).max()
            else:
                last_ts_existing = None
        except Exception as e:
            logger.warning("Failed to read existing CSV for cutoff: %s", e)
            last_ts_existing = None
    else:
        last_ts_existing = None

    if last_ts_existing is not None:
        # Append anything strictly newer than last existing timestamp
        mask = ts_all > last_ts_existing
        new_rows = table.loc[mask].copy()
        logger.info(
            "Existing last timestamp=%s; new rows to append=%d",
            last_ts_existing,
            len(new_rows),
        )
    else:
        # First run: write the full table as returned by HND
        new_rows = table.copy()
        logger.info(
            "No existing data; seeding CSV with all %d scraped rows",
            len(new_rows),
        )

    if new_rows.empty:
        logger.info("No new 15‑minute intervals to append")
        return None

    # Add UTC timestamps
    ts_local = pd.to_datetime(new_rows["timestamp_local"])
    ts_local = ts_local.dt.tz_localize(TZ_LOCAL)
    ts_utc = ts_local.dt.tz_convert(pytz.UTC)
    new_rows["timestamp_utc"] = ts_utc.dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Reorder columns to canonical schema
    new_rows = new_rows[
        [
            "timestamp_local",
            "timestamp_utc",
            "waterlevel_cm",
            "discharge_m3s",
            "status_level",
            "status_flow",
        ]
    ]

    mode = "a" if path.exists() else "w"
    logger.info(
        "Appending %d new rows to %s (mode=%s, header=%s)",
        len(new_rows),
        path,
        mode,
        not path.exists(),
    )
    new_rows.to_csv(path, mode=mode, header=not path.exists(), index=False)

    last_row = new_rows.iloc[-1].to_dict()
    logger.info("Last appended row: %s", last_row)
    return last_row


# ----------------- Orchestration -----------------


def run_once() -> None:
    """
    Single scrape‑and‑upload pass.

    - Scrape full 15‑minute tables from HND.
    - Append only new 15‑minute intervals to data/eisbach_15min.csv.
    - Update JSON snapshot with the last appended row.
    - Upload canonical CSV + JSON to GCS.
    """
    logger.info("=== run_once() starting ===")
    table = fetch_eisbach_table()
    last_row = append_new_rows(CANONICAL_CSV, table)

    if last_row is not None:
        write_latest_snapshot(last_row)
        csv_object = f"{GCS_PREFIX}/eisbach_15min.csv"
        json_object = f"{GCS_PREFIX}/latest_snapshot.json"
        upload_to_gcs(CANONICAL_CSV, csv_object)
        upload_to_gcs(LATEST_JSON, json_object)
    else:
        logger.info("Skipping GCS upload because no new data was appended")

    logger.info("=== run_once() completed ===")


def _sleep_until_next_quarter_hour():
    """
    Sleep until the next 15‑minute boundary + 1 second in Europe/Berlin.

    Example: if now is 16:07, sleep until 16:15:01; if 16:16, sleep until 16:30:01.
    """
    now = dt.datetime.now(TZ_LOCAL)
    # Determine next quarter: 0,15,30,45
    current_minute = now.minute
    next_quarter = ((current_minute // 15) + 1) * 15
    if next_quarter >= 60:
        # roll over to next hour
        next_time = (now + dt.timedelta(hours=1)).replace(
            minute=0, second=1, microsecond=0
        )
    else:
        next_time = now.replace(
            minute=next_quarter, second=1, microsecond=0
        )

    sleep_seconds = (next_time - dt.datetime.now(TZ_LOCAL)).total_seconds()
    if sleep_seconds < 0:
        sleep_seconds = 1  # safety fallback

    logger.info(
        "Sleeping %.1f seconds until next run at local time %s",
        sleep_seconds,
        next_time,
    )
    time.sleep(sleep_seconds)


def main_loop() -> None:
    """
    Long‑lived loop: initial seed, then scrape at 15‑minute boundaries.
    """
    logger.info("Entering main_loop()")
    # Initial run immediately on startup (seeds last 24h if CSV empty)
    run_once()
    while True:
        _sleep_until_next_quarter_hour()
        try:
            run_once()
        except Exception as e:
            logger.exception("Error in run_once: %s", e)


if __name__ == "__main__":
    main_loop()
