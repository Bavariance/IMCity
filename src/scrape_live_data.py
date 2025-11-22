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

# Canonical 15‑minute time-series lives in live_data/
LIVE_DIR = PROJECT_ROOT / "live_data"
LIVE_DIR.mkdir(parents=True, exist_ok=True)
CANONICAL_CSV = LIVE_DIR / "eisbach_15min.csv"
LATEST_JSON = LIVE_DIR / "latest_snapshot.json"

logger.info("Canonical CSV path: %s", CANONICAL_CSV)
logger.info("Latest JSON path: %s", LATEST_JSON)

# ----------------- GCS config -----------------

GCS_BUCKET = os.environ.get("GCS_BUCKET")              # e.g. "imcity_live_data"
GCS_PREFIX = os.environ.get("GCS_PREFIX", "live_data")  # folder inside bucket
logger.info("GCS_BUCKET=%r, GCS_PREFIX=%r", GCS_BUCKET, GCS_PREFIX)

_storage_client: storage.Client | None = None

# Timezone used by HND / GKD (MEZ/MESZ)
TZ_LOCAL = pytz.timezone("Europe/Berlin")

# ----------------- GCS helpers -----------------


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
    # Ask HND explicitly for 15‑minute data of the last 1 day
    params: dict[str, str] = {
        "methode": methode,
        "setdiskr": "15",
        "days": "1",        # 24h window from HND side
    }
    logger.info("Requesting HND table: methode=%s params=%s", methode, params)
    resp = requests.get(HND_BASE, params=params, timeout=15)
    logger.info("HND response: status=%s url=%s", resp.status_code, resp.url)
    resp.raise_for_status()

    from io import StringIO
    # Let pandas parse German decimals correctly
    # decimal="," → comma is decimal separator; thousands="." → dot is thousands separator
    tables = pd.read_html(StringIO(resp.text), decimal=",", thousands=".")  # key fix
    logger.info("Found %d tables for methode=%s", len(tables), methode)
    if not tables:
        raise RuntimeError(f"No tables found for methode={methode}")

    df = tables[0]
    logger.debug("Raw table head for %s:\n%s", methode, df.head())

    # First two columns: timestamp (DD.MM.YYYY HH:MM) and value
    df = df.iloc[:, :2].copy()
    df.columns = ["timestamp_local_raw", "value_raw"]

    # value_raw is now already numeric or convertible to numeric
    df["value"] = pd.to_numeric(df["value_raw"], errors="coerce")
    if df["value"].isna().any():
        n_bad = df["value"].isna().sum()
        logger.warning(
            "Found %d non‑numeric values in %s table after decimal parsing",
            n_bad,
            methode,
        )

    # Convert to canonical local timestamp format
    ts = pd.to_datetime(df["timestamp_local_raw"], dayfirst=True)
    df["timestamp_local"] = ts.dt.strftime("%Y-%m-%d %H:%M")

    # Keep only canonical timestamp + value
    df = df[["timestamp_local", "value"]]

    # HND lists newest first; sort ascending so we can append naturally
    df = df.sort_values("timestamp_local").reset_index(drop=True)
    logger.info(
        "Fetched %d rows for %s; oldest=%s newest=%s (values: %.1f...%.1f)",
        len(df),
        methode,
        df["timestamp_local"].iloc[0],
        df["timestamp_local"].iloc[-1],
        df["value"].iloc[0],
        df["value"].iloc[-1],
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
    Append rows newer than existing data and keep only the last 24h
    (hard cap: max 96 points for 15‑minute data).
    """
    # Parse timestamps of the freshly scraped table
    ts_all = pd.to_datetime(table["timestamp_local"])

    # Determine cutoff based on existing data
    if path.exists():
        try:
            existing_df = pd.read_csv(path)
            if not existing_df.empty:
                last_ts_existing = pd.to_datetime(
                    existing_df["timestamp_local"]
                ).max()
                cutoff = last_ts_existing
            else:
                existing_df = pd.DataFrame()
                cutoff = ts_all.min() - dt.timedelta(minutes=15)
        except Exception as e:
            logger.warning("Failed to read existing CSV; seeding from scratch: %s", e)
            existing_df = pd.DataFrame()
            cutoff = ts_all.min() - dt.timedelta(minutes=15)
    else:
        existing_df = pd.DataFrame()
        cutoff = ts_all.min() - dt.timedelta(minutes=15)

    # Only take rows newer than what we already have
    mask_new = ts_all > cutoff
    new_rows = table.loc[mask_new].copy()
    logger.info(
        "Existing cutoff=%s; new rows to append=%d (from %d scraped rows)",
        cutoff,
        len(new_rows),
        len(table),
    )

    if new_rows.empty and not existing_df.empty:
        logger.info("No new 15‑minute intervals to append")
        # Still enforce 24h/96pt pruning on existing data
        full_df = existing_df
    elif new_rows.empty and existing_df.empty:
        logger.info("No data at all to write")
        return None
    else:
        # Add UTC timestamps for the new rows
        ts_local = pd.to_datetime(new_rows["timestamp_local"])
        ts_local = ts_local.dt.tz_localize(TZ_LOCAL)
        ts_utc = ts_local.dt.tz_convert(pytz.UTC)
        new_rows["timestamp_utc"] = ts_utc.dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Fill status columns if they are missing (for safety)
        if "status_level" not in new_rows.columns:
            new_rows["status_level"] = "Rohdaten"
        if "status_flow" not in new_rows.columns:
            new_rows["status_flow"] = "Rohdaten"

        # Combine existing and new
        if existing_df.empty:
            full_df = new_rows
        else:
            full_df = pd.concat([existing_df, new_rows], ignore_index=True)

    # --- Canonicalize, drop duplicates, and enforce 24h / 96‑point window ---

    full_df["timestamp_local_dt"] = pd.to_datetime(full_df["timestamp_local"])
    full_df = full_df.sort_values("timestamp_local_dt")

    # Drop duplicate timestamps, keep the latest values
    full_df = full_df.drop_duplicates(subset=["timestamp_local"], keep="last")

    # Truncate to last 24 hours in *naive* local time
    now_local = dt.datetime.now().replace(second=0, microsecond=0)
    cutoff_24h = now_local - dt.timedelta(hours=24)

    mask_24h = full_df["timestamp_local_dt"] >= cutoff_24h
    pruned_df = full_df.loc[mask_24h].copy()


     # Hard cap to last 96 points (15‑minute resolution over 24h)
    if len(pruned_df) > 96:
        pruned_df = pruned_df.tail(96)

    # If nothing remains after pruning, avoid IndexError and bail out cleanly
    if pruned_df.empty:
        logger.info(
            "No data to write after truncation; live CSV empty"
        )
        # Still write an empty file with header if you want a canonical schema
        pruned_df = full_df.head(0)[
            [
                "timestamp_local",
                "timestamp_utc",
                "waterlevel_cm",
                "discharge_m3s",
                "status_level",
                "status_flow",
            ]
        ]
        pruned_df.to_csv(path, mode="w", header=True, index=False)
        return None

    # Final ordering and column selection
    pruned_df = pruned_df.sort_values("timestamp_local_dt")
    pruned_df = pruned_df[
        [
            "timestamp_local",
            "timestamp_utc",
            "waterlevel_cm",
            "discharge_m3s",
            "status_level",
            "status_flow",
        ]
    ]

    logger.info(
        "After truncation to last 24h / 96 points, rows in live CSV=%d",
        len(pruned_df),
    )

    pruned_df.to_csv(path, mode="w", header=True, index=False)
    last_row = pruned_df.iloc[-1].to_dict()
    logger.info("Last row after truncation: %s", last_row)
    return last_row


# ----------------- Snapshot helper -----------------


def write_latest_snapshot(row: dict) -> None:
    """Write the latest observation as a small JSON file in live_data/."""
    tmp = LATEST_JSON.with_suffix(".tmp")
    logger.info("Writing latest snapshot to %s", LATEST_JSON)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(row, f)
    os.replace(tmp, LATEST_JSON)

# ----------------- Orchestration -----------------


def run_once() -> None:
    """
    Single scrape‑and‑upload pass.
    """
    logger.info("=== run_once() starting ===")
    table = fetch_eisbach_table()
    
    # This function now handles loading, merging, and pruning to 24h
    last_row = append_new_rows(CANONICAL_CSV, table)

    # If append_new_rows returned a row, it means the file was updated (or seeded)
    if last_row is not None:
        write_latest_snapshot(last_row)
        csv_object = f"{GCS_PREFIX}/eisbach_15min.csv"
        json_object = f"{GCS_PREFIX}/latest_snapshot.json"
        upload_to_gcs(CANONICAL_CSV, csv_object)
        upload_to_gcs(LATEST_JSON, json_object)
    else:
        logger.info("No new data appended; skipping upload.")

    logger.info("=== run_once() completed ===")


def _sleep_until_next_quarter_hour():
    """
    Sleep until the next 15‑minute boundary + 1 second in Europe/Berlin.
    """
    now = dt.datetime.now(TZ_LOCAL)
    current_minute = now.minute
    next_quarter = ((current_minute // 15) + 1) * 15
    if next_quarter >= 60:
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
    run_once()
    while True:
        _sleep_until_next_quarter_hour()
        try:
            run_once()
        except Exception as e:
            logger.exception("Error in run_once: %s", e)


if __name__ == "__main__":
    main_loop()
