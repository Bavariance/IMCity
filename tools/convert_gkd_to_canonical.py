# tools/convert_gkd_to_canonical.py

from pathlib import Path
import logging

import pandas as pd
import pytz

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eisbach.convert")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

RAW_LEVEL = DATA_DIR / "waterlevel_messwerte_gkd.csv"
RAW_FLOW  = DATA_DIR / "flowrate_messwerte_gkd.csv"
OUT_CSV   = DATA_DIR / "eisbach_15min.csv"

logger.info(f"PROJECT_ROOT={PROJECT_ROOT}")
logger.info(f"DATA_DIR={DATA_DIR}")
logger.info(f"RAW_LEVEL={RAW_LEVEL}")
logger.info(f"RAW_FLOW={RAW_FLOW}")
logger.info(f"OUT_CSV={OUT_CSV}")


def load_level() -> pd.DataFrame:
    logger.info("Loading level data from %s", RAW_LEVEL)
    df = pd.read_csv(
        RAW_LEVEL,
        sep=";",
        decimal=",",
        quotechar='"',
        encoding="utf-8",
    )
    logger.info("Level rows=%d, columns=%s", len(df), list(df.columns))
    df.rename(
        columns={
            "Datum": "timestamp_local",
            "Wasserstand [cm]": "waterlevel_cm",
            "Prüfstatus": "status_level",
        },
        inplace=True,
    )
    return df


def load_flow() -> pd.DataFrame:
    logger.info("Loading flow data from %s", RAW_FLOW)
    df = pd.read_csv(
        RAW_FLOW,
        sep=";",
        decimal=",",
        quotechar='"',
        encoding="utf-8",
    )
    logger.info("Flow rows=%d, columns=%s", len(df), list(df.columns))
    df.rename(
        columns={
            "Datum": "timestamp_local",
            "Abfluss [m³/s]": "discharge_m3s",
            "Prüfstatus": "status_flow",
        },
        inplace=True,
    )
    return df


def add_utc_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Adding UTC timestamps")
    tz = pytz.timezone("Europe/Berlin")
    ts_local = pd.to_datetime(df["timestamp_local"])
    ts_local = ts_local.dt.tz_localize(tz)
    ts_utc = ts_local.dt.tz_convert(pytz.UTC)
    df["timestamp_utc"] = ts_utc.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return df


def main():
    logger.info("Starting conversion to canonical CSV")
    level = load_level()
    flow = load_flow()

    logger.info("Merging level and flow on timestamp_local")
    merged = pd.merge(
        level,
        flow,
        on="timestamp_local",
        how="outer",
        sort=True,
    )
    logger.info("Merged rows=%d", len(merged))

    merged = add_utc_timestamp(merged)

    merged = merged[
        [
            "timestamp_local",
            "timestamp_utc",
            "waterlevel_cm",
            "discharge_m3s",
            "status_level",
            "status_flow",
        ]
    ]

    merged.to_csv(OUT_CSV, index=False)
    logger.info("Wrote canonical CSV %s with %d rows", OUT_CSV, len(merged))


if __name__ == "__main__":
    main()
