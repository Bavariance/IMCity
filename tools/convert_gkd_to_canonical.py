# tools/convert_gkd_to_canonical.py

from pathlib import Path
import pandas as pd
import pytz

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# raw GKD downloads (adjust filenames)
RAW_LEVEL = DATA_DIR / "waterlevel_messwerte_gkd.csv"
RAW_FLOW  = DATA_DIR / "flowrate_messwerte_gkd.csv"

OUT_CSV   = DATA_DIR / "eisbach_15min.csv"


def load_level() -> pd.DataFrame:
    # Example structure:
    # Datum;"Wasserstand [cm]";Prüfstatus
    # "1964-11-01 00:00";117,00;Geprueft
    df = pd.read_csv(
        RAW_LEVEL,
        sep=";",
        decimal=",",
        quotechar='"',
        encoding="utf-8",
    )
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
    # Example structure:
    # Datum;"Abfluss [m³/s]";Prüfstatus
    # "1970-11-01 00:00";24,4;Geprueft
    df = pd.read_csv(
        RAW_FLOW,
        sep=";",
        decimal=",",
        quotechar='"',
        encoding="utf-8",
    )
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
    tz = pytz.timezone("Europe/Berlin")
    ts_local = pd.to_datetime(df["timestamp_local"])
    ts_local = tz.localize(ts_local)
    ts_utc = ts_local.astimezone(pytz.UTC)
    df["timestamp_utc"] = ts_utc.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return df


def main():
    level = load_level()
    flow = load_flow()

    # Outer join in case level/flow coverage differs slightly
    merged = pd.merge(
        level,
        flow,
        on="timestamp_local",
        how="outer",
        sort=True,
    )

    merged = add_utc_timestamp(merged)

    # Reorder columns
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
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
