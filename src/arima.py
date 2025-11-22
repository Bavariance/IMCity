import logging
import sys
import os
from pathlib import Path
import numpy as np

# --- 1. Force NumPy to use all Cores ---
# This must be done before importing pandas/numpy to ensure BLAS libraries pick it up
os.environ["OMP_NUM_THREADS"] = "20"
os.environ["MKL_NUM_THREADS"] = "20"
os.environ["OPENBLAS_NUM_THREADS"] = "20"
os.environ["VECLIB_MAXIMUM_THREADS"] = "20"
os.environ["NUMEXPR_NUM_THREADS"] = "20"

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults

# -------------------------------------------------------------------
# Logging Configuration
# -------------------------------------------------------------------

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Paths & constants
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "trained_model.pkl"

FLOW_PATH = DATA_DIR / "flowrate_beginn_bis_31.12.2024_ezw_0.csv"
LEVEL_PATH = DATA_DIR / "waterlevel_beginn_bis_31.12.2024_ezw_0.csv"

FREQ = "15min"
DAILY_STEPS = 96
TRAIN_CUTOFF = 30000  # Approx 312 days of data

# -------------------------------------------------------------------
# Custom Progress Callback
# -------------------------------------------------------------------

def progress_callback(xk):
    """
    Callback function called after each iteration of the optimizer.
    Prints a dot to show the process is alive.
    """
    sys.stdout.write(".")
    sys.stdout.flush()

# -------------------------------------------------------------------
# Data loading & preprocessing
# -------------------------------------------------------------------

def load_hydro_data() -> pd.DataFrame:
    """Load flow and water level data, join them, and ensure no NaNs exist."""
    logger.info(f"Loading flow data from {FLOW_PATH}...")
    if not FLOW_PATH.exists():
        logger.error(f"Flow data file not found: {FLOW_PATH}")
        raise FileNotFoundError(f"{FLOW_PATH} does not exist.")

    flow = pd.read_csv(FLOW_PATH, sep=";", decimal=",", parse_dates=["Datum"], na_values=["Lücke", "--", ""])
    flow = (flow.rename(columns={"Datum": "timestamp", "Abfluss [m³/s]": "flow"})
            .set_index("timestamp").sort_index()[["flow"]])
    flow = flow[~flow.index.duplicated(keep='first')].asfreq(FREQ)

    logger.info(f"Loading level data from {LEVEL_PATH}...")
    if not LEVEL_PATH.exists():
        logger.error(f"Level data file not found: {LEVEL_PATH}")
        raise FileNotFoundError(f"{LEVEL_PATH} does not exist.")

    level = pd.read_csv(LEVEL_PATH, sep=";", decimal=",", parse_dates=["Datum"], na_values=["Lücke", "--", ""])
    level = (level.rename(columns={"Datum": "timestamp", "Wasserstand [cm]": "level"})
             .set_index("timestamp").sort_index()[["level"]])
    level = level[~level.index.duplicated(keep='first')].asfreq(FREQ)

    logger.info("Joining datasets and cleaning NaNs...")
    df = flow.join(level, how="inner")
    original_len = len(df)
    
    df = df.ffill(limit=4).dropna()
    df["flow"] = pd.to_numeric(df["flow"], errors="coerce")
    df["level"] = pd.to_numeric(df["level"], errors="coerce")
    df = df.dropna()
    
    logger.info(f"Data loaded successfully. Rows: {len(df)} (dropped {original_len - len(df)} NaNs)")
    return df

# -------------------------------------------------------------------
# Model fitting & Saving
# -------------------------------------------------------------------

def fit_level_model(
    df: pd.DataFrame,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, DAILY_STEPS),
    cutoff: int = TRAIN_CUTOFF
) -> ARIMAResults:
    """Fit SARIMAX model with progress monitoring."""
    
    if len(df) > cutoff:
        logger.info(f"Dataset size ({len(df)}) exceeds cutoff. Trimming to last {cutoff} rows.")
        df_train = df.tail(cutoff)
    else:
        df_train = df

    logger.info(f"Initializing SARIMAX{order}x{seasonal_order} on {len(df_train)} samples...")
    
    model = ARIMA(
        endog=df_train["level"],
        exog=df_train[["flow"]],
        order=order,
        seasonal_order=seasonal_order,
    )

    logger.info("Fitting model. Each dot '.' represents an optimization step.")
    print("Optimization Progress: ", end="")
    
    # FIX: Pass callback inside method_kwargs
    # This ensures it reaches the underlying SARIMAX fit method
    res = model.fit(
        low_memory=True,  # Helps on laptop
        method_kwargs={
            "disp": True,               # Show final convergence info
            "callback": progress_callback # Show live heartbeat
        }
    )
    
    print("\n") # Newline after dots
    logger.info("Model fitting complete.")
    return res

def save_model(res: ARIMAResults, path: Path = MODEL_PATH):
    """Save the fitted model to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving model to {path}...")
    try:
        res.save(path)
        logger.info("Model saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        raise

def load_model(path: Path = MODEL_PATH) -> ARIMAResults:
    """Load a pre-trained model from disk."""
    if not path.exists():
        logger.error(f"Model file not found at {path}")
        raise FileNotFoundError(f"No model found at {path}. Train it first!")
    logger.info(f"Loading model from {path}...")
    return ARIMAResults.load(path)

# -------------------------------------------------------------------
# Forecasting helpers
# -------------------------------------------------------------------

def forecast_level(
    res: ARIMAResults,
    n_steps: int = DAILY_STEPS,
    last_observed_flow: float = None,
    exog_future: pd.DataFrame = None,
) -> pd.Series:
    """Forecast water level."""
    if exog_future is None:
        if last_observed_flow is None:
             logger.error("Forecast called without flow data.")
             raise ValueError("Must provide last_observed_flow or exog_future")
             
        exog_future = pd.DataFrame(
            {"flow": [last_observed_flow] * n_steps},
            index=pd.RangeIndex(start=0, stop=n_steps)
        )

    logger.info(f"Generating forecast for next {n_steps} steps...")
    forecast = res.forecast(steps=n_steps, exog=exog_future)
    return forecast

# -------------------------------------------------------------------
# Execution
# -------------------------------------------------------------------

if __name__ == "__main__":
    try:
        logger.info("--- STARTING TRAINING PIPELINE ---")
        logger.info(f"Using {os.environ.get('OMP_NUM_THREADS')} threads for BLAS operations.")

        # 1. Train & Save Mode
        df_all = load_hydro_data()
        
        logger.info(f"Flow range: {df_all['flow'].min()} - {df_all['flow'].max()}")
        logger.info(f"Level range: {df_all['level'].min()} - {df_all['level'].max()}")
        
        model_res = fit_level_model(df_all)
        save_model(model_res)
        
        # 2. Inference Phase (Simulation)
        logger.info("\n--- STARTING INFERENCE SIMULATION ---")
        loaded_model = load_model()
        current_flow = df_all["flow"].iloc[-1]
        
        logger.info(f"Last observed flow: {current_flow} m³/s")
        prediction = forecast_level(
            loaded_model, 
            n_steps=96, 
            last_observed_flow=current_flow
        )
        
        print("\nForecast Output Head:")
        print(prediction.head())
        logger.info("Pipeline finished successfully.")

    except Exception as e:
        logger.critical(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)
