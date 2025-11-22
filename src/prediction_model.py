from abc import ABC, abstractmethod
from typing import List

import pandas as pd
from settings import MODELS_DIR
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_squared_error
import pickle
from settings import MODELS_DIR
import os
import io
from google.cloud import storage

class PredictionModel(ABC):

    @abstractmethod
    def predict_next_time_step(self):
        raise NotImplementedError


class EisbachEnsemblePredictionModel(PredictionModel):

    def __init__(self, filename):
        self.EISBACH_FILENAME = filename
        self.feature_cols: List[str]
        self.df = pd.DataFrame()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.last_predictions = []
        self.rf, self.xgb_model, self.lgb_model = self.load_models()


    def read_eisbach_csv(self):
        
        bucket_name = os.environ["GCS_BUCKET"]
        prefix = os.environ.get("GCS_PREFIX", "live_data")
        object_name = f"{prefix}/eisbach_15min.csv"

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(object_name)

        # Download CSV into memory and read with pandas
        raw_bytes = blob.download_as_bytes()
        self.df = pd.read_csv(io.BytesIO(raw_bytes))

        # Map to the internal schema used during training
        # Use local time column as datetime index
        self.df["datetime"] = pd.to_datetime(
            self.df["timestamp_local"], format="%Y-%m-%d %H:%M"
        )
        self.df["wasserstand"] = self.df["waterlevel_cm"].astype(float)

        # Sort and set index
        self.df = (
            self.df.sort_values("datetime")
                   .reset_index(drop=True)
                   .set_index("datetime")
        )

        # =========================
        # 2. Lag-Features bauen
        # =========================
        # 15-Minuten-Daten:
        #   lag_1  = 15 Minuten vorher
        #   lag_4  = 1 Stunde vorher
        #   lag_8  = 2 Stunden vorher
        #   lag_96 = 24 Stunden vorher


        lags = [1, 2, 3, 4, 96]

        for lag in lags:
            self.df[f'lag_{lag}'] = self.df['wasserstand'].shift(lag)

        self.df['roll_mean_1h'] = self.df['wasserstand'].rolling(window=4).mean()  # 1 h
        self.df['roll_mean_6h'] = self.df['wasserstand'].rolling(window=24).mean()  # 6 h
        self.df['roll_std_6h'] = self.df['wasserstand'].rolling(window=24).std()  # 6 h Std-Abw.

        # Zeit-Features (für Tages- / Wochenmuster)
        self.df['hour'] = self.df.index.hour
        self.df['dayofweek'] = self.df.index.dayofweek
        self.df['month'] = self.df.index.month
        self.df['minute'] = self.df.index.minute

        # Durch Shift + Rolling entstehen NaNs am Anfang -> entfernen
        self.df = self.df.dropna()

        # =========================
        # 5. Features (X) und Ziel (y)
        # =========================

        lag_cols = [f'lag_{lag}' for lag in lags]
        extra_cols = ['roll_mean_1h', 'roll_mean_6h', 'roll_std_6h',
                      'hour', 'dayofweek', 'month', 'minute']

        self.feature_cols = lag_cols + extra_cols

        X = self.df[self.feature_cols]
        y = self.df['wasserstand']
        # z.B. letzte 7 Tage als Test (7 * 24 * 4 Schritte)
        test_size = 7 * 24 * 4
        if len(self.df) <= test_size:
            raise ValueError("Zu wenige Daten für diesen Test-Split – test_size kleiner wählen.")

        self.X_train, self.X_test = X.iloc[:-test_size], X.iloc[-test_size:]
        self.y_train, self.y_test = y.iloc[:-test_size], y.iloc[-test_size:]

    def predict_next_time_step(self):
        self.read_eisbach_csv()

        # 10.1 Neuen Zeitstempel definieren
        next_timestamp = self.df.index.max() + pd.Timedelta(minutes=15)

        # 10.2 Hilfs-DataFrame mit letzter Zeile kopieren
        last_row = self.df.iloc[[-1]].copy()
        last_row.index = [next_timestamp]

        # 10.3 Lags für den neuen Zeitpunkt bauen:
        #     Für einen Step-ahead-Tree-Ansatz verwenden wir hier einfach
        #     dieselben Feature-Werte wie in der letzten Zeile,
        #     d.h. wir nehmen an, dass das Pattern der Lags/rolls sich
        #     für den nächsten Schritt kaum ändert.
        #     (Für 1-Step-Forward-Forecasting ist das ein üblicher pragmatischer Ansatz.)

        X_next = last_row[self.feature_cols]

        # 10.4 Ensemble-Vorhersage für genau diesen einen neuen Zeitpunkt

        pred_rf_next = self.rf.predict(X_next)[0]
        pred_xgb_next = self.xgb_model.predict(X_next)[0]
        pred_lgb_next = self.lgb_model.predict(X_next)[0]


        prediction = (pred_rf_next + pred_xgb_next + pred_lgb_next) / 3.0
        self.last_predictions.append(prediction)
        return prediction

    def load_models(self):
        with open(MODELS_DIR / "Level/rf_model.pkl", "rb") as f:
            rf = pickle.load(f)

        with open(MODELS_DIR / "Level/xgb_model.pkl", "rb") as f:
            xgb_model = pickle.load(f)

        with open(MODELS_DIR / "Level/lgb_model.pkl", "rb") as f:
            lgb_model = pickle.load(f)

        return rf, xgb_model, lgb_model

class EisbachEnsembleFlowPredictionModel(PredictionModel):

    def __init__(self, filename):
        self.EISBACH_FILENAME = filename
        self.feature_cols: List[str]
        self.df = pd.DataFrame()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.last_predictions = []
        self.rf, self.xgb_model, self.lgb_model = self.load_models()

    def read_eisbach_csv(self):
        self.df = pd.read_csv(
            self.EISBACH_FILENAME,
            sep=';',  # wie in deinem Beispiel
            encoding='utf-8'
        )
        # Spalten umbenennen
        self.df = self.df.rename(columns={
            'Datum': 'datetime',
            'Abfluss [m³/s]': 'abfluss'
        })

        # Datum/Zeit parsen
        self.df['datetime'] = pd.to_datetime(self.df['datetime'], format='%Y-%m-%d %H:%M')

        # Kommazahlen in float umwandeln (deutsches Komma)
        self.df['abfluss'] = self.df['abfluss'].astype(str).str.replace(',', '.').astype(float)

        # Nach Zeit sortieren und als Index setzen
        self.df = self.df.sort_values('datetime').reset_index(drop=True)
        self.df = self.df.set_index('datetime')

        # =========================
        # 2. Lag-Features bauen
        # =========================
        # 15-Minuten-Daten:
        #   lag_1  = 15 Minuten vorher
        #   lag_4  = 1 Stunde vorher
        #   lag_8  = 2 Stunden vorher
        #   lag_96 = 24 Stunden vorher

        lags = [1, 2, 3, 4, 96]

        for lag in lags:
            self.df[f'lag_{lag}'] = self.df['abfluss'].shift(lag)

        self.df['roll_mean_1h'] = self.df['abfluss'].rolling(window=4).mean()  # 1 h
        self.df['roll_mean_6h'] = self.df['abfluss'].rolling(window=24).mean()  # 6 h
        self.df['roll_std_6h'] = self.df['abfluss'].rolling(window=24).std()  # 6 h Std-Abw.

        # Zeit-Features (für Tages- / Wochenmuster)
        self.df['hour'] = self.df.index.hour
        self.df['dayofweek'] = self.df.index.dayofweek
        self.df['month'] = self.df.index.month


        # Durch Shift + Rolling entstehen NaNs am Anfang -> entfernen
        self.df = self.df.dropna()

        # =========================
        # 5. Features (X) und Ziel (y)
        # =========================

        lag_cols = [f'lag_{lag}' for lag in lags]
        extra_cols = ['roll_mean_1h', 'roll_mean_6h', 'roll_std_6h',
                      'hour', 'dayofweek', 'month']

        self.feature_cols = lag_cols + extra_cols

        X = self.df[self.feature_cols]
        y = self.df['abfluss']
        # z.B. letzte 7 Tage als Test (7 * 24 * 4 Schritte)
        test_size = 7 * 24 * 4
        if len(self.df) <= test_size:
            raise ValueError("Zu wenige Daten für diesen Test-Split – test_size kleiner wählen.")

        self.X_train, self.X_test = X.iloc[:-test_size], X.iloc[-test_size:]
        self.y_train, self.y_test = y.iloc[:-test_size], y.iloc[-test_size:]

    def predict_next_time_step(self):
        self.read_eisbach_csv()

        # 10.1 Neuen Zeitstempel definieren
        next_timestamp = self.df.index.max() + pd.Timedelta(minutes=15)

        # 10.2 Hilfs-DataFrame mit letzter Zeile kopieren
        last_row = self.df.iloc[[-1]].copy()
        last_row.index = [next_timestamp]

        # 10.3 Lags für den neuen Zeitpunkt bauen:
        #     Für einen Step-ahead-Tree-Ansatz verwenden wir hier einfach
        #     dieselben Feature-Werte wie in der letzten Zeile,
        #     d.h. wir nehmen an, dass das Pattern der Lags/rolls sich
        #     für den nächsten Schritt kaum ändert.
        #     (Für 1-Step-Forward-Forecasting ist das ein üblicher pragmatischer Ansatz.)

        X_next = last_row[self.feature_cols]

        # 10.4 Ensemble-Vorhersage für genau diesen einen neuen Zeitpunkt

        pred_rf_next = self.rf.predict(X_next)[0]
        pred_xgb_next = self.xgb_model.predict(X_next)[0]
        pred_lgb_next = self.lgb_model.predict(X_next)[0]

        prediction = (pred_rf_next + pred_xgb_next + pred_lgb_next) / 3.0
        self.last_predictions.append(prediction)
        return prediction

    def load_models(self):
        with open(MODELS_DIR / "Flow/rf_model.pkl", "rb") as f:
            rf = pickle.load(f)

        with open(MODELS_DIR / "Flow/xgb_model.pkl", "rb") as f:
            xgb_model = pickle.load(f)

        with open(MODELS_DIR / "Flow/lgb_model.pkl", "rb") as f:
            lgb_model = pickle.load(f)

        return rf, xgb_model, lgb_model

def main():
    model = EisbachEnsembleFlowPredictionModel("../data/waterlevel_beginn_bis_31.12.2024_ezw_0.csv")
    prediction = model.predict_next_time_step()
    print("prediction: ", prediction)
