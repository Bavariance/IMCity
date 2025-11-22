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

class ModelTrainer(ABC):

    @abstractmethod
    def train_models(self):
        raise NotImplementedError



# TODO: Implement models

class EisbachEnsembleFlowTrainer(ModelTrainer):

    def __init__(self, filename):
        self.EISBACH_FILENAME = filename
        self.feature_cols: List[str]
        self.df = pd.DataFrame()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.last_predictions = []
        self.rf, self.xgb_model, self.lgb_model = self.init_models()
        self.read_eisbach_csv()


    # Reads the csv file filename and returns a dataframe with date and flow
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



    def train_models(self):
        # 1. Random Forest
        print("Starting training for Random Forest.")
        self.rf.fit(self.X_train, self.y_train)
        print("Training for Random Forest complete.")
        print("Starting training for XGBoost.")
        # 2. XGBoost
        self.xgb_model.fit(self.X_train, self.y_train)
        print("Training for XGB complete.")
        print("Starting training for LighGBM.")
        # 3. LightGBM
        self.lgb_model.fit(self.X_train, self.y_train)
        print("Training for LightGBM complete.")

    def init_models(self):
        rf = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )

        # XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

        # LightGBM
        lgb_model = lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

        return rf, xgb_model, lgb_model

    def build_ensemble_and_predict(self):
        # We predict the next value for X_test (input containing data of last 30 days)
        pred_rf = self.rf.predict(self.X_test)
        pred_xgb = self.xgb_model.predict(self.X_test)
        pred_lgb = self.lgb_model.predict(self.X_test)
        # Mean over all predictions is our final prediction
        ensemble_pred = (pred_xgb + pred_rf + pred_lgb) / 3
        return ensemble_pred

    def evaluate_error(y_test, ensemble_pred):
        mse = mean_squared_error(y_test, ensemble_pred)
        print("Mean Squared Error (MSE):", mse)

    def output_comparison(y_test, ensemble_pred):
        comparison = pd.DataFrame({
            'wasserstand_real': y_test,
            'ensemble_pred': ensemble_pred
        }, index=y_test.index)
        comparison["error"] = comparison["ensemble_pred"] - comparison["wasserstand_real"]
        output_path = "wasserstand_predictions_vs_actual.csv"
        comparison.to_csv(output_path, sep=';', decimal=',')

    def predict_next_time_step(self):
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

        return (pred_rf_next + pred_xgb_next + pred_lgb_next) / 3.0

    def save_models(self):
        # Random Forest speichern
        with open(MODELS_DIR / "rf_model.pkl", "wb") as f:
            pickle.dump(self.rf, f)

        # XGBoost speichern
        with open(MODELS_DIR / "xgb_model.pkl", "wb") as f:
            pickle.dump(self.xgb_model, f)

        # LightGBM speichern
        with open(MODELS_DIR / "lgb_model.pkl", "wb") as f:
            pickle.dump(self.lgb_model, f)

    def load_models(self):
        with open(MODELS_DIR / "rf_model.pkl", "rb") as f:
            rf = pickle.load(f)

        with open(MODELS_DIR / "xgb_model.pkl", "rb") as f:
            xgb_model = pickle.load(f)

        with open(MODELS_DIR / "lgb_model.pkl", "rb") as f:
            lgb_model = pickle.load(f)

        return rf, xgb_model, lgb_model


def main():
    model = EisbachEnsembleFlowTrainer("../data/flowrate_beginn_bis_31.12.2024_ezw_0.csv")
    model.train_models()

main()