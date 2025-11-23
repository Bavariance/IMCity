import requests
from datetime import datetime, timedelta
import pandas as pd



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




# Open-Meteo API - kostenlos, keine API-Key nötig
def get_munich_weather():
    """Holt aktuelle und historische Wetterdaten für München"""

    # München Koordinaten
    lat, lon = 48.1351, 11.5820

    # Zeitraum: letzte 2 Stunden (für 30-min Intervalle)
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=2)

    # Open-Meteo API
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["temperature_2m", "relative_humidity_2m"],
        "start_date": start_time.strftime("%Y-%m-%d"),
        "end_date": end_time.strftime("%Y-%m-%d"),
        "timezone": "Europe/Berlin"
    }

    response = requests.get(url, params=params)
    data = response.json()

    # Daten verarbeiten
    times = data["hourly"]["time"]
    temps_c = data["hourly"]["temperature_2m"]
    humidity = data["hourly"]["relative_humidity_2m"]

    # Celsius zu Fahrenheit
    temps_f = [(t * 9 / 5) + 32 if t is not None else None for t in temps_c]

    # DataFrame erstellen
    df = pd.DataFrame({
        "Date": pd.to_datetime(times),
        "Temperature": temps_f,
        "Humidity": humidity
    })

    # Nur gültige Werte (keine None)
    df = df.dropna()

    # Auf 30-min Intervalle resamplen (nimm letzte 3 Werte)
    df = df.tail(3)

    return df

def get_data_weather_and_store_file():
    # Wetterdaten holen
    df = get_munich_weather()

    print("Munich Weather Data (Real-time)")
    print("=" * 60)
    print(df.to_string(index=False))
    print("\n")

    # Settlement Berechnung: sum of (temperature*2 + humidity) für alle Intervalle
    if len(df) >= 3:
        settlement = sum(row["Temperature"] * 2 + row["Humidity"] for _, row in df.iterrows())

        print("Market Settlement Calculation")
        print("=" * 60)
        for idx, row in df.iterrows():
            calc = row["Temperature"] * 2 + row["Humidity"]
            print(f"{row['Date']}: ({row['Temperature']:.1f}*2 + {row['Humidity']:.0f}) = {calc:.1f}")

        print(f"\nSettlement Value: {settlement:.1f}")
    else:
        print("Nicht genug Daten für Settlement-Berechnung")

    # Daten speichern
    df.to_csv("../data/munich_weather_data.csv", index=False)
    print(f"\n✓ Daten gespeichert in: munich_weather_data.csv")

# TODO: Implement models

class WeatherModel(ModelTrainer):

    def __init__(self, filename):
        self.WEATHER_FILE = filename
        self.feature_cols: List[str]
        self.df = pd.DataFrame()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.last_predictions = []
        self.rf, self.xgb_model, self.lgb_model = self.init_models()
        self.read_file()


    # Reads the csv file filename and returns a dataframe with date and temperature and humidity
    def read_file(self):
        """Liest Wetterdaten und berechnet Settlement-Wert als Ziel"""
        get_data_weather_and_store_file()
        self.df = pd.read_csv(self.WEATHER_FILE)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.sort_values('Date').reset_index(drop=True)
        self.df = self.df.set_index('Date')

        # Settlement-Wert berechnen: Temperature*2 + Humidity
        self.df['settlement'] = self.df['Temperature'] * 2 + self.df['Humidity']

        # Lag-Features (30-min Intervalle simuliert)
        lags = [1, 2, 3, 4, 6, 12]  # 30min, 1h, 1.5h, 2h, 3h, 6h zurück
        for lag in lags:
            self.df[f'lag_{lag}'] = self.df['settlement'].shift(lag)

        # Rolling-Features
        self.df['roll_mean_2h'] = self.df['settlement'].rolling(window=4).mean()
        self.df['roll_mean_6h'] = self.df['settlement'].rolling(window=12).mean()
        self.df['roll_std_6h'] = self.df['settlement'].rolling(window=12).std()

        # Zeit-Features
        self.df['hour'] = self.df.index.hour
        self.df['dayofweek'] = self.df.index.dayofweek
        self.df['month'] = self.df.index.month
        self.df['minute'] = self.df.index.minute

        # NaNs entfernen
        self.df = self.df.dropna()

        # Features definieren
        lag_cols = [f'lag_{lag}' for lag in lags]
        extra_cols = ['roll_mean_2h', 'roll_mean_6h', 'roll_std_6h',
                      'hour', 'dayofweek', 'month', 'minute']
        self.feature_cols = lag_cols + extra_cols

        X = self.df[self.feature_cols]
        y = self.df['settlement']

        # Train/Test Split (80/20)
        split_idx = int(len(self.df) * 0.8)
        self.X_train, self.X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        self.y_train, self.y_test = y.iloc[:split_idx], y.iloc[split_idx:]

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

    def evaluate_error(self, y_test, ensemble_pred):
        mse = mean_squared_error(y_test, ensemble_pred)
        print("Mean Squared Error (MSE):", mse)

    def output_comparison(self, y_test, ensemble_pred):
        comparison = pd.DataFrame({
            'wasserstand_real': y_test,
            'ensemble_pred': ensemble_pred
        }, index=y_test.index)
        comparison["error"] = comparison["ensemble_pred"] - comparison["wasserstand_real"]
        output_path = "wasserstand_predictions_vs_actual.csv"
        comparison.to_csv(output_path, sep=';', decimal=',')

    def predict_next_time_step(self):
        self.read_file()
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

    def predict_until_target(self, target_datetime):
        self.read_file()
        """Predicts settlement values until target datetime (Sonntag 10:00)"""
        current_time = self.df.index.max()
        predictions = []

        # Kopie der letzten Zeile für iterative Prediction
        last_row = self.df.iloc[[-1]].copy()

        pred = 0
        while current_time < target_datetime:
            current_time += pd.Timedelta(minutes=30)  # 30-min Intervalle

            # Features für nächsten Zeitpunkt bauen
            X_next = last_row[self.feature_cols]

            # Ensemble-Prediction
            pred_rf = self.rf.predict(X_next)[0]
            pred_xgb = self.xgb_model.predict(X_next)[0]
            pred_lgb = self.lgb_model.predict(X_next)[0]
            pred = (pred_rf + pred_xgb + pred_lgb) / 3.0

            predictions.append({
                'datetime': current_time,
                'settlement_pred': pred
            })

            # Update last_row für nächste Iteration (Lags verschieben)
            last_row['settlement'].iloc[0] = pred
            last_row.index = [current_time]
            last_row['hour'] = current_time.hour
            last_row['minute'] = current_time.minute
            last_row['dayofweek'] = current_time.dayofweek

        return pred

    def save_models(self):
        # Random Forest speichern
        with open(MODELS_DIR / "Weather/rf_model.pkl", "wb") as f:
            pickle.dump(self.rf, f)

        # XGBoost speichern
        with open(MODELS_DIR / "Weather/xgb_model.pkl", "wb") as f:
            pickle.dump(self.xgb_model, f)

        # LightGBM speichern
        with open(MODELS_DIR / "Weather/lgb_model.pkl", "wb") as f:
            pickle.dump(self.lgb_model, f)

    def load_models(self):
        with open(MODELS_DIR / "Weather/rf_model.pkl", "rb") as f:
            rf = pickle.load(f)

        with open(MODELS_DIR / "Weather/xgb_model.pkl", "rb") as f:
            xgb_model = pickle.load(f)

        with open(MODELS_DIR / "Weather/lgb_model.pkl", "rb") as f:
            lgb_model = pickle.load(f)

        return rf, xgb_model, lgb_model


def main():
    model = WeatherModel("../data/munich_weather_data.csv")
    model.train_models()
    model.save_models()
    pred = model.build_ensemble_and_predict()
    model.evaluate_error(model.y_test, pred)
main()

