
#This class is used to train and run ensemble models

from enum import Enum
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_squared_error

EISBACH_FILENAME = "../data/waterlevel_beginn_bis_31.12.2024_ezw_0.csv"


class PredictionType(Enum):
    FLIGHTS = 1
    EISBACH = 2
    WEATHER = 3
    AIRPORT = 4
    ETFS = 5


#Reads the csv file filename and returns a dataframe with date and wasserstand
def read_eisbach_csv(df, filename):
    df = pd.read_csv(
        filename,
        sep=';',  # wie in deinem Beispiel
        encoding='utf-8'
    )
    # Spalten umbenennen
    df = df.rename(columns={
        'Datum': 'datetime',
        'Wasserstand [cm]': 'wasserstand'
    })

    # Datum/Zeit parsen
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M')

    # Kommazahlen in float umwandeln (deutsches Komma)
    df['wasserstand'] = df['wasserstand'].astype(str).str.replace(',', '.').astype(float)

    # Nach Zeit sortieren und als Index setzen
    df = df.sort_values('datetime').reset_index(drop=True)
    df = df.set_index('datetime')

    # =========================
    # 2. Lag-Features bauen
    # =========================
    # 15-Minuten-Daten:
    #   lag_1  = 15 Minuten vorher
    #   lag_4  = 1 Stunde vorher
    #   lag_8  = 2 Stunden vorher
    #   lag_96 = 24 Stunden vorher


    lags = [1, 4, 8, 96]

    for lag in lags:
        df[f'lag_{lag}'] = df['wasserstand'].shift(lag)

    # Zeilen mit NaN (am Anfang) entfernen
    df = df.dropna()

    feature_cols = [f'lag_{lag}' for lag in lags]
    X = df[feature_cols]
    y = df['wasserstand']

    # z.B. letzte 7 Tage als Test (7 * 24 * 4 Schritte)
    test_size = 365 * 24 * 4
    if len(df) <= test_size:
        raise ValueError("Zu wenige Daten für diesen Test-Split – test_size kleiner wählen.")

    X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
    y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

    return df, X_train, X_test, y_train, y_test


def train_models(X_train, Y_train, rf, xgb_model, lgb_model):
    # 1. Random Forest
    print("Starting training for Random Forest.")
    rf.fit(X_train, Y_train)
    print("Training for Random Forest complete.")
    print("Starting training for XGBoost.")
    # 2. XGBoost
    xgb_model.fit(X_train, Y_train)
    print("Training for XGB complete.")
    print("Starting training for LighGBM.")
    # 3. LightGBM
    lgb_model.fit(X_train, Y_train)
    print("Training for LightGBM complete.")

def init_models():
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


def build_ensemble_and_predict(X_test, rf, xgb_model, lgb_model):
    #We predict the next value for X_test (input containing data of last 30 days)
    pred_rf = rf.predict(X_test)
    pred_xgb = xgb_model.predict(X_test)
    pred_lgb = lgb_model.predict(X_test)
    #Mean over all predictions is our final prediction
    ensemble_pred = (pred_rf + pred_xgb + pred_lgb) / 3
    return ensemble_pred

def evaluate_error(y_test, ensemble_pred):
    mse = mean_squared_error(y_test, ensemble_pred)
    print("Mean Squared Error (MSE):", mse)



def main():
    df = pd.DataFrame()
    df, X_train, X_test, y_train, y_test = read_eisbach_csv(df, EISBACH_FILENAME)
    rf, xgb_model, lgb_model = init_models()
    train_models(X_train, y_train, rf, xgb_model, lgb_model)
    ensemble_pred = build_ensemble_and_predict(X_test, rf, xgb_model, lgb_model)
    evaluate_error(y_test, ensemble_pred)


main()




