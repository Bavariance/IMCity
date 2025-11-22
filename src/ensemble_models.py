
#This class is used to train and run ensemble models

from enum import Enum
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_squared_error

X_train, X_test, Y_train, Y_test = pd.DataFrame()
EISBACH_FILENAME = "data/eisbach.csv"


class PredictionType(Enum):
    FLIGHTS = 1
    EISBACH = 2
    WEATHER = 3
    AIRPORT = 4
    ETFS = 5

def init_models():
    return 0

#Reads the csv file filename and returns a dataframe with date and wasserstand
def read_eisbach_csv(filename):
    df = pd.read_csv(filename, sep=';', encoding='utf-8', skiprows=range(0, 13))
    df = df.rename(columns={'Datum': 'date', 'Mittelwert': 'wasserstand'})
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df['wasserstand'] = df['wasserstand'].str.replace(',', '.').astype(float)
    df = df[['date', 'wasserstand']]


    #We define a lag for the last three days ()
    for lag in [1, 2, 3]:
        df[f'lag_{lag}'] = df['wasserstand'].shift(lag)

    # Fehlende Werte entfernen (am Anfang fehlen Lags!)
    df = df.dropna()

    return df

def init_data(df):
    x = df['lag_1', 'lag_2', 'lag_3']
    y = df['wasserstand']

    #We need to split training and testing data (last 30 days)
    #Training data are the first 30 samples and test data the last 30 samples
    X_train, X_test = x[:-30], x[-30:]
    Y_train, Y_test = y[:-30], y[-30:]

    return X_train, X_test, Y_train, Y_test

def train_models(X_train, Y_train, rf, xgb_model, lgb_model):
    # 1. Random Forest
    rf.fit(X_train, Y_train)
    # 2. XGBoost
    xgb_model.fit(X_train, Y_train)
    # 3. LightGBM
    lgb_model.fit(X_train, Y_train)


def build_ensemble(rf, xgb_model, lgb_model):
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



def predict(x_new, prediction_type: PredictionType):
    return 0


if __name__ == "__main__":
    df_raw = pd.read_csv(EISBACH_FILENAME)






