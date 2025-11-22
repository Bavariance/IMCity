
#This class is used to train and run ensemble models

from enum import Enum

x_train = {}
y_train = {}

class PredictionType(Enum):
    FLIGHTS = 1
    EISBACH = 2
    WEATHER = 3
    AIRPORT = 4
    ETFS = 5

def init_models():
    return 0

def crawl_data(prediction_type: PredictionType):
    return 0

def train_models(prediction_type: PredictionType, X_train, Y_train):
    return 0

def predict(X_new, prediction_type: PredictionType):
    return 0






