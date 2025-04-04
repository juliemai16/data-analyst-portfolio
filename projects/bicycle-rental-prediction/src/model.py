from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb

def create_tf_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def create_lgb_model():
    model = LGBMRegressor(n_estimators=1000)
    return model

def create_rf_model():
    model = RandomForestRegressor()
    return model

def create_gb_model():
    model = GradientBoostingRegressor()
    return model

def create_xgb_model():
    model = xgb.XGBRegressor()
    return model
