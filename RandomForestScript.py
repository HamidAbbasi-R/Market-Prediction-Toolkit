#%% IMPORT LIBRARIES
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime
import functions as fns
# reload the functions module
import importlib
importlib.reload(fns)

#%% Step 0: Set up parameters and get data
print("Setting up parameters...", end="")
# data parameters
symbol='EURUSD'        # 'EURUSD' or 'MSFT.US' or 'MSFT.US-24'
fromNow = True
timeFrame = 'H4'
Nbars = 5000
endTime = datetime(2024, 10, 30, 16, 0, 0)  # in case fromNow is False
MA_period = 20

# Simulated dataset
artificial_returns = False
N_repeat = 2
means_generative = [0.002, -0.002, 0]*N_repeat
covars_generative = [0.02, 0.02, 0.02]*N_repeat
durations = [500,500,500]*N_repeat

# ML parameters
lookback = 5
features_dict = {
    # price
    'close':            False,
    'open':             False,
    'high':             False,
    'low':              False,
    'upward':           False,

    # time
    'hour':             False,

    # log return
    'log_return':       False,
    'MA_log_return':    False,
    'EMA_log_return':   True,
    
    # volume
    'volume':           False,
    'MA_volume':        False,
    'EMA_volume':       False,
    
    # volatility
    'volatility':       False,
    'MA_volatility':    False,
    'EMA_volatility':   True,

    # Technical indicators
    'ATR':              False,
    'ADX':              False,
    'RSI':              False,
    
    # Macro features
    'US500':            False,
    'IYR.US':           False,
    'SHY.US':           False,
}

target = 'EMA_log_return'
forward_target = 0      # 0 means the same day, 1 means the next day, etc.
test_split = 0.5        # 0.7 means 70% test, 30% train

# get the data
if fromNow:
    endTime = datetime.now()
data = fns.GetPriceData(
    symbol, 
    endTime, 
    timeFrame, 
    Nbars, 
    indicators_dict={
        'ATR':   features_dict['ATR'],
        'ADX':   features_dict['ADX'],
        'RSI':   features_dict['RSI'],
    },
    MA_period=MA_period)

for feature in ['US500', 'IYR.US', 'SHY.US']:
    if features_dict[feature]:
        # data[feature] = fns.GetPriceData(feature, endTime, timeFrame, Nbars)['close']
        data_feature = fns.GetPriceData(feature, endTime, timeFrame, Nbars)
        if len(data_feature) != len(data):
            raise ValueError(f"Data length mismatch: {feature}")
        data[feature] = data_feature['close']

#%% Step 1: Preprocess the data
# Generate lagged features
data = fns.create_features_and_target(
    lookback, 
    features_dict, 
    data, 
    forward_target,
    target,
    )

# Features (X) and target (y)
X = fns.prepare_feature_matrix(
    lookback, 
    features_dict, 
    data)

# forward looking target
y_scaled = data['target'].values
data.dropna(inplace=True)

# Scale data
scaler_X = StandardScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y_scaled.reshape(-1, 1))

# Plot features correlation matrix
fns.plot_features_correlation_matrix(X, features_dict, lookback)
fns.plot_binary_features_relations(3, 4, X, features_dict, lookback)

# Split data into training and testing sets
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(
    X_scaled, y_scaled, test_size=test_split, 
    # random_state=42, 
    # stratify=y,       # stratification is used for classification problems
    )

# inverse transform the scaled data
y_test = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1))
y_train = scaler_y.inverse_transform(y_train_scaled.reshape(-1, 1))

#%% Step 2: Train Random Forest Regressor
model = RandomForestRegressor(
    n_estimators=200, 
    bootstrap=True,     # use bootstrapping (sampling parts of the whole dataset for each tree)
    criterion='squared_error',
    random_state=None,
    n_jobs=-1,  # use all processors
    verbose=1,  # show progress
    )
model.fit(X_train_scaled, y_train_scaled)

# Make predictions
y_pred_train_scaled = model.predict(X_train_scaled)
y_pred_test_scaled = model.predict(X_test_scaled)

# inverse transform the scaled data
y_pred_train = scaler_y.inverse_transform(y_pred_train_scaled.reshape(-1, 1))
y_pred_test = scaler_y.inverse_transform(y_pred_test_scaled.reshape(-1, 1))

#%% Step 3: Plot the results
fns.plot_actual_vs_predicted_linear(y_test, y_pred_test, quantile=0.15, fraction=1)
fns.plot_actual_vs_predicted_linear(y_train, y_pred_train, quantile=None, fraction=1)

fns.plot_error_histogram(y_test, y_train, y_pred_test, y_pred_train)
#%% PLOT TIME SERIES (DEBUGGING)
x = data['time']        # or data.index
y1 = 'close'
y1_MA = None

y2 = 'log_return'
y2_MA = None
fns.plot_time_series(
    data,
    x, y1, y1_MA, y2, y2_MA,
    symbol, timeFrame, 
    artificial_returns, durations, means_generative)