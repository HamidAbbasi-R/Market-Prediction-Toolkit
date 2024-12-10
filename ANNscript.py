#%% IMPORTS
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
from plotly.subplots import make_subplots
import functions as fns
# reload the functions module
import importlib
importlib.reload(fns)

def evaluate_model(scaler_y, X_test, y_test, model, criterion):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        test_loss = criterion(predictions, y_test)
        print(f"Test Loss: {test_loss.item()}")

    predictions = scaler_y.inverse_transform(predictions.numpy())
    y_test = scaler_y.inverse_transform(y_test.numpy())
    return y_test,predictions

def plot_prediction_actual_time_series(data, indices_test, predictions):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=data['close'], 
        mode="lines", 
        name="Actual", 
    ))
    fig.add_trace(go.Scatter(
        x=indices_test,
        y=predictions[:,0], 
        mode="markers", 
        name="Predicted", 
    ))
    fig.update_layout(
        title="Actual vs Predicted Price",
        xaxis_title="Time",
        yaxis_title="Price",
    )
    fig.show(renderer="vscode")

#%% Step 0: Set up parameters and get data
print("Setting up parameters...", end="")
# data parameters
symbol='EURUSD'        # 'EURUSD' or 'MSFT.US' or 'MSFT.US-24'
fromNow = True
timeFrame = 'D1'
Nbars = 10000
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
    'ATR':              True,
    'ADX':              False,
    'RSI':              False,
    
    # Macro features
    'US500':            False,
    'IYR.US':           False,
    'SHY.US':           False,
}

target = 'EMA_log_return'
forward_target = 0      # 0 means the same day, 1 means the next day, etc.
test_split = 0.05        # 0.7 means 70% test, 30% train
epochs = 500
N_neurons_1st_layer = 200
N_neurons_2nd_layer = 100


if artificial_returns:
    data = fns.generate_artificial_returns(
        means_generative,
        covars_generative,
        durations,
        )
else:
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
    
#%% Step 1: Load and preprocess data

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
y = data['target'].values
data.dropna(inplace=True)

# Scale data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y.reshape(-1, 1))

# Plot features correlation matrix
fns.plot_features_correlation_matrix(X, features_dict, lookback)
fns.plot_binary_features_relations(3, 4, X, features_dict, lookback)

# Train-test split
indices = np.arange(X.shape[0])

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
    X, y, indices, test_size=test_split, random_state=42)

# # Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

#%% Step 2: Define the model
class PricePredictor(nn.Module):
    def __init__(self, input_size):
        super(PricePredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, N_neurons_1st_layer)
        self.fc2 = nn.Linear(N_neurons_1st_layer, N_neurons_2nd_layer)
        self.fc3 = nn.Linear(N_neurons_2nd_layer, 1)
        self.relu = nn.ReLU()       # other activation functions: Sigmoid, Tanh
        # self.sigmoid = nn.Sigmoid()
        # self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = PricePredictor(input_size=X.shape[1])

#%% Step 3: Define the loss function and optimizer
criterion = nn.MSELoss()        # other loss functions: L1Loss, CrossEntropyLoss
# criterion = nn.L1Loss()
# criterion = nn.SmoothL1Loss()
# criterion = nn.CrossEntropyLoss()
# criterion = nn.BCELoss()
# criterion = nn.BCEWithLogitsLoss()
# criterion = nn.KLDivLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # alternative optimizer: SGD or RMSprop
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
# optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)

#%% Step 4: Train the model

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

#%% Step 5: Evaluate the model
y_test, predictions = evaluate_model(scaler_y, X_test, y_test, model, criterion)
y_train, predictions_train = evaluate_model(scaler_y, X_train, y_train, model, criterion)

#%% Step 6: Inverse transform predictions and actual values
# Display results
fns.plot_predictions_actual(y_test[:1000], predictions[:1000], target)
fns.plot_predictions_actual(y_train[:1000], predictions_train[:1000], target)

fns.plot_actual_vs_predicted_linear(y_test, predictions, 0.1)
fns.plot_actual_vs_predicted_linear(y_train, predictions_train, quantile=None, fraction=1)

fns.plot_error_histogram(y_test, y_train, predictions, predictions_train)

plot_prediction_actual_time_series(data, indices_test, predictions)

#%% Step 7: Evaluate the model on another dataset
# Get the data
# symbol='USDJPY'
# fromNow = False
# timeFrame = 'H1'
Nbars = 100000
# endTime = datetime(2024, 5, 30, 16, 0, 0)  # in case fromNow is False
endTime = data['time'].iloc[0]
dataNew = fns.GetPriceData(symbol, endTime, timeFrame, Nbars, source='MT5', MA_period=MA_period)

dataNew = fns.create_features_and_target(lookback, features_dict, dataNew, forward_target)

XNew = fns.prepare_feature_matrix(lookback, features_dict, dataNew)
yNew = dataNew[target].values

XNew = scaler_X.fit_transform(XNew)
yNew = scaler_y.fit_transform(yNew.reshape(-1, 1))

XNew = torch.tensor(XNew, dtype=torch.float32)
yNew = torch.tensor(yNew, dtype=torch.float32).view(-1, 1)

yNew, predictionsNew = evaluate_model(scaler_y, XNew, yNew, model, criterion)

# plot_error_histogram(yNew, predictionsNew)
# plot_prediction_actual_time_series(dataNew, np.arange(XNew.shape[0]), predictionsNew)
fns.plot_actual_vs_predicted_linear(yNew, predictionsNew)

#%% PLOT TIME SERIES (DEBUGGING)
x = data['time']        # or data.index
y1 = 'log_return'
y1_MA = None

y2 = 'close'
y2_MA = None

fns.plot_time_series(
    data,
    x, y1, y1_MA, y2, y2_MA,
    symbol, timeFrame, 
    artificial_returns, durations, means_generative)
