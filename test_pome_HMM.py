#%%
from pomegranate.hmm import DenseHMM
from pomegranate.distributions import Normal
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import functions as fns

# Generate synthetic log return data (replace with actual log return data)
# data parameters
symbol='EURUSD'
fromNow = True
timeFrame = 'H1' 
Nbars = 10000
endTime = datetime(2024, 10, 30, 16, 0, 0)  # in case fromNow is False

# Artificial return values
artificial_returns = True
means = [-0.1, 0.1, 0]*1
covars = [0.1, 0.1, 0.1]*1
durations = [500,1500,1000]*1

if artificial_returns:
    data = fns.generate_artificial_returns(
        means,
        covars,
        durations,
        )
    log_returns = np.array(data['log_return'])
else:
    if fromNow:
        endTime = datetime.now()
    data = fns.GetPriceData(symbol, endTime, timeFrame, Nbars, source='MT5')
    log_returns = np.log(data['close']).diff().dropna().values



fig = go.Figure(data=go.Histogram(x=log_returns))
fig.update_layout(
    title="Log Return Distribution",
    xaxis_title="Log Returns",
    yaxis_title="Frequency",
)
fig.show()

#%%
# Define a variable to control the main diagonal of the transition matrix
main_diagonal_value = 0.8

# Define the transition matrix with the main diagonal controlled by the variable
edges = [
    [main_diagonal_value, (1 - main_diagonal_value) / 2, (1 - main_diagonal_value) / 2],
    [(1 - main_diagonal_value) / 2, main_diagonal_value, (1 - main_diagonal_value) / 2],
    [(1 - main_diagonal_value) / 2, (1 - main_diagonal_value) / 2, main_diagonal_value]
]
# Define each hidden state's emission distribution with mean and variance (not standard deviation)
neutral_dist = Normal(means=[0.1], covariance_type='full',)
bearish_dist = Normal(means=[0], covariance_type='full',)
bullish_dist = Normal(means=[-0.1], covariance_type='full',)

# Create the HMM with three hidden states and their respective distributions
model = DenseHMM(
    distributions=[bullish_dist, neutral_dist, bearish_dist],  # Emission distributions for each state
    edges=edges,
    # starts=[0.33, 0.33, 0.34],  # Initial probabilities for each state
)

# Train the model on log return data
# Reshape log_returns as a 2D array (n_samples, n_features) to fit pomegranate's input format
X = log_returns.reshape(1, -1, 1)
model.fit(X)

# Predict hidden states for each observation
hidden_states = model.predict(X)
hidden_states = np.array(hidden_states)
hidden_states = hidden_states.reshape(-1)
# print("Predicted hidden states:", hidden_states)

# create histograms for each hidden state
histograms = []
# reduce to 1d array
for i in range(3):
    state_data = log_returns[hidden_states == i]
    histogram = go.Histogram(x=state_data, name=f"State {i}", opacity=0.5)
    histograms.append(histogram)

# Create figure
fig = go.Figure(data=histograms)
fig.update_layout(
    title="Hidden State Distributions",
    xaxis_title="Log Returns",
    yaxis_title="Frequency",
    barmode="overlay",
)
fig.show()


#%%

from hmmlearn import hmm

np.random.seed(42)
model = hmm.GaussianHMM(n_components=3, covariance_type="full")
model.startprob_ = np.array([0.6, 0.3, 0.1])
model.transmat_ = np.array([[0.7, 0.2, 0.1],
                            [0.3, 0.5, 0.2],
                            [0.3, 0.3, 0.4]])
model.means_ = np.array([[0.0, 0.0], [3.0, -3.0], [5.0, 10.0]])
model.covars_ = np.tile(np.identity(2), (3, 1, 1))
X, Z = model.sample(10)

