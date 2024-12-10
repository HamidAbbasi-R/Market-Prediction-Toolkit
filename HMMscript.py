#%% IMPORTS
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.io import renderers
from datetime import datetime
import importlib
import functions as fns
# reload the functions module
importlib.reload(fns)

#%% USER INPUTS
print("Setting up parameters...", end="")
# data parameters
symbol='EURUSD'
fromNow = True
timeFrame = 'M30'
Nbars = 100000
endTime = datetime(2024, 10, 30, 16, 0, 0)  # in case fromNow is False
MA_length = 20
use_MA_for_log_return = False

# Artificial return values
artificial_returns = False
N_repeat = 8
means_generative = [0.005, -0.005, 0]*N_repeat
covars_generative = [0.002, 0.002, 0.002]*N_repeat
durations = [450,400,700]*N_repeat
# for means=0.005 and covars=0.002 'tied' and 'viterbi' or 'map' works well! 
# But this is trivial.

# HMM parameters
hiddenStates = 3
n_iter = 1000
covariance_type = 'tied'   # 'spherical', 'diag', 'full', 'tied'
algorithm = 'viterbi'  # 'viterbi', 'map'
verbose = True
training_fraction = 1
features_dict = {
    'return':               False,

    'log_return':           True,
    'MA_log_return':        False,
    'EMA_log_return':       False,

    'log_volatility':       True,
    'MA_log_volatility':    False,
    'EMA_log_volatility':   False,

    'log_volume':           False,
    'MA_log_volume':        False,
    'EMA_log_volume':       False,

    'log_ATR':              False,
}

# plot parameters
show_posteriors = False
show_candlesticks = False
begin_candles = 0
fraction_candles = 0.1

print("Done!")

#%% GET HISTORICAL DATA
print("Getting historcal data...", end="")
# calculate percentage change and volatility

if artificial_returns:
    data = fns.generate_artificial_returns(
        means_generative,
        covars_generative,
        durations,
        )
    features_dict = {
        'return':           False,
        'log_return':       True,
        'log_volatility':   False,
        'log_volume':       False,
        'log_ATR':          False,
    }
else:
    if fromNow:
        endTime = datetime.now()
    data = fns.GetPriceData(
        symbol, 
        endTime, 
        timeFrame, 
        Nbars, 
        indicators_dict={
            'ATR':     True,
            'ADX':     False,
            'RSI':     False,
            },    
        source='MT5')
    data['log_ATR'] = np.log(data['ATR'])

print("Done!")

#%% TRAIN HMM
print("Training model...", end="")
initial_means = [max(means_generative), 0, min(means_generative)] if artificial_returns else [np.nanquantile(data['log_return'], 0.9), 0, np.nanquantile(data['log_return'], 0.1)]
data, model = fns.TrainHMM_hmmlearn(
    data = data,
    training_fraction = training_fraction,
    hiddenStates=hiddenStates, 
    features_dict=features_dict,
    n_iter=n_iter,
    algorithm=algorithm,
    covariance_type=covariance_type,
    verbose=verbose,
    initial_means=initial_means,
    random_transmat=True,
    )

# get the means and the main diagonal of covars
means = model.means_
covars = model.covars_
covars = np.array([np.diag(covars[i]) for i in range(hiddenStates)])
print("Done!")

#%% CREATE HMM FIGURES
print('Creating figures...', end=' ')

features_index = {}
true_count = 0  # This will keep track of the index for True values
for key, value in features_dict.items():
    if value:
        features_index[key] = true_count
        true_count += 1
    else:
        features_index[key] = -1

# create figures
figs, fig_posterior_time_series = fns.CreateFiguresHMMs(
    data,
    training_fraction,
    features_index,
    means,
    covars,
    show_log_volatility=False if artificial_returns else True,
    show_log_volume=False if artificial_returns else True,
    show_log_ATR=False if artificial_returns else True,
    )

# add candlesticks 
if show_candlesticks:
    # blue=0, red=1, green=2
    fig_candles = fns.plot_candlesticks(
        data, 
        show_states=True, 
        begin=begin_candles,
        fraction=fraction_candles,
        )
    figs.append(fig_candles)


print('Done!')
#%% RENDER FIGURES
print('Rendering figures...', end=' ')

# Keep track of legend entries to avoid duplicates
existing_trace_names = set()

fig_subplts = make_subplots(
    rows=int(np.ceil(len(figs)/2)), cols=2,
    vertical_spacing=0.1,
    horizontal_spacing=0.1,
    )

for i, figure in enumerate(figs):
    if figure is None:
        continue
    row = int(i // 2 + 1)
    col = int(i % 2 + 1)
    fns.add_traces_with_legendgroup(fig_subplts, figure, row=row, col=col, existing_trace_names=existing_trace_names)

fig_subplts.update_layout(
    showlegend=True,
    title = f'HMM Analysis, Asset: {symbol}, Timeframe: {timeFrame}, Nbars: {len(data)}, Training fraction: {training_fraction}' + (' (Artificial data)' if artificial_returns else ''),
    # use the full length of the screen
    autosize=True,
    # height=1200,
    barmode='overlay',
    template='seaborn',
    legend=dict(
        title="Traces",  # Optional: Add a title for the legend
        x=1.1,  # Position the legend outside the main plot area for clarity
        y=1
    )
    )
# update x axis of subplots
renderers.default = 'browser'

# fig_subplts.show()
fig_subplts.write_html('figure.html', auto_open=True)

# posterior time series
if show_posteriors:
    features = [key for key, value in features_dict.items() if value]
    fig_sub_time_series = make_subplots(
        rows=len(features), cols=1,
        vertical_spacing=0.1,
        )

    for i, feature in enumerate(features):
        for trace in fig_posterior_time_series[feature].data:
            fig_sub_time_series.add_trace(trace, row=i+1, col=1)
        fig_sub_time_series.update_xaxes(title_text=fig_posterior_time_series[feature].layout.xaxis.title.text, row=i+1, col=1)
        fig_sub_time_series.update_yaxes(title_text=fig_posterior_time_series[feature].layout.yaxis.title.text, row=i+1, col=1)


    fig_sub_time_series.update_layout(
        showlegend=True,
        title = f'Posterior probabilities, Asset: {symbol}, Timeframe: {timeFrame}, Nbars: {len(data)}, Training fraction: {training_fraction}',
        # use the full length of the screen
        autosize=True,
        # height=1200,
        # barmode='overlay',
        # legend=dict(
        #     title="Traces",  # Optional: Add a title for the legend
        #     x=1.1,  # Position the legend outside the main plot area for clarity
        #     y=1
        # )
        )
    fig_sub_time_series.write_html('figure_posterior_time_series.html', auto_open=True)

print('Done!')
#%% PLOT HISTOGRAMS (DEBUGGING)
fig = go.Figure()
fig.add_trace(go.Histogram(
    x=data['log_volume'],
    name='volume',
    opacity=0.7,
    ))
# fig.add_vline(
#     x=0,
#     line=dict(
#         color='black',
#         width=1,
#     ),
# )
fig.show(renderer='vscode')

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