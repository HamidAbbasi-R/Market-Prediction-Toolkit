from tqdm import tqdm
# from pomegranate import HiddenMarkovModel, State, NormalDistribution
from datetime import datetime
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM, GMMHMM, MultinomialHMM
import pandas as pd
import pandas_ta as ta
from pandas.plotting import register_matplotlib_converters
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
# render plotly in browser
import plotly.io as pio
import MetaTrader5 as mt5
import warnings
register_matplotlib_converters()
pio.renderers.default = 'vscode'
mt5.initialize()
# disable all the warnings
warnings.filterwarnings('ignore')

def GetPriceData(
        symbol, 
        endTime = datetime.now(),
        timeframe = 'M5',
        Nbars = 1000,
        source = 'MT5',
        indicators_dict = {
            'ATR':      False,
            'ADX':      False,
            'RSI':      False,
        },
        MA_period = 20,
        ):
    
    if source=='MT5':
        # move the hour forward by 2 hours 
        endTime = endTime + pd.DateOffset(hours=2)

        # if Nbars is larger than 99999, get the data in chunks
        rates = pd.DataFrame()  # Initialize an empty DataFrame
        while Nbars > 0:
            Nbars_chunk = min(Nbars, 200000)
            Nbars -= Nbars_chunk

            rates_chunk = mt5.copy_rates_from(
                symbol, 
                ConvertTimeFrametoMT5(timeframe), 
                endTime,
                Nbars_chunk,
            )

            # convert to pandas DataFrame
            rates_chunk = pd.DataFrame(rates_chunk)

            # Add the retrieved chunk to the overall list
            rates = pd.concat([rates, rates_chunk], ignore_index=True)

            # Update endTime to the last time of the retrieved data
            endTime = rates_chunk['time'][0]  # Assuming the data is sorted in reverse chronological order
            
            # convert the endTime from int64 to datetime
            endTime = pd.to_datetime(endTime, unit='s')
            
        # convert times to UTC+1
        rates['time']=pd.to_datetime(rates['time'], unit='s')
        rates['time'] = rates['time'] + pd.DateOffset(hours=-2)

        rates['hour'] = rates['time'].dt.hour

        rates['MA_close'] = rates['close'].rolling(MA_period).mean()
        rates['EMA_close'] = rates['close'].ewm(span=MA_period, adjust=False).mean()

        # remove nans
        rates = rates.dropna()
        rates.rename(columns={'tick_volume': 'volume'}, inplace=True)
        rates['MA_volume'] = rates['volume'].rolling(MA_period).mean()
        rates['EMA_volume'] = rates['volume'].ewm(span=MA_period, adjust=False).mean()
        
        rates['log_volume'] = np.log(rates['volume'])
        rates['MA_log_volume'] = rates['log_volume'].rolling(MA_period).mean()
        rates['EMA_log_volume'] = rates['log_volume'].ewm(span=MA_period, adjust=False).mean()
        
        rates['log_return'] = np.log(rates['close'] / rates['close'].shift(1))
        rates['MA_log_return'] = rates['log_return'].rolling(MA_period).mean()       
        rates['EMA_log_return'] = rates['log_return'].ewm(span=MA_period, adjust=False).mean()
        
        rates['volatility'] = rates['log_return'].rolling(MA_period).std()
        rates['MA_volatility'] = rates['volatility'].rolling(MA_period).std()   
        rates['EMA_volatility'] = rates['volatility'].ewm(span=MA_period, adjust=False).std()
        
        rates['log_volatility'] = np.log(rates['volatility'])
        rates['MA_log_volatility'] = rates['log_volatility'].rolling(MA_period).mean()
        rates['EMA_log_volatility'] = rates['log_volatility'].ewm(span=MA_period, adjust=False).mean()
        
        rates['MA_volume'] = rates['volume'].rolling(MA_period).mean()
        rates['EMA_volume'] = rates['volume'].ewm(span=MA_period, adjust=False).mean()
        
        rates['upward'] = (rates['log_return'] > 0).astype(int)
            

        if indicators_dict['ATR']:
            rates['ATR'] = ta.atr(rates['high'], rates['low'], rates['close'], length=MA_period)
            
        if indicators_dict['ADX']:
            ADX = ta.adx(rates['high'], rates['low'], rates['close'], length=MA_period)
            rates['ADX'] = ADX[f'ADX_{MA_period}']

        if indicators_dict['RSI']:
            rates['RSI'] = ta.rsi(rates['close'], length=MA_period)
      
        return rates
    
    elif source=='yfinance':
        startTime = get_start_time(endTime, timeframe, Nbars)
        # convert the symbol to the format required by yfinance
        # AVAILABLE ASSETS
        # 'USDJPY=X' , 'USDCHF=X' , 'USDCAD=X', 
        # 'EURUSD=X' , 'GBPUSD=X' , 'AUDUSD=X' , 'NZDUSD=X', 
        # 'BTC-USD', 'ETH-USD', 'BNB-USD', 
        # 'XRP-USD', 'ADA-USD', 'SOL-USD', 'DOT-USD'
        if symbol[:3] in ['BTC', 'ETH', 'XRP', 'BNB', 'ADA', 'DOGE', 'DOT', 'SOL']:
            symbol = symbol[:3] + '-' + symbol[3:]
        else:
            symbol = symbol + '=X'
            # pass
        # convert timeframe to yfinance format
        timeframe = ConvertTimeFrametoYfinance(timeframe)
        rates = GetPriceData_Yfinance(symbol, startTime, endTime, timeframe)
        # change keys name from Close, Open, High, Low to close, open, high, low
        rates = rates.rename(columns={'Close':'close', 'Open':'open', 'High':'high', 'Low':'low'})
        # change keys name from Date to time
        rates['time'] = rates.index
        return rates

def ConvertTimeFrametoYfinance(timeframe):
    timeframes = {
        'M1': '1m',
        'M5': '5m',
        'M15': '15m',
        'M30': '30m',
        'H1': '1h',
        'H4': '4h',
        'D1': '1d',
        'W1': '1wk',
        'MN1': '1mo'
    }
    return timeframes.get(timeframe, 'Invalid timeframe')

def ConvertTimeFrametoMT5(timeframe):
    timeframes = {
        'M1': mt5.TIMEFRAME_M1,
        'M2': mt5.TIMEFRAME_M2,
        'M3': mt5.TIMEFRAME_M3,
        'M4': mt5.TIMEFRAME_M4,
        'M5': mt5.TIMEFRAME_M5,
        'M6': mt5.TIMEFRAME_M6,
        'M10': mt5.TIMEFRAME_M10,
        'M12': mt5.TIMEFRAME_M12,
        'M15': mt5.TIMEFRAME_M15,
        'M20': mt5.TIMEFRAME_M20,
        'M30': mt5.TIMEFRAME_M30,
        'H1': mt5.TIMEFRAME_H1,
        'H2': mt5.TIMEFRAME_H2,
        'H3': mt5.TIMEFRAME_H3,
        'H4': mt5.TIMEFRAME_H4,
        'H6': mt5.TIMEFRAME_H6,
        'H8': mt5.TIMEFRAME_H8,
        'H12': mt5.TIMEFRAME_H12,
        'D1': mt5.TIMEFRAME_D1,
        'W1': mt5.TIMEFRAME_W1,
        'MN1': mt5.TIMEFRAME_MN1
    }
    return timeframes.get(timeframe, 'Invalid timeframe')

def GetPriceData_Yfinance(
        symbol, 
        start_time, 
        end_time, 
        timeframe,
        ):
    import yfinance as yf
    OHLC = yf.Ticker(symbol).history(
                # [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo]
                interval=timeframe,
                # period=Duration,
                start = start_time,
                end = end_time,
            )
    return OHLC

def get_start_time(
        endTime, 
        timeframe, 
        Nbars,
        ):
    import re
    from datetime import timedelta
    def get_time_per_bar(timeframe):
    # Use regex to capture the numeric part and the unit
        match = re.match(r'([A-Za-z]+)(\d+)', timeframe)
        if not match:
            raise ValueError(f"Invalid timeframe format: {timeframe}")
    
        unit = match.group(1).upper()  # Get the letter part (M, H, D)
        value = int(match.group(2))    # Get the numeric part

        # Convert unit to appropriate timedelta
        if unit == 'M':  # Minutes
            return timedelta(minutes=value)
        elif unit == 'H':  # Hours
            return timedelta(hours=value)
        elif unit == 'D':  # Days
            return timedelta(days=value)
        else:
            raise ValueError(f"Unsupported timeframe unit: {unit}")

    # Get time per bar based on the timeframe
    time_per_bar = get_time_per_bar(timeframe)

    # Calculate total time to subtract
    total_time = time_per_bar * Nbars

    # Calculate the startTime
    startTime = endTime - total_time

    return startTime

def getPriceData_Crypto(symbol, endTime, timeframe, Nbars):
    import ccxt
    startTime = get_start_time(endTime, timeframe, Nbars)
    # convert timeframe to yfinance format
    timeframe = ConvertTimeFrametoYfinance(timeframe)
    phemex = ccxt.phemex({
        'enablerateLimit': True,
        'apiKey': '2dce697d-1b4a-412b-9d74-03c8c77d9cd3',
        'secret': 'ofeJqgVBH-xycduChQVqGOqnX6iwxV49BbQ-qyDzlGIyZDgwMGMyMy1iZjMzLTQ3NDEtYjJlYS01ZmM1ZTUyNTA0NjI',
    })
    bars = phemex.fetch_ohlcv(symbol, timeframe=timeframe, limit=Nbars)
    # convert to pandas DataFrame
    bars = pd.DataFrame(bars, columns=['time', 'open', 'high', 'low', 'close', 'tick-volume'])
    # convert times to UTC+1
    bars['time'] = pd.to_datetime(bars['time'], unit='ms')
    return bars

def plot_actual_vs_predicted_linear(y_test, predictions, quantile=None, fraction=1):
    # calculate the linear regression
    x = predictions[:,0]
    y = y_test[:,0]
    m, b = np.polyfit(x, y, 1)
    r2 = np.corrcoef(x, y)[0,1]**2
    
    # define a color gradient from two colors the same size as the data
    endindex = int(fraction * len(y_test))
    colors = np.linspace(0, 1, endindex)
    x = predictions[:endindex,0]
    y = y_test[:endindex,0]
    
    
    if quantile is not None:
        q1 = np.quantile(predictions[:endindex,0], quantile)
        q2 = np.quantile(predictions[:endindex,0], 1-quantile)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="markers",
        showlegend=False,
        marker=dict(
            # I want the color to gradually change
            color= colors,
            size=3,
        ),
    ))
    fig.update_xaxes(
        # same scale for x and y axes
        scaleanchor="y",
    )
    fig.add_trace(go.Scatter(
        x=[np.min(y_test[:endindex,0]), np.max(y_test[:endindex,0])],
        y=[np.min(y_test[:endindex,0]), np.max(y_test[:endindex,0])],
        mode="lines",
        line=dict(
            dash="dash",
            color="gray",
            width=1,
        ),
        showlegend=False
    ))
    fig.add_annotation(
        x=0.05,
        y=0.95,
        xref='paper',
        yref='paper',
        text=f"y = {m:.2f}x + {b:.2f}, R^2 = {r2:.2f}",
        showarrow=False,
        font=dict(
            color="blue",
        ),
    )
    fig.add_vline(
        x=0,
        line=dict(
            dash="dash",
            color="grey",
            width=1,
        ),
    )
    fig.add_hline(
        y=0,
        line=dict(
            dash="dash",
            color="grey",
            width=1,
        ),
    )
    fig.add_trace(go.Scatter(
        x=x,
        y=m*x + b,
        mode="lines",
        line=dict(
            color="blue",
            width=1,
        ),
        showlegend=False,
    ))
    
    if quantile is not None:
        fig.add_vrect(
            x0=-1,
            x1=q1,
            fillcolor="red",
            opacity=0.2,
            layer = "below",
            # no borders
            line_width=0,
        )
        fig.add_vrect(
            x0=q2,
            x1=1,
            fillcolor="green",
            opacity=0.2,
            layer = "below",
            line_width=0,
        )
    fig.update_layout(
        # title="Actual vs Predicted Price",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        # same width and height
        width=600,
        height=600,
        # limit axis range
        xaxis_range=[
            np.min(y_test[:,0]) - 0.1 * np.abs(np.min(y_test[:,0])), 
            np.max(y_test[:,0]) + 0.1 * np.abs(np.max(y_test[:,0]))
            ],
    )
    fig.show(renderer="vscode")

def create_features_and_target(
        lookback, 
        features_dict, 
        data, 
        forward_target,
        target,
        ):
    
    # lagged features
    for feature, use_feature in features_dict.items():
        for i in range(1, lookback + 1):
            if use_feature:
                data[f"{feature}_lag_{i}"] = data[feature].shift(i)

    data['target'] = data[target].shift(-forward_target).values
    data.dropna(inplace=True)

    return data

def prepare_feature_matrix(
        lookback, 
        features_dict, 
        data):

    # create empty feature matrix
    X = np.zeros((data.shape[0], 0))
    # use close price as a feature
    for feature, use_feature in features_dict.items():
        if use_feature:
            lagged_columns = [f"{feature}_lag_{i}" for i in range(1, lookback + 1)]
            X = np.concatenate([X, data[lagged_columns].values], axis=1)

    return X

def plot_time_series(
        data,
        x, y1, y1_MA, y2, y2_MA,
        symbol, timeFrame,
        artificial_returns=False, durations=None, means_generative=None):
    
    fig_subplots = make_subplots(
        rows=2 if y2 is not None else 1, 
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
    )

    # First subplot for log return
    fig_subplots.add_trace(go.Scatter(
        x=x,
        y=data[y1],
        mode='lines',
        connectgaps=False,
        line=dict(
            color='black',
            width=1,
        ),
    ), row=1, col=1)

    if y1_MA is not None:
        fig_subplots.add_trace(go.Scatter(
        x=x,
        y=data[y1_MA],
        mode='lines',
        connectgaps=False,
        line=dict(
            color='blue',
            width=1,
        ),
    ), row=1, col=1)

    # Second subplot for close price
    if y2 is not None:
        fig_subplots.add_trace(go.Scatter(
            x=x,
            y=data[y2],
            mode='lines',
            line=dict(
                color='black',
                width=1,
            ),
        ), row=2, col=1)

        if y2_MA is not None:
            fig_subplots.add_trace(go.Scatter(
                x=x,
                y=data[y2_MA],
                mode='lines',
                # do not fill gaps
                connectgaps=False,
                # name='Log Return'و
                line=dict(
                    color='blue',
                    width=1,
                ),
            ), row=2, col=1)

    # Update layout
    fig_subplots.update_layout(
        title = f'{symbol}, Timeframe={timeFrame}, Nbars={len(data)}',
        xaxis2_title='Time',
        yaxis_title=y1,
        yaxis2_title=y2 if y2 is not None else None,
        # height=600,
        showlegend=False,
    )

    if artificial_returns:
        # crate a rectangular region for each duration
        start_i = 0
        colors = []*len(durations)
        for i in range(len(durations)):
            if means_generative[i] > 0:
                colors.append('green')
            elif means_generative[i] < 0:
                colors.append('red')
            else:
                colors.append('blue')
        
            fig_subplots.add_vrect(
                x1=data['time'].iloc[start_i:start_i+durations[i]].iloc[-1],
                x0=data['time'].iloc[start_i:start_i+durations[i]].iloc[0],
                fillcolor=colors[i],
                opacity=0.2,
                layer='below',
                line_width=0,
                row=1, col=1
            )

            fig_subplots.add_vrect(
                x0=data['time'].iloc[start_i:start_i+durations[i]].iloc[0],
                x1=data['time'].iloc[start_i:start_i+durations[i]].iloc[-1],
                fillcolor=colors[i],
                opacity=0.2,
                layer='below',
                line_width=0,
                row=2, col=1
            )
            start_i += durations[i]

    fig_subplots.show(renderer='vscode')

def plot_predictions_actual(y_test, predictions, target):
    fig = go.Figure()
    for i in range(len(y_test)):
        fig.add_trace(go.Scatter(
        x=[i, i],
        y=[y_test[i,0], predictions[i,0]],
        mode="lines + markers",
        line=dict(
            color='black',
            width=1,
        ),
        marker=dict(
            color='red',
            size=3,
        ),
        showlegend=False,
        ))
    
    if target == 'log_return' or target == 'MA_log_return' or target == 'EMA_log_return':
        fig.add_hline(
            y=0,
            line=dict(
                # use dash
                dash='dash',
                color='gray',
                width=1,
            ),
        )
    fig.update_layout(
        xaxis_title="Index",
        yaxis_title="Value",
    )
    fig.show(renderer="vscode")

def plot_error_histogram(y_test, y_train, predictions, predictions_train):
    fig = go.Figure()
    hist_test  = y_test[:,0]  - predictions[:,0]
    hist_train = y_train[:,0] - predictions_train[:,0]
    fig.add_trace(go.Histogram(
        x=hist_test, 
        name="Test",
        opacity=0.5,
        histnorm='probability',
    ))
    fig.add_trace(go.Histogram(
        x=hist_train,
        name="Train",
        opacity=0.5,
        histnorm='probability',
    ))
    fig.add_vline(
        x=0,
        line=dict(
            color="black",
            width=1,
        ),
    )
    fig.update_layout(
        # title="Error Distribution",
        xaxis_title="Error",
        yaxis_title="Probability",
        # overlay histograms
        barmode="overlay",
    )
    # save the figure as an jpg file
    fig.write_image("error_histogram.jpg")
    fig.show(renderer="vscode")

def plot_binary_features_relations(f1,f2, X, features_dict, lookback):
    x_labels = list([f"{feature}_lag_{i}" for feature, use_feature in features_dict.items() if use_feature for i in range(1, lookback + 1)])
    x = X[:,f1]
    y = X[:,f2]
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="markers",
        marker=dict(
            color="blue",
            size=3,
            opacity=0.5,
        ),
        showlegend=False,
    ))
    fig.update_layout(
        xaxis_title=x_labels[f1],
        yaxis_title=x_labels[f2],
        xaxis=dict(scaleanchor="y"),
    )
    fig.add_trace(go.Scatter(
        x=[xmin,xmin,xmax,xmax,xmin],
        y=[ymin,ymax,ymax,ymin,ymin],
        mode="lines",
        line=dict(
            color="black",
            width=1,
        ),
        showlegend=False,
    ))

    fig.show(renderer="vscode")

def plot_features_correlation_matrix(X, features_dict, lookback):
    # calculate the correlation matrix of the features
    corr_matrix = np.corrcoef(X.T)
    # replace the main diagonal with NaN
    np.fill_diagonal(corr_matrix, np.nan)
    x_labels = list([f"{feature}_lag_{i}" for feature, use_feature in features_dict.items() if use_feature for i in range(1, lookback + 1)])
    y_labels = x_labels
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=x_labels,
        y=y_labels,
        colorscale='Viridis',
    ))
    # y scale anchor is x
    fig.update_layout(
        xaxis=dict(scaleanchor="y"),
    )
    fig.show(renderer="vscode")

def TrainHMM_hmmlearn(
    data,
    training_fraction,
    hiddenStates,
    features_dict,
    n_iter = 1000,
    algorithm = 'viterbi',
    covariance_type = 'full',
    verbose = False,
    initial_means = None,
    random_transmat = False,
    ):

    # Drop NaN values (first row)
    data = data.dropna()

    # Prepare data for HMM (reshape to 2D)
    features = []
    for feature, use_feature in features_dict.items():
        if use_feature:
            features.append(np.array(data[feature]))
    
    X = np.column_stack(features) if features else np.empty((len(data), 0))

    # Initialize and train a Gaussian HMM with 3 hidden states [the options instead of GaussianHMM are MultinomialHMM, GMMHMM]
    # means_prior = np.array([[-1.0], [0.0], [1.0]])
    model = GaussianHMM(
        n_components=hiddenStates, 
        covariance_type=covariance_type, 
        n_iter=n_iter,
        algorithm=algorithm,
        verbose=verbose,
        tol = 1e-7,
        init_params="stcm",
        implementation='scaling',
        )

    # model.startprob_ = np.full(hiddenStates, 1/hiddenStates)

    if random_transmat:
        # an array of random numbers between 0.1, 0.9
        main_diag = np.random.uniform(0.3, 0.95, hiddenStates)
    else:
        main_diag = np.full(hiddenStates, 0.7)
    model.transmat_ = np.array([
        [main_diag[0], (1 - main_diag[0]) / 2, (1 - main_diag[0]) / 2],
        [(1 - main_diag[1]) / 2, main_diag[1], (1 - main_diag[1]) / 2],
        [(1 - main_diag[2]) / 2, (1 - main_diag[2]) / 2, main_diag[2]]
    ])

    if initial_means is not None:
        model.means_ = np.array(initial_means).reshape(-1, 1)

    # model.covars_ = np.array([[[0.04]],[[0.04]],[[0.04]]])
    # model = MultinomialHMM(n_components=hiddenStates, n_iter=n_iter, tol=1e-4)
    # model = GMMHMM(n_components=hiddenStates, n_mix=3, n_iter=n_iter, tol=1e-4)

    # model.means_ = np.array([[-0.1], [0], [0.1]])

    # Fit the model on the first half of the data
    model.fit(X[:int(len(X) * training_fraction)])

    # Ensure no row of transmat_ has zero sum
    # model.transmat_ += 1e-6
    # model.transmat_ /= model.transmat_.sum(axis=1)[:, np.newaxis]

    # Predict hidden states for each candle
    hidden_states = model.predict(X)

    # Add hidden states back to the original data (aligning with index)
    data['hidden_state'] = hidden_states

    return data, model

def generate_artificial_returns(means, covs, durations):
    """
    Generates artificial return values for an imaginary financial asset 
    given periods of bullish, bearish, and neutral regimes.

    Parameters:
    - means: 1D array of means for the Gaussian distribution of return values in each period.
             Positive for bullish, negative for bearish, and zero for neutral.
    - covs: 1D array of covariances (variances) for the Gaussian distribution in each period.
    - durations: 1D array of durations for each period.

    Returns:
    - returns: 1D array of return values.
    """
    if not (len(means) == len(covs) == len(durations)):
        raise ValueError("means, covs, and durations must have the same length.")

    log_returns = []
    for mean, cov, duration in zip(means, covs, durations):
        period_returns = np.random.normal(loc=mean, scale=cov, size=duration)
        log_returns.extend(period_returns)

    # close = np.exp(np.cumsum(log_returns))
    close = np.zeros(len(log_returns)+1)
    close[0] = 100
    for i in range(len(log_returns)):
        close[i+1] = close[i] * np.exp(log_returns[i])
    # close[1:] = close[:-1] * np.exp(log_returns)

    # add a nan element to the beginning of the log_returns
    log_returns = np.insert(log_returns, 0, np.nan)
    
    data = pd.DataFrame({
        'time': pd.date_range(start='2022-01-01', periods=len(log_returns), freq='H'),      # arbitrary start date and hourly frequency
        'log_return': log_returns,
        'close': close,
    })
    # drop nan values
    data = data.dropna()
    return data

def CreateFiguresHMMs(
        data, 
        training_fraction,
        features_index,
        means=None,
        covars=None,
        show_log_returns=True, 
        show_log_volatility=True,
        show_log_volume=True,
        show_log_ATR=True,
        show_share_hidden_states=True,
        show_theoretical_gaussian=True,
        show_posterior_time_series=True,
        ):

    def process_posterior_figs(fig, x, title):
        # posterior probability
        xmin = xmax = 0
        for state in np.arange(Nstates):
            xmin = min(xmin, np.min(x[start:end][hidden_states[start:end] == state]))
            xmax = max(xmax, np.max(x[start:end][hidden_states[start:end] == state]))
        x = np.linspace(xmin, xmax, 5000)
        
        posterior_P = posterior_probability(
            x, 
            means[:, features_index[title]],
            covars[:, features_index[title]],
            )
        
        for state in np.arange(Nstates):
            fig.add_trace(go.Scatter(
                x=x,
                y=posterior_P[state],
                mode='lines',
                name=f'Posterior {state}',
                line=dict(
                    color=colors[state],
                    width=2,
                ),
                showlegend=True,
            ))
        title = title.replace('_', ' ').title()
        
        fig.update_layout(
            xaxis_title=title,
            yaxis_title='Probability',
        )

    if training_fraction == 1: 
        flag_test = False
    else:
        flag_test = True
    
    # get the features
    hidden_states = data['hidden_state']
    Nstates = len(np.unique(hidden_states))
    if means is None: show_theoretical_gaussian = False

    # get the colors for each hidden state
    colors, _ = get_colormap_colors('Set1', len(np.unique(hidden_states)))

    # initialize the figures
    fig_log_returns = go.Figure()
    fig_log_volatility = go.Figure()
    fig_log_volume = go.Figure()
    fig_log_ATR = go.Figure()
    fig_share = go.Figure()
    fig_posterior_log_return = go.Figure()
    fig_posterior_log_volatility = go.Figure()
    fig_posterior_log_volume = go.Figure()
    fig_posterior_log_ATR = go.Figure()
    features = ['log_return', 'log_volatility', 'log_volume', 'log_ATR']
    fig_posterior_time_series = {feature: go.Figure() for feature in features}


    # create a figure for each feature
    cases = ['train', 'test'] if flag_test else ['train']
    for case in cases:
        if case == 'train':
            start = 0
            end = int(len(data) * training_fraction)
        elif case == 'test':
            start = int(len(data) * training_fraction)
            end = len(data)

        if show_log_returns:
            log_returns = data['log_return']
            PDF_log_return_states = [[] for _ in range(Nstates)]
            for state in np.arange(Nstates):
                fig_log_returns.add_trace(go.Histogram(
                    x=log_returns[start:end][hidden_states[start:end] == state],
                    name=f'State {state} - {case}',
                    histnorm = 'probability density',   # options are 'probability', 'percent', 'density', 'probability density'
                    marker_color=colors[state],
                    opacity=0.5 if not flag_test else 0.3 if case == 'train' else 0.7,
                ))

                # theoretical gaussian distribution
                if show_theoretical_gaussian and case == 'train' and features_index['log_return']!=-1:
                    x = np.linspace(np.min(log_returns[start:end][hidden_states[start:end] == state]), np.max(log_returns[start:end][hidden_states[start:end] == state]), 1000)
                    PDF_log_return_states[state] = Gaussian_distribution(x, means[state,features_index['log_return']], covars[state,features_index['log_return']])
                    # scale the gaussian to the maximum value of histogram
                    # y = y[0] * max_val / np.max(y[0])
                    fig_log_returns.add_trace(go.Scatter(
                        x=x,
                        y=PDF_log_return_states[state],
                        mode='lines',
                        name=f'State {state} Gaussian',
                        line=dict(
                            color=colors[state],
                            width=2,
                        ),
                        showlegend=True,
                    ))

                # show the maximum value of the histogram as a vline
                fig_log_returns.add_vline(
                    x=np.mean(log_returns[start:end][hidden_states[start:end] == state]),
                    line=dict(
                        color=colors[state],
                        width=1,
                        dash='dash' if case == 'train' else 'solid',
                    ),
                )

            fig_log_returns.update_layout(
                xaxis_title='Log Return (%)',
                yaxis_title='Probability',
                barmode='overlay',
            )
            # show the x=0 line
            fig_log_returns.add_vline(
                x=0,
                line=dict(
                    color='black',
                    width=1,
                ),
            )

            # posterior probability
            if show_theoretical_gaussian and case == 'train' and features_index['log_return']!=-1:
                process_posterior_figs(fig_posterior_log_return, log_returns, 'log_return')
       
        if show_log_volatility:
            log_volatility = data['log_volatility']
            PDF_log_volatility_states = [[] for _ in range(Nstates)]
            for state in np.arange(Nstates):
                fig_log_volatility.add_trace(go.Histogram(
                    x=log_volatility[start:end][hidden_states[start:end] == state],
                    name=f'State {state} - {case}',
                    histnorm='probability density',
                    marker_color=colors[state],
                    opacity=0.5 if not flag_test else 0.3 if case == 'train' else 0.7,
                ))

                # theoretical gaussian distribution (NEEDS MORE WORK)
                if show_theoretical_gaussian and case == 'train' and features_index['log_volatility']!=-1:
                    x = np.linspace(
                        np.min(log_volatility[start:end][hidden_states[start:end] == state]), 
                        np.max(log_volatility[start:end][hidden_states[start:end] == state]), 
                        1000)
                    PDF_log_volatility_states[state] = Gaussian_distribution(x, means[state,features_index['log_volatility']], covars[state,features_index['log_volatility']])

                    fig_log_volatility.add_trace(go.Scatter(
                        x=x,
                        y=PDF_log_volatility_states[state],
                        mode='lines',
                        name=f'State {state} Gaussian',
                        line=dict(
                            color=colors[state],
                            width=2,
                        ),
                        showlegend=True,
                    ))

            fig_log_volatility.update_layout(
                # title='Distribution of volatility for Each Hidden State',
                xaxis_title='Log Volatility',
                yaxis_title='Probability',
                barmode='overlay',
            )

            # posterior probability
            if show_theoretical_gaussian and case == 'train' and features_index['log_volatility']!=-1:
                process_posterior_figs(fig_posterior_log_volatility, log_volatility, 'log_volatility')
            
        if show_log_volume:
            log_volume = data['log_volume']
            PDF_log_volume_states = [[] for _ in range(Nstates)]
            for state in np.arange(Nstates):
                fig_log_volume.add_trace(go.Histogram(
                    x=log_volume[start:end][hidden_states[start:end] == state],
                    name=f'State {state} - {case}',
                    histnorm='probability density',
                    marker_color=colors[state],
                    opacity=0.5 if not flag_test else 0.3 if case == 'train' else 0.7,
                ))
                
                # theoretical gaussian distribution (NEEDS MORE WORK)
                if show_theoretical_gaussian and case == 'train' and features_index['log_volume']!=-1:
                    x = np.linspace(
                        np.min(log_volume[start:end][hidden_states[start:end] == state]), 
                        np.max(log_volume[start:end][hidden_states[start:end] == state]), 
                        1000)
                    PDF_log_volume_states[state] = Gaussian_distribution(x, means[state,features_index['log_volume']], covars[state,features_index['log_volume']])
                    fig_log_volume.add_trace(go.Scatter(
                        x=x,
                        y=PDF_log_volume_states[state],
                        mode='lines',
                        name=f'State {state} Gaussian',
                        line=dict(
                            color=colors[state],
                            width=2,
                        ),
                        showlegend=True,
                    ))

            fig_log_volume.update_layout(
                # title='Distribution of Volume for Each Hidden State',
                xaxis_title='Log Volume',
                yaxis_title='Probability',
                barmode='overlay',
            )

            # posterior probability
            if show_theoretical_gaussian and case == 'train' and features_index['log_volume']!=-1:
                process_posterior_figs(fig_posterior_log_volume, log_volume, 'log_volume')

        if show_log_ATR:
            log_ATR = data['log_ATR']
            PDF_log_ATR_states = [[] for _ in range(Nstates)]
            for state in np.arange(Nstates):
                fig_log_ATR.add_trace(go.Histogram(
                    x=log_ATR[start:end][hidden_states[start:end] == state],
                    name=f'State {state} - {case}',
                    histnorm='probability density',
                    marker_color=colors[state],
                    opacity=0.5 if not flag_test else 0.3 if case == 'train' else 0.7,
                ))

                # theoretical gaussian distribution (NEEDS MORE WORK)
                if show_theoretical_gaussian and case == 'train' and features_index['log_ATR']!=-1:
                    x = np.linspace(
                        np.min(log_ATR[start:end][hidden_states[start:end] == state]), 
                        np.max(log_ATR[start:end][hidden_states[start:end] == state]), 
                        1000)
                    PDF_log_ATR_states[state] = Gaussian_distribution(x, means[state,features_index['log_ATR']], covars[state,features_index['log_ATR']])
                    fig_log_ATR.add_trace(go.Scatter(
                        x=x,
                        y=PDF_log_ATR_states[state],
                        mode='lines',
                        name=f'State {state} Gaussian',
                        line=dict(
                            color=colors[state],
                            width=2,
                        ),
                        showlegend=True,
                    ))

            fig_log_ATR.update_layout(
                # title='Distribution of ATR for Each Hidden State',
                xaxis_title='Log ATR',
                yaxis_title='Probability',
                barmode='overlay',
            )

            # posterior probability
            if show_theoretical_gaussian and case == 'train' and features_index['log_ATR']!=-1:
                process_posterior_figs(fig_posterior_log_ATR, log_ATR, 'log_ATR')

        if show_share_hidden_states:
            # plot the share of each hidden state in the dataset
            state_counts = {}
            for state in np.arange(Nstates):
                state_counts[state] = (data[start:end]['hidden_state'] == state).sum() / len(data[start:end]['hidden_state']) * 100
            state_counts = pd.Series(state_counts)
            
            fig_share.add_trace(go.Bar(
                x=state_counts.index,
                y=state_counts.values,
                name=f'State Share (%) - {case}', 
                marker_color=colors,
                opacity=0.5 if not flag_test else 0.3 if case == 'train' else 0.7,
            ))
            fig_share.update_layout(
                # title='Share of Each Hidden State in the Dataset',
                xaxis_title='State',
                yaxis_title='Share',
            )
            # x axis is categorical
            fig_share.update_xaxes(type='category')

    if show_posterior_time_series:
        data = PopulateMissingTimesWithNans(data)
        features = ['log_return', 'log_volatility', 'log_volume', 'log_ATR']
        for feature in features:
            if features_index[feature] != -1:
                posterior_probs = posterior_probability(
                    data[feature],
                    means[:, features_index[feature]],
                    covars[:, features_index[feature]],
                    )
                
                for state in range(Nstates): 
                    data[f'posterior_{feature}_state_{state}'] = posterior_probs[state]

        for feature in features:
            if features_index[feature] != -1:
                y0 = np.zeros(len(data))
                for state in range(Nstates):
                    fig_posterior_time_series[feature].add_trace(go.Scatter(
                        x=data['time'],
                        # x=np.arange(len(data)),
                        y=data[f'posterior_{feature}_state_{state}'] + y0,
                        mode='lines',
                        name=f'Posterior {feature} State {state}',
                        line=dict(
                            color=colors[state],
                            width=0,
                        ),
                        fill = 'tonexty', # option is 'tozeroy', 'tozerox', 'tonexty', 'tonextx'
                        showlegend=False,
                        # do not show hover
                        hoverinfo='skip',
                        # gaps are not shown
                        # connectgaps=True,
                    ))
                    y0 += data[f'posterior_{feature}_state_{state}']

                fig_posterior_time_series[feature].update_layout(
                    xaxis_title='Time',
                    yaxis_title=f'Probability {feature}',
                    # xaxis={'type':'category'}
                )

    figs = [
        fig_log_returns, 
        fig_log_volatility, 
        fig_log_volume, 
        fig_log_ATR, 
        fig_share, 
        fig_posterior_log_return, 
        fig_posterior_log_volatility, 
        fig_posterior_log_volume, 
        fig_posterior_log_ATR,
        ]
    
    # figs.extend(list(fig_posterior_time_series.values()))

    # remove any empty figures
    figs = [fig for fig in figs if fig.data]
    return figs, fig_posterior_time_series

def Gaussian_distribution(x, mean, covar):
    # return norm.pdf(x, loc=mean, scale=np.sqrt(covar))
    return np.exp(-0.5 * (x - mean) ** 2 / covar) / np.sqrt(2 * np.pi * covar)

def PopulateMissingTimesWithNans(data):
    # calculate the time difference between the first two dates
    time_diff = data['time'].diff().iloc[1]

    # calculate the start and end time 
    start_time = data['time'].iloc[0]
    end_time = data['time'].iloc[-1]

    # Create a continuous time index covering every hour within the date range of the original DataFrame
    all_range = pd.date_range(start=start_time, end=end_time, freq=time_diff)

    # Reindex the original DataFrame with the new time index and keep the original data
    data = data.set_index('time').reindex(all_range).reset_index()

    # rename index to time
    data = data.rename(columns={'index':'time'})
    return data

def posterior_probability(x, means, covars, priors=None):
    # assume equal priors if not provided
    if priors is None:
        priors = [1/len(means)] * len(means)

    PDF_states =[np.exp(-0.5 * (x - mean) ** 2 / covar) / np.sqrt(2 * np.pi * covar) for mean, covar in zip(means, covars)]
    
    # Calculate the evidence P(x)
    P_x = sum(PDF_state * prior for PDF_state, prior in zip(PDF_states, priors))
    
    # Calculate P(A | x) and P(B | x) and P(C | x) using Bayes' rule. it's called posterior probability
    posterior_P = [(PDF_state * prior) / P_x for PDF_state, prior in zip(PDF_states, priors)]
    
    return posterior_P

def get_colormap_colors(cmap_name, N):
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap(cmap_name)
    # if colormap is qualitative, use the first N colors
    # if colormap is quantitative, use N evenly spaced colors
    if cmap.N == 256:
        colors_array = [cmap(i / N) for i in range(N)]  # Get RGBA colors for N evenly spaced values
    else:
        colors_array = [cmap(i) for i in range(N)]  # Get the first N colors
    # convert to rgba format
    colors = [f'rgba({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)},1)' for color in colors_array]
    # colors opacified
    colors_opac = [f'rgba({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)},0.5)' for color in colors_array]
    return colors, colors_opac

def add_traces_with_legendgroup(subplot, figure, row, col, existing_trace_names):
    for trace in figure.data:
        # If the trace name is already in the legend, don't show the legend for this trace
        if trace.name in existing_trace_names:
            trace.showlegend = False
        else:
            existing_trace_names.add(trace.name)  # Track unique trace names
            trace.showlegend = True  # Ensure legend entry is displayed
        trace.legendgroup = trace.name  # Set legendgroup to trace name for unified control
        subplot.add_trace(trace, row=row, col=col)
    # Set axis titles for each subplot
    subplot.update_xaxes(title_text=figure.layout.xaxis.title.text, row=row, col=col)
    subplot.update_yaxes(title_text=figure.layout.yaxis.title.text, row=row, col=col)
    # show the legend
    subplot.update_layout(showlegend=True)

def plot_candlesticks(
        symbol1, 
        symbol2=None,
        titles=['Pair1', 'Pair2'],
        show_states=False,
        begin=0,        # from 0 to 1
        fraction=1,     # from 0 to 1
        ):
    
    # if OHL data is not provided, use Scatter plot instead of Candlestick
    if symbol1['open'].isnull().all():
        flagScatter = True
    else:
        flagScatter = False

    # cut the data to a fraction from the "begin" point
    symbol1 = symbol1.iloc[int(begin*len(symbol1)):int(begin*len(symbol1))+int(fraction*len(symbol1))]
    if symbol2 is not None:
        symbol2 = symbol2.iloc[int(begin*len(symbol2)):int(begin*len(symbol2))+int(fraction*len(symbol2))]

    # if pair2 is not None, create a subplot with linked x-axis
    # use plotly
    fig = make_subplots(
        rows=3 if symbol2 is not None else 1, 
        cols=1)
    fig.add_trace(go.Candlestick(
        x    = symbol1['time'],
        open = symbol1['open'],
        high = symbol1['high'],
        low  = symbol1['low'],
        close= symbol1['close'],
        name = titles[0],
        hoverinfo=None,
    ), row=1, col=1)
    if symbol2 is not None:
        fig.add_trace(go.Candlestick(
            x     = symbol2['time'],
            open  = symbol2['open'],
            high  = symbol2['high'],
            low   = symbol2['low'],
            close = symbol2['close'],
            name  = titles[1],
            hoverinfo=None,
        ), row=2, col=1)
        fig.update_layout(
            xaxis2_title='Time',
            yaxis2_title=titles[1],
        )

    fig.update_layout(
        xaxis_title='Time',
        yaxis_title=titles[0],
    )
    fig.update_traces(
        increasing_line_color='rgb(8,153,129)',
        decreasing_line_color='rgb(242,54,69)',
        increasing_fillcolor='rgb(8,153,129)',
        decreasing_fillcolor='rgb(242,54,69)',
    )
    fig.update_xaxes(
        # type='category',
        rangeslider_visible=False,
        matches='x',
        )
    
    # add states to the chart
    if show_states:
        states = np.array(symbol1['hidden_state'])
        # use alternating colors the same size as len(np.unique(states))
        colors = ['blue', 'red', 'green']
        for i in tqdm(range(len(states))):
            if i == 0:
                start = symbol1['time'].iloc[i]
            elif states[i] != states[i-1]:
                end = symbol1['time'].iloc[i]
                fig.add_vrect(
                    x0=start, 
                    x1=end, 
                    fillcolor=colors[states[i]], 
                    opacity=0.2, 
                    layer='below', 
                    line_width=0,
                    )
                start = symbol1['time'].iloc[i]
    
    return fig
