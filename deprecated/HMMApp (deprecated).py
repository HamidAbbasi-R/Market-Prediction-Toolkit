#%%
from dash import Dash, dcc, html, Input, Output, State
import datetime
import numpy as np
import functions as fns
from datetime import datetime

app = Dash(__name__)

app.layout = html.Div([
    dcc.Input(id='symbol', type='text', placeholder='Symbol'),
    dcc.DatePickerSingle(id='endTime',
        date=datetime.now()
    ),
    dcc.Checklist(id='fromNow',
        options=[
            {'label': 'From Now', 'value': 'yes'}
        ],
        value=['yes']  # Set the checkbox as selected by default
    ),
    dcc.Dropdown(id='timeFrame',
        options=[
            {'label': '1 Minute', 'value': 'M1'},
            {'label': '5 Minutes', 'value': 'M5'},
            {'label': '15 Minutes', 'value': 'M15'},
            {'label': '30 Minutes', 'value': 'M30'},
            {'label': '1 Hour', 'value': 'H1'},
            {'label': '4 Hours', 'value': 'H4'},
            {'label': '1 Day', 'value': 'D1'},
            {'label': '1 Week', 'value': 'W1'},
            {'label': '1 Month', 'value': 'MN1'}
        ],
        placeholder="Select Time Frame",
        value='H1'  # Default value if needed
    ),
    dcc.Input(id='Nbars', type='number', placeholder='Nbars'),
    # define a slider for training fraction between 0 and 1
    dcc.Input(id='training_fraction', type='number', placeholder='Training fraction'),
    dcc.Input(id='hiddenStates', type='number', placeholder='Hidden States'),
    dcc.Input(id='n_iter', type='number', placeholder='N Iterations'),
    dcc.Checklist(id='train_return',
        options=[
            {'label': 'Train Return', 'value': 'yes'}
        ],
        value=[]
    ),
    dcc.Checklist(id='train_log_return',
        options=[
            {'label': 'Train Log Return', 'value': 'yes'}
        ],
        value=['yes']
    ),
    dcc.Checklist(id='train_log_volatility',
        options=[
            {'label': 'Train Log Volatility', 'value': 'yes'}
        ],
        value=[]
    ),
    dcc.Checklist(id='train_log_volume',
        options=[
            {'label': 'Train Log Volume', 'value': 'yes'}
        ],
        value=[]
    ),
    dcc.Checklist(id='show_candlesticks',
        options=[
            {'label': 'Show Candlesticks', 'value': 'yes'}
        ],
        value=[]
    ),
    dcc.Input(id='begin', type='number', placeholder='Begin'),
    dcc.Input(id='frac', type='number', placeholder='Fraction'),

    html.Button('Submit', id='submit-btn', n_clicks=0),

    html.Div([
        dcc.Graph(id='plot1', style={'width': '50%', 'display': 'inline-block'}),
        dcc.Graph(id='plot2', style={'width': '50%', 'display': 'inline-block'}),
        dcc.Graph(id='plot3', style={'width': '50%', 'display': 'inline-block'}),
        dcc.Graph(id='plot4', style={'width': '50%', 'display': 'inline-block'}),
        dcc.Graph(id='plot5', style={'width': '100%', 'display': 'inline-block'}),
    ])
])

@app.callback(
    [Output('plot1', 'figure'),
     Output('plot2', 'figure'),
     Output('plot3', 'figure'),
     Output('plot4', 'figure'),
     Output('plot5', 'figure')],
    [Input('submit-btn', 'n_clicks')],
    [State('symbol', 'value'),
     State('endTime', 'date'),
     State('fromNow', 'value'),
     State('timeFrame', 'value'),
     State('Nbars', 'value'),
     State('training_fraction', 'value'),
     State('hiddenStates', 'value'),
     State('n_iter', 'value'),
     State('train_return', 'value'),
     State('train_log_return', 'value'),
     State('train_log_volatility', 'value'),
     State('train_log_volume', 'value'),
     State('show_candlesticks', 'value'),
     State('begin', 'value'),
     State('frac', 'value')]
)

def update_graph(n_clicks_submit, symbol, end_time, from_now, time_frame, nbars, training_fraction, hidden_states,
                  n_iter, train_return, train_log_return, train_log_volatility, train_log_volume,
                  show_candlesticks, begin, frac):
    global data

    if n_clicks_submit > 0:
        if n_clicks_submit == 1:  # Only run the first time the submit button is clicked
            print('Getting data...', end=' ')
            if from_now:
                end_time = datetime.now()    # get data from Now to the past
            data = fns.GetPriceData(symbol, end_time, time_frame, nbars, source='MT5')
            print('Done')
        
        return fns.TrainHMM_hmmlearn(data, training_fraction, hidden_states, 
                         n_iter, train_return, train_log_return, train_log_volatility, 
                         train_log_volume, show_candlesticks, begin, frac)
    
    return {}, {}, {}, {}, {}

if __name__ == '__main__':
    app.run_server(
        # port = 8090, 
        # debug=True,
        )