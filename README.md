# HMM for Financial Market Regime Detection

This Python project uses a Hidden Markov Model (HMM) to analyze financial market regimes based on asset price features such as return, volatility, and volume. The tool works with forex, stocks, and crypto data, leveraging the **MetaTrader 5 (MT5) Python API** to fetch data and the **hmmlearn** package to train the HMM.

---

## Features

1. **Flexible Dataframe Support**: 
   - Works with data ranging from **1-minute to 1-day timeframes**.
   - Compatible with forex, stocks, and cryptocurrency assets.

2. **Hidden State Analysis**:
   - Supports any number of hidden states, with **3 hidden states (bullish, bearish, neutral)** recommended for financial narratives.
   - Allows exploration of the relationship between hidden states and financial features.

3. **Feature Training Options**:
   - Train the HMM on any combination of the following features:
     - Return
     - Log Return
     - Log Volatility
     - Log Volume
     - Log ATR (Average True Range)
   - **Feature Interplay Analysis**: For example, compare volatility trends across states (e.g., low volatility often aligns with bullish regimes in stocks).

4. **Interactive Visualizations**:
   - **Histogram Plots**: View the distribution of features within each hidden state.
   - **Posterior Probability**: 
     - Plot the posterior probability of each state across feature values (e.g., log return).
     - Visualize the posterior probability for each data point (e.g., each candle).

---

## Installation

1. Do not needs installation at the moment. Just clone the repository and run the HMMscript.

---

## Usage

1. **Fetching Data**:
   - Fetch data using the MT5 Python API for your preferred asset and timeframe.
   - Example assets: EUR/USD (forex), BTC/USD (crypto), AAPL (stocks).

2. **Training the HMM**:
   - Train the model on selected features (e.g., log returns and log volatility).
   - Specify the number of hidden states (e.g., 3 for bullish, bearish, neutral).

3. **Visualizing Results**:
   - Use the provided Plotly-powered functions to generate interactive visualizations:
     - Feature histograms per state.
     - Posterior probabilities for states across feature values and data points.

---

## Example Workflow

1. Fetch hourly close prices for **BTC/USD**.
2. Train the HMM on **log returns** and **log volatility** with 3 hidden states.
3. Analyze the interplay between volatility and regimes:
   - Verify that bearish regimes align with high volatility.
   - Validate that bullish regimes correspond to low volatility.
4. Visualize the results:
   - Histograms of log returns per state.
   - Posterior probabilities over time.

---

## Insights

- **State Interpretations**:
  - Bullish: Typically associated with positive returns and low volatility.
  - Bearish: Usually correlates with negative returns and high volatility.
  - Neutral: Moderate returns with medium volatility.

- **Cautionary Note**: Misalignment of state characteristics (e.g., bearish with low volatility) may indicate data issues or unexpected market conditions.

---

## Future Enhancements

- Support for additional features (e.g., technical indicators).
- Optimization of state-transition probabilities.
- Incorporation of advanced HMM variants (e.g., Bayesian HMM).

---

## Acknowledgments

- **MT5 Python API** for seamless data integration.
- **hmmlearn** for HMM modeling.
- **Plotly** for interactive and insightful visualizations.

## Interactive Plotly Chart
Download and open [figure_artificial.html](figure_artificial.html) in your browser.