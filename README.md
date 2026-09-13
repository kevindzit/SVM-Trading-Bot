# SVM Trading Bot

A machine learning trading bot that uses Support Vector Machine (SVM) models to predict cryptocurrency price movements and execute automated trades.


### 1. Data Collection (`Pull_Data/`)
- **bybit_data_downloader.py**: Downloads historical OHLCV data from Bybit API
- Supports multiple timeframes (15m, 1h, 4h, etc.)
- Automatically handles API rate limits and data pagination

### 2. Model Training (`SVM_Bot/`)
- **SVM_Model_Gen.py**: Creates and trains SVM models for price prediction
- Features technical indicators: EMA, MACD, RSI, Bollinger Bands, ATR, OBV
- Supports different SVM kernels (RBF, Linear, Polynomial, Sigmoid)
- Saves trained models and scalers for later use

### 3. Backtesting (`SVM_backtest/`)
- **Backtesting.py**: Tests strategy performance on historical data
- **Backtest_MoreTrades.py**: Extended backtesting with more trading opportunities
- Generates performance charts and statistics
- Compares strategy returns vs buy-and-hold

### 4. Utility Functions (`nice_funcs.py`)
- Close all positions on Bybit exchange
- Fetch wallet holdings and portfolio value
- Colored terminal output for better visibility

## Quick Start

### 1. Download Data
```python
# Edit Pull_Data/bybit_data_downloader.py
SYMBOL = "BTCUSDT"
INTERVAL_MINUTES = "60"
START_DATE_STR = "2024-01-01 00:00:00"
END_DATE_STR = "2025-01-01 00:00:00"

# Run the script
python Pull_Data/bybit_data_downloader.py
```

### 2. Train SVM Model
```python
# Edit SVM_Bot/SVM_Model_Gen.py
CSV_FILE_PATH = "path/to/your/data.csv"
SVM_KERNEL = 'rbf'  # or 'linear', 'poly', 'sigmoid'
SVM_C = 10
SVM_GAMMA = 0.01

# Run training
python SVM_Bot/SVM_Model_Gen.py
```

### 3. Backtest Strategy
```python
# Edit SVM_backtest/Backtesting.py
CSV_FILE = "path/to/your/data.csv"
INITIAL_CAPITAL = 10000
TRADE_VALUE_USD = 500

# Run backtest
python SVM_backtest/Backtesting.py
```

## Features

- **Multiple SVM Kernels**: RBF, Linear, Polynomial, Sigmoid
- **Technical Indicators**: 10+ indicators including EMA, MACD, RSI, Bollinger Bands
- **Automated Backtesting**: Performance analysis with charts and statistics
- **Risk Management**: Position sizing and commission handling
- **Exchange Integration**: Direct integration with Bybit API
- **Data Management**: Automated data collection and storage

## Key Files

- `SVM_Model_Gen.py`: Train new SVM models
- `Backtesting.py`: Test strategy performance
- `bybit_data_downloader.py`: Download market data
- `nice_funcs.py`: Trading utility functions

## Model Performance

The charts in `SVM_backtest/Backtest_Results/` are historical results. They have not
been regenerated since the Bollinger band width correction described below.
Rerun the corrected backtests before using those charts to report current results.

The system generates detailed backtest results including:
- Total return vs buy-and-hold
- Sharpe ratio and maximum drawdown
- Win rate and profit factor
- Visual equity curves and volume charts

## Feature Consistency

Training and both backtest scripts use the same ten features in the same order.
`BB_Width_20_2` is the upper Bollinger band minus the lower band, in price units,
using a 20 bar window and two standard deviations.

The backtests previously used `bollinger_wband()`, which divides that width by the
middle band and multiplies by 100. Passing this percentage into a scaler trained
on raw width changed the model input. Both backtests now use raw width to match
the training script. The training formula is unchanged.

Features are calculated before prices are scaled for simulated trade sizing.
Model and scaler path lookup now happens inside `main()`, so the feature functions
can be imported without local model files.

## Feature Tests

Use Python 3.12 and install the pinned test dependencies from the project root:

```bash
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -v
```

The tests use synthetic OHLCV data and call the actual training and backtest
feature functions. They check all ten feature values and their order, verify raw
Bollinger width against its formula, and check that trade price scaling leaves
model features unchanged. They also check that importing a backtest does not
search for local model files.

GitHub Actions runs these checks on pushes and pull requests. The tests do not
download market data, load saved models, train an SVM, or place trades. They check
feature consistency. Backtest performance still needs a separate run with market
data and a matching model and scaler.

## Requirements

- Python 3.12 for the tested setup
- pandas, numpy, scikit-learn
- ccxt (for exchange integration)
- ta (technical analysis library)
- backtesting.py library
- matplotlib (for charts)

## Configuration

Each script contains user-configurable parameters at the top:
- File paths for data and models
- SVM hyperparameters (C, gamma, kernel)
- Trading parameters (capital, position size)
- API credentials (for live trading)

## Note

This is a research and educational project. MIT License.
