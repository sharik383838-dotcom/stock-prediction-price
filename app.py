import os
import datetime
import traceback
from flask import Flask, render_template, render_template_string, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Determine absolute paths for reliable template and static loading across all environments
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR,
    static_folder=STATIC_DIR,
    static_url_path='/static'
)

# Feature columns used for training and prediction
FEATURE_COLS = [
    'Close',
    'Close_Lag1',
    'Close_Lag2',
    'Close_Lag3',
    'Close_Lag5',
    'SMA_5',
    'SMA_10',
    'SMA_20',
    'EMA_12',
    'Dist_SMA5',
    'Dist_SMA20',
    'Return_1d',
    'Return_5d',
    'HL_Spread',
    'Vol_SMA5'
]


def compute_next_trading_date(last_date):
    """
    Compute the approximate next trading day skipping standard weekend days.
    """
    if isinstance(last_date, (pd.Timestamp, datetime.datetime)):
        last_date = last_date.date()
    weekday = last_date.weekday()
    if weekday == 4:  # Friday -> Monday
        delta = datetime.timedelta(days=3)
    elif weekday == 5:  # Saturday -> Monday
        delta = datetime.timedelta(days=2)
    else:
        delta = datetime.timedelta(days=1)
    return last_date + delta


def engineer_features(df):
    """
    Engineer price and technical features to avoid data leakage while capturing
    momentum, moving averages, volatility, and volume indicators.
    """
    df = df.copy()

    # Lagged close prices
    df['Close_Lag1'] = df['Close'].shift(1)
    df['Close_Lag2'] = df['Close'].shift(2)
    df['Close_Lag3'] = df['Close'].shift(3)
    df['Close_Lag5'] = df['Close'].shift(5)

    # Moving Averages
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()

    # Distances from Moving Averages (normalized)
    df['Dist_SMA5'] = (df['Close'] - df['SMA_5']) / df['SMA_5']
    df['Dist_SMA20'] = (df['Close'] - df['SMA_20']) / df['SMA_20']

    # Returns / Momentum
    df['Return_1d'] = df['Close'].pct_change(1)
    df['Return_5d'] = df['Close'].pct_change(5)

    # Volatility / Spread
    df['HL_Spread'] = (df['High'] - df['Low']) / df['Close']

    # Volume features (handle zero or missing volume safely)
    volume_series = df['Volume'] if 'Volume' in df.columns else pd.Series(0, index=df.index)
    df['Vol_SMA5'] = volume_series.rolling(window=5).mean().fillna(0)

    # Target: The change in price for the next trading day
    df['Target_Change'] = df['Close'].shift(-1) - df['Close']
    df['Next_Close'] = df['Close'].shift(-1)

    return df


@app.route('/')
def index():
    """
    Render main single-page dashboard with serverless-safe fallback resolution.
    """
    try:
        return render_template('index.html')
    except Exception as template_err:
        # Fallback search paths in case execution context changes on serverless hosts
        candidate_paths = [
            os.path.join(TEMPLATE_DIR, 'index.html'),
            os.path.join(BASE_DIR, 'templates', 'index.html'),
            os.path.join(os.getcwd(), 'templates', 'index.html'),
            os.path.join(os.getcwd(), 'api', 'templates', 'index.html')
        ]
        for path in candidate_paths:
            if os.path.isfile(path):
                with open(path, 'r', encoding='utf-8') as f:
                    return render_template_string(f.read())
        
        # If all candidates fail, raise original error
        raise template_err


@app.route('/health')
def health():
    """Simple health check endpoint for monitoring & deployment validation."""
    return jsonify({
        'status': 'ok',
        'service': 'ai-stock-predictor',
        'template_dir': TEMPLATE_DIR,
        'template_exists': os.path.isfile(os.path.join(TEMPLATE_DIR, 'index.html'))
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(silent=True) or {}
        ticker_symbol = data.get('ticker', '').strip().upper()

        if not ticker_symbol:
            return jsonify({
                'success': False,
                'error': 'Please enter a valid stock ticker symbol (e.g. AAPL, TSLA, RELIANCE.NS).'
            }), 400

        # Fetch 2 years of daily data via yfinance
        try:
            ticker_obj = yf.Ticker(ticker_symbol)
            history = ticker_obj.history(period='2y', interval='1d')
        except Exception as fetch_err:
            return jsonify({
                'success': False,
                'error': f"Failed to connect to market data provider for '{ticker_symbol}': {str(fetch_err)}"
            }), 502

        if history is None or history.empty or 'Close' not in history.columns or len(history) < 30:
            return jsonify({
                'success': False,
                'error': f"No sufficient historical trading data found for '{ticker_symbol}'. "
                         f"Please verify the ticker symbol is correct, valid, and actively traded."
            }), 404

        # Extract currency with safe fallback
        currency = 'USD'
        try:
            currency = ticker_obj.fast_info.get('currency', 'USD') or 'USD'
        except Exception:
            currency = 'USD'

        # Feature engineering
        engineered_df = engineer_features(history)

        # The very last row has today's feature values, but its Target_Change is NaN (tomorrow hasn't occurred yet)
        latest_row = engineered_df.iloc[-1:]
        latest_features = latest_row[FEATURE_COLS]

        # Valid rows for training (where target is known)
        clean_df = engineered_df.dropna(subset=FEATURE_COLS + ['Target_Change', 'Next_Close'])

        if len(clean_df) < 20:
            return jsonify({
                'success': False,
                'error': f"Insufficient clean trading records ({len(clean_df)} valid sessions) to train model for '{ticker_symbol}'."
            }), 400

        # Chronological train/test split (80% train, 20% test) to prevent future data leakage
        split_idx = int(len(clean_df) * 0.8)
        train_df = clean_df.iloc[:split_idx]
        test_df = clean_df.iloc[split_idx:]

        X_train = train_df[FEATURE_COLS]
        y_train = train_df['Target_Change']

        X_test = test_df[FEATURE_COLS]
        test_actual_next_close = test_df['Next_Close']

        # Train Random Forest Regressor on historical training split (n_jobs=1 for serverless stability)
        eval_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=4,
            random_state=42,
            n_jobs=1
        )
        eval_model.fit(X_train, y_train)

        # Evaluate model performance on chronological out-of-sample test set
        pred_test_changes = eval_model.predict(X_test)
        pred_test_closes = test_df['Close'] + pred_test_changes
        mae = float(mean_absolute_error(test_actual_next_close, pred_test_closes))
        mean_test_price = float(test_actual_next_close.mean())
        mae_pct = float((mae / mean_test_price) * 100) if mean_test_price > 0 else 0.0

        # Retrain on full historical dataset to predict the upcoming next trading day
        full_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=4,
            random_state=42,
            n_jobs=1
        )
        full_model.fit(clean_df[FEATURE_COLS], clean_df['Target_Change'])

        # Predict next trading day's price change
        pred_change = float(full_model.predict(latest_features)[0])

        latest_close = float(engineered_df['Close'].iloc[-1])
        predicted_close = float(latest_close + pred_change)
        price_change = float(predicted_close - latest_close)
        price_change_pct = float((price_change / latest_close) * 100) if latest_close > 0 else 0.0

        # Determine bullish vs bearish trend
        trend = "Bullish" if price_change >= 0 else "Bearish"

        # Date calculations
        last_timestamp = engineered_df.index[-1]
        latest_date_str = last_timestamp.strftime('%Y-%m-%d')
        latest_date_display = last_timestamp.strftime('%b %d, %Y')

        next_trading_dt = compute_next_trading_date(last_timestamp)
        predicted_date_str = next_trading_dt.strftime('%Y-%m-%d')
        predicted_date_display = next_trading_dt.strftime('%b %d, %Y')

        # Chart historical data: last 60 trading days for clear, responsive visualization
        chart_window = engineered_df.tail(60)
        historical_chart_data = []
        for idx, row in chart_window.iterrows():
            historical_chart_data.append({
                'date': idx.strftime('%Y-%m-%d'),
                'displayDate': idx.strftime('%b %d'),
                'close': round(float(row['Close']), 2)
            })

        return jsonify({
            'success': True,
            'ticker': ticker_symbol,
            'currency': currency,
            'latest_close': round(latest_close, 2),
            'latest_date': latest_date_str,
            'latest_date_display': latest_date_display,
            'predicted_price': round(predicted_close, 2),
            'predicted_date': predicted_date_str,
            'predicted_date_display': predicted_date_display,
            'price_change': round(price_change, 2),
            'price_change_pct': round(price_change_pct, 2),
            'trend': trend,
            'mae': round(mae, 2),
            'mae_pct': round(mae_pct, 2),
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'historical_data': historical_chart_data
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f"Failed to predict for symbol: {str(e)}"
        }), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting AI Stock Predictor server on http://127.0.0.1:{port}")
    app.run(host='127.0.0.1', port=port, debug=True)
