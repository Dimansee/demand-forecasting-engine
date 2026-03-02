import xgboost as xgb
import pandas as pd
from utils.feature_engineering import create_features

def xgb_forecast(df, prophet_forecast_df, steps=30):
    # Prepare Training Data
    train_df = create_features(df.copy(), is_future=False)
    
    # Align Prophet trend
    prophet_trend = prophet_forecast_df.set_index('ds')['trend']
    train_df['prophet_trend'] = train_df['Date'].map(prophet_trend)
    train_df['residual'] = train_df['Sales'] - train_df['prophet_trend']

    # MUST match the features created in feature_engineering.py
    feature_cols = ['month', 'dayofweek', 'dayofmonth', 'is_winter', 'is_weekend', 'is_payday']
    
    X = train_df[feature_cols]
    y = train_df['residual']

    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3)
    model.fit(X, y)

    # Prepare Future Data
    last_date = df['Date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps)
    future_df = pd.DataFrame({'Date': future_dates})
    future_df = create_features(future_df, is_future=True)
    
    X_future = future_df[feature_cols]
    residual_forecast = model.predict(X_future)

    return residual_forecast