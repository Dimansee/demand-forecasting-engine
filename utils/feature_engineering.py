import pandas as pd

def create_features(df, is_future=False):
    # Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 1. Create the base temporal columns first
    df['month'] = df['Date'].dt.month
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['dayofmonth'] = df['Date'].dt.day  # <--- Define this FIRST
    
    # 2. Create derivative columns
    df['is_winter'] = df['month'].isin([11,12,1,2]).astype(int)
    df['is_weekend'] = df['dayofweek'].isin([5,6]).astype(int)
    df['is_payday'] = df['dayofmonth'].apply(lambda x: 1 if x <= 5 else 0)

    if not is_future:
        # These only work for historical training data
        df['lag_7'] = df['Sales'].shift(7)
        df['rolling_mean_7'] = df['Sales'].rolling(window=7, min_periods=1).mean()
        return df.dropna()
    
    return df