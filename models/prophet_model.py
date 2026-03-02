from prophet import Prophet
import pandas as pd

def prophet_forecast(df, steps=30):

    df_prophet = df.rename(columns={'Date':'ds','Sales':'y'})

    # Holiday Calendar (India Example)
    holidays = pd.DataFrame({
        'holiday': 'festival',
        'ds': pd.to_datetime([
            '2024-10-31',  # Diwali
            '2024-03-25',  # Holi
            '2024-01-26',  # Republic Day
            '2024-08-15'   # Independence Day
        ])
    })

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.5, # Increased from default 0.05 for more flexibility
        seasonality_prior_scale=10.0 # Allows seasonality to fit larger swings
    )
    # Add country-specific holidays automatically
    model.add_country_holidays(country_name='IN')
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=steps)
    forecast = model.predict(future)

    return forecast[['ds','yhat','trend']]