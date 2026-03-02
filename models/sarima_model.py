from statsmodels.tsa.statespace.sarimax import SARIMAX

def sarima_forecast(series, steps=30):
    model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,12))
    model_fit = model.fit(disp=False)
    forecast = model_fit.forecast(steps=steps)
    return forecast