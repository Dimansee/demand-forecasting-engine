import numpy as np

def mape(actual, forecast):
    """Calculates Mean Absolute Percentage Error"""
    actual, forecast = np.array(actual), np.array(forecast)
    mask = actual != 0
    if not np.any(mask):
        return 0
    return np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100

def calculate_variance(actual, forecast):
    """Calculates raw variance and percentage variance"""
    actual = np.array(actual)
    forecast = np.array(forecast)
    variance = actual - forecast
    # Avoid division by zero
    variance_pct = np.where(actual != 0, (variance / actual) * 100, 0)
    return variance, variance_pct