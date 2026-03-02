def ensemble_forecast(sarima, prophet, xgb):
    import numpy as np
    import pandas as pd

    # Ensure all are flat numpy arrays for calculation
    s = np.array(sarima).flatten()
    p = np.array(prophet).flatten()
    x = np.array(xgb).flatten()

    min_len = min(len(s), len(p), len(x))

    # Weighted Average
    final = (0.4 * s[:min_len]) + (0.3 * p[:min_len]) + (0.3 * x[:min_len])
    
    return final