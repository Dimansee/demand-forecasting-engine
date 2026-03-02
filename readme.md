# 📦 Advanced Demand Forecasting Tool (Hybrid-Ensemble)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_sharp_white.svg)](YOUR_DEPLOYED_LINK_HERE)

An end-to-end machine learning solution designed to forecast retail demand (e.g., apparel/hoodies) using a hybrid approach. This tool combines statistical models, additive models, and gradient boosting to capture trend, seasonality, and complex residuals.

## 🚀 Key Features
* **Hybrid Engine:** Blends SARIMA, Facebook Prophet, and XGBoost for robust predictions.
* **Residual Learning:** Uses XGBoost to specifically model the "errors" of the trend, capturing holiday spikes and payday effects.
* **Log-Transformation Pipeline:** Automatically stabilizes variance in volatile sales data.
* **Validation Lab:** Integrated "Actual vs. Forecast" tool with automated variance analysis and MAPE tracking.
* **Interactive Visualization:** Dynamic Plotly dashboards for SKU-level exploration.



## 🧠 The Methodology
This project implements a **weighted ensemble ($0.4, 0.3, 0.3$)**:
1.  **SARIMA:** Captures linear seasonality and autocorrelation.
2.  **Prophet:** Handles macroscopic trends and non-linear holiday effects (e.g., Diwali, Holi).
3.  **XGBoost:** A regressor trained on Prophet's residuals using calendar features (is_weekend, is_payday).

## 🛠️ Installation & Usage

### 1. Clone the repository
```bash
git clone [https://github.com/your-username/forecast-app.git](https://github.com/your-username/forecast-app.git)
cd forecast-app