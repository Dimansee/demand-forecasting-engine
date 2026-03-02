import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta

# Model and Utility Imports
from models.sarima_model import sarima_forecast
from models.prophet_model import prophet_forecast
from models.xgb_model import xgb_forecast
from models.ensemble import ensemble_forecast
from utils.metrics import mape, calculate_variance

# --- DOCUMENTATION DOWNLOAD FEATURE ---
def download_pdf():
    with open("Forecasting_Documentation.pdf", "rb") as f:
        pdf_data = f.read()
    return pdf_data

st.sidebar.header("📚 Resources")

# Button to download the PDF locally
try:
    st.sidebar.download_button(
        label="📄 Download PDF Documentation",
        data=download_pdf(),
        file_name="Demand_Forecaster_User_Guide.pdf",
        mime="application/pdf",
        help="Download the full technical documentation and user guide."
    )
except FileNotFoundError:
    st.sidebar.warning("Documentation PDF not found in root directory.")


st.set_page_config(page_title="Demand Forecaster Pro", layout="wide")

st.title("📦 Advanced Demand Forecasting Tool")

# --- 1. TEMPLATE DOWNLOAD ---
with st.expander("ℹ️ New User? Download the CSV Template"):
    template_df = pd.DataFrame({
        'Date': [(pd.Timestamp.now() - timedelta(days=x)).strftime('%d-%m-%Y') for x in range(30, 0, -1)],
        'Sales': np.random.randint(50, 150, 30),
        'SKU': ['HOODIE01'] * 30
    })
    st.download_button("📥 Download Template", template_df.to_csv(index=False).encode('utf-8'), "template.csv", "text/csv")

# --- 2. UPLOAD HISTORICAL DATA ---
file = st.file_uploader("Step 1: Upload Historical Sales Data", type=['csv'])

if file:
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip().str.lower()
    
    # Auto-detection
    date_col = [col for col in df.columns if 'date' in col][0]
    sales_col = [col for col in df.columns if 'sale' in col or 'qty' in col][0]
    sku_col = [col for col in df.columns if 'sku' in col or 'item' in col][0]
    
    df.rename(columns={date_col: 'Date', sales_col: 'Sales', sku_col: 'SKU'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Date', 'Sales'])

    # SKU Selection
    sku_list = df['SKU'].unique()
    selected_sku = st.selectbox("Select SKU", sku_list)
    sku_df = df[df['SKU'] == selected_sku].sort_values('Date')
    series = sku_df.set_index('Date')['Sales']

    # --- 3. FORECAST SETTINGS ---
    st.subheader("Step 2: Forecast Settings")
    col_a, col_b = st.columns(2)
    with col_a:
        model_choice = st.selectbox("Select Model", ['Ensemble', 'SARIMA', 'Prophet', 'XGBoost'])
    with col_b:
        steps = st.slider("Days to Forecast", 7, 90, 30)

    # --- 4. RUN FORECAST ---
    series_log = np.log1p(series)
    sku_df_log = sku_df.copy()
    sku_df_log['Sales'] = np.log1p(sku_df_log['Sales'])

    with st.spinner('Generating Forecast...'):
        sarima_log = sarima_forecast(series_log, steps)
        prophet_full = prophet_forecast(sku_df_log, steps)
        prophet_history = prophet_full.iloc[:-steps]
        prophet_future = prophet_full.iloc[-steps:]
        xgb_log_res = xgb_forecast(sku_df_log, prophet_history, steps)

        if model_choice == 'SARIMA':
            final_log = sarima_log.values
        elif model_choice == 'Prophet':
            final_log = prophet_future['yhat'].values
        elif model_choice == 'XGBoost':
            final_log = prophet_future['trend'].values + xgb_log_res
        else:
            final_log = ensemble_forecast(sarima_log, prophet_future['yhat'], (prophet_future['trend'].values + xgb_log_res))

        final_forecast = np.maximum(np.expm1(final_log), 0)

    # Prepare Forecast DataFrame
    future_dates = pd.date_range(start=series.index[-1] + timedelta(days=1), periods=steps)
    forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': final_forecast})

    # --- 5. MAIN CHART ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, name='History'))
    fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Forecast'], name='Forecast', line=dict(dash='dash')))
    st.plotly_chart(fig, use_container_width=True)

    # --- 6. VALIDATION FEATURE ---
    st.divider()
    st.header("🔍 Actual vs Forecast Validation")
    
    val_file = st.file_uploader("Step 3: Upload 'Actual' Sales for the Forecast Period", type=['csv'])

    if val_file:
        v_df = pd.read_csv(val_file)
        v_df.columns = v_df.columns.str.strip().str.lower()
        
        try:
            # FIXED: Corrected the index from [c] to [0]
            vd_col = [c for c in v_df.columns if 'date' in c][0]
            vs_col = [c for c in v_df.columns if 'sale' in c or 'qty' in c][0]
            
            # Robust date parsing
            v_df[vd_col] = pd.to_datetime(v_df[vd_col], format='mixed', dayfirst=True, errors='coerce')
            v_df = v_df.dropna(subset=[vd_col]) 
            
            # Merge and Compare
            comparison = pd.merge(forecast_df, v_df[[vd_col, vs_col]], left_on='Date', right_on=vd_col)
            comparison = comparison.rename(columns={vs_col: 'Actual'})

            if not comparison.empty:
                var, var_pct = calculate_variance(comparison['Actual'], comparison['Forecast'])
                comparison['Variance'] = var
                comparison['Variance %'] = var_pct

                # Comparison Plot
                fig_v = go.Figure()
                fig_v.add_trace(go.Scatter(x=comparison['Date'], y=comparison['Actual'], name='Actual Sales', line=dict(color='green')))
                fig_v.add_trace(go.Scatter(x=comparison['Date'], y=comparison['Forecast'], name='Forecasted', line=dict(color='orange', dash='dash')))
                st.plotly_chart(fig_v, use_container_width=True)

                # Results Table
                st.subheader("Variance Analysis Table")
                st.dataframe(comparison.style.format({
                    'Forecast': '{:.2f}',
                    'Actual': '{:.2f}',
                    'Variance': '{:.2f}',
                    'Variance %': '{:.2f}%'
                }).highlight_max(axis=0, subset=['Variance %'], color='#ff4b4b'))
                
                real_mape = mape(comparison['Actual'], comparison['Forecast'])
                st.metric("Real-World MAPE", f"{round(real_mape, 2)}%")
            else:
                st.warning("No matching dates found. Make sure the uploaded file covers the forecasted dates.")
                
        except IndexError:
            st.error("Validation file error: Could not find columns containing 'date' or 'sales'.")