import streamlit as st
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np

st.set_page_config(page_title="Sales Forecast App", layout="centered")
st.title("📈 Sales Forecasting App")
st.write("Upload your sales CSV with `Date` and `Sales` columns to predict next month's sales.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    # Feature engineering
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['Lag_1'] = df['Sales'].shift(1)
    df['Lag_2'] = df['Sales'].shift(2)
    df = df.dropna()

    X = df[['Month', 'Year', 'Lag_1', 'Lag_2']]
    y = df['Sales']

    X_train, X_test = X[:-3], X[-3:]
    y_train, y_test = y[:-3], y[-3:]

    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    st.success(f"Model trained successfully ✅ | RMSE: {rmse:.2f}")

    # Forecast next month
    last_row = df.iloc[-1]
    next_month = (last_row['Month'] % 12) + 1
    next_year = last_row['Year'] + 1 if next_month == 1 else last_row['Year']
    lag_1 = last_row['Sales']
    lag_2 = df.iloc[-2]['Sales']

    next_X = pd.DataFrame([{
        'Month': next_month,
        'Year': next_year,
        'Lag_1': lag_1,
        'Lag_2': lag_2
    }])
    next_pred = model.predict(next_X)[0]
    st.metric(label="📊 Predicted Sales for Next Month", value=f"{next_pred:.2f}")
    # Confidence Interval (approximate, assuming normal distribution of errors)
    lower_bound = next_pred - 1.96 * rmse
    upper_bound = next_pred + 1.96 * rmse

    st.info(f"We are 95% confident that sales will be between **{lower_bound:.2f}** and **{upper_bound:.2f}** next month.")


    # Plot
    fig, ax = plt.subplots()
    ax.plot(df['Date'], y, label='Actual Sales')
    ax.plot(df['Date'].iloc[-3:], y_pred, label='Predicted (test)', color='orange')
    ax.set_title('Sales Forecast')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
