import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the CSV
df = pd.read_csv("sample_data.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# 2. Create features
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df['Lag_1'] = df['Sales'].shift(1)
df['Lag_2'] = df['Sales'].shift(2)

# Drop NA rows from lags
df = df.dropna()

# 3. Features and Target
X = df[['Month', 'Year', 'Lag_1', 'Lag_2']]
y = df['Sales']

# 4. Train-test split (last 3 months as test)
X_train, X_test = X[:-3], X[-3:]
y_train, y_test = y[:-3], y[-3:]

# 5. Model training
model = xgb.XGBRegressor()
model.fit(X_train, y_train)

# 6. Predict and evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.2f}")

# 7. Forecast next month
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

next_pred = model.predict(next_X)
print(f"Predicted sales for next month: {next_pred[0]:.2f}")

# 8. Plot it
plt.plot(df['Date'], y, label='Actual Sales')
plt.plot(df['Date'].iloc[-3:], y_pred, label='Predicted (test)')
plt.legend()
plt.title('Sales Forecast')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
