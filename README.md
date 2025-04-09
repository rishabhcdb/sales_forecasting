# 📈 Sales Forecasting Machine learning Model
A real-time, ML-powered Streamlit web application that predicts future sales based on historical data using the XGBoost regression model.

Upload your CSV sales data and get instant predictions, RMSE evaluation, and clean visualizations — all in one click.

---

## 🚀 Features

- Upload historical sales data via CSV (`Date`, `Sales` columns)
- Performs preprocessing and **feature engineering** (lag features, date handling)
- Trains an **XGBoost regression model** on the fly
- Calculates and displays **RMSE** to indicate model accuracy
- Forecasts next period's sales (monthly/weekly)
- Visualizes **actual vs predicted sales** using Matplotlib
- Built with an intuitive **Streamlit UI** — no coding needed to use!

---

## 🛠 Tech Stack

- **Python**
- **XGBoost** – ML model for time series regression
- **scikit-learn** – train-test split, metrics (RMSE), data prep
- **Pandas** – data manipulation and feature creation
- **Matplotlib** – plotting actual vs predicted sales
- **Streamlit** – frontend for uploading CSV and interacting with the model

---

1. Clone the Repository
git clone https://github.com/your-username/sales-forecasting-app.git
cd sales-forecasting-app

2. Install Dependencies
pip install -r requirements.txt

Running the App Locally
streamlit run app.py


🧠 How It Works
Upload a CSV file with two columns: Date and Sales.

The app parses the file, creates lag features (Lag_1, Lag_2), and encodes the Date column into Month and Year.

Trains an XGBoostRegressor on this processed data.

Predicts the next month’s sales and calculates RMSE.

Displays a side-by-side plot of real vs predicted sales.

