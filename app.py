import streamlit as st
import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load trained model
model = joblib.load("stock.pkl")

# Load dataset to fit scaler (same as training data)
df = pd.read_csv("NFLX.csv")
df['Date'] = pd.to_datetime(df['Date'])
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day
df.drop('Date', axis=1, inplace=True)

# Standardize data
scaler = StandardScaler()
scaler.fit(df.drop(columns=['Close']))

# Streamlit UI
st.title("📈 Netflix Stock Price Prediction")
st.sidebar.markdown("Enter stock details to predict the closing price.")

# Input fields
Open = st.sidebar.number_input("Open Price", min_value=0.0, format="%.2f")
High = st.sidebar.number_input("High Price", min_value=0.0, format="%.2f")
Low = st.sidebar.number_input("Low Price", min_value=0.0, format="%.2f")
Adj_Close = st.sidebar.number_input("Adjusted Close Price", min_value=0.0, format="%.2f")
Volume = st.sidebar.number_input("Trading Volume", min_value=0.0, format="%.2f")
year = st.sidebar.number_input("Year", min_value=2000, max_value=2100, step=1)
month = st.sidebar.number_input("Month", min_value=1, max_value=12, step=1)
day = st.sidebar.number_input("Day", min_value=1, max_value=31, step=1)

# Prediction function
def predict_price(Open, High, Low, Adj_Close, Volume, year, month, day):
    features = np.array([[Open, High, Low, Adj_Close, Volume, year, month, day]])
    features = scaler.transform(features)
    prediction = model.predict(features)
    return prediction[0]

# Predict button
if st.sidebar.button("Predict Closing Price"):
    predicted_price = predict_price(Open, High, Low, Adj_Close, Volume, year, month, day)
    st.success(f"Predicted Closing Price: ${predicted_price:.2f}")
