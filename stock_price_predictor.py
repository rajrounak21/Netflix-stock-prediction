import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import pickle

# Load dataset
netflix = pd.read_csv('NFLX.csv')

# Convert 'Date' to datetime format and extract features
netflix['Date'] = pd.to_datetime(netflix['Date'])
netflix['year'] = netflix['Date'].dt.year
netflix['month'] = netflix['Date'].dt.month
netflix['day'] = netflix['Date'].dt.day
netflix.drop('Date', axis=1, inplace=True)

# Check for missing values
print("Missing values:\n", netflix.isnull().sum())

# Check for duplicates
print("Duplicate rows:", netflix.duplicated().sum())

# Display dataset info
print(netflix.info())
netflix.to_csv("stock_clean.csv")
# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(netflix.corr(), annot=True, cmap='coolwarm', cbar=True)
plt.title("Feature Correlation Heatmap")
plt.show()

# Pairplot to check scatter relationships
sns.pairplot(netflix)
plt.show()

# Distribution of each feature
for col in netflix.columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(netflix[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.show()

# Train-test split
X = netflix.drop('Close', axis=1)
y = netflix['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Make predictions
y_pred = lr.predict(X_test)

# Model evaluation
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"R² Score: {r2:.4f}")
print(f"Mean Squared Error: {mse:.4f}")

# Prediction function
def pred(Open, High, Low, Adj_Close, Volume, year, month, day):
    features = np.array([[Open, High, Low, Adj_Close, Volume, year, month, day]])
    features = scaler.transform(features)  # Use transform, not fit_transform
    prediction = lr.predict(features)
    return prediction[0]

# Example prediction
Open, High, Low, Adj_Close, Volume, year, month, day = 200.45, 250.45, 150.45, 100.34, 100.45, 2020, 8, 4
res = pred(Open, High, Low, Adj_Close, Volume, year, month, day)
print(f"Predicted Closing Price: {res:.2f}")

# Save the model
import joblib
joblib.dump(lr,'stock.pkl')

print("Model saved as stock.pkl")
