import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Step 1: Download stock data
ticker = "RELIANCE.NS"
data = yf.download(ticker, period="1y")

# Optional: Save to CSV
data.to_csv("reliance_1y_data2.csv")

# Step 2.1: Create target variable: Next day's close
data["Next_Close"] = data["Close"].shift(-1)
data.dropna(inplace=True)  # Remove last row (has NaN in Next_Close)

# Remove ticker symbol from column headers if present (flatten MultiIndex)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)
# Save the final data (with Next_Close) to CSV
data.to_csv("reliance_1y_data_with_next_close.csv", index=True)



# Step 2.2: Define features and labels
features = ['Open', 'High', 'Low', 'Close', 'Volume']
X = data[features]
y = data["Next_Close"]

# Split into training and testing (no shuffle)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = lr_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)

# Step 2.3: Plot actual vs predicted
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Actual Close', linewidth=2)
plt.plot(y_pred, label='Predicted Close', linewidth=2)
plt.legend()
plt.title('Linear Regression - Next Day Close Price')
plt.xlabel('Test Sample Index')
plt.ylabel('Stock Price')
plt.grid(True)
plt.show()

# got Mean Squared Error (MSE): 411.6133054479076