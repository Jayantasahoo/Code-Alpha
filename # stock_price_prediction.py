# stock_price_prediction.ipynb

# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import yfinance as yf

# Step 2: Load Data
ticker = 'AAPL'  # Apple Inc. as an example
start_date = '2010-01-01'
end_date = '2020-12-31'
data = yf.download(ticker, start=start_date, end=end_date)

# Step 3: Preprocess Data
data = data[['Close']]
data = data.dropna()

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create training and test sets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Create datasets with time steps
def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        X.append(a)
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape input to be [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Step 4: Build LSTM Model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Step 5: Train the Model
model.fit(X_train, y_train, batch_size=1, epochs=1)

# Step 6: Make Predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform predictions
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Inverse transform actual values
y_train = scaler.inverse_transform([y_train])
y_test = scaler.inverse_transform([y_test])

# Step 7: Evaluate the Model
# Plot baseline and predictions
plt.figure(figsize=(14, 5))
plt.plot(data.index[:len(train_predict)], y_train.flatten(), label='Train Actual')
plt.plot(data.index[len(train_predict):len(train_predict) + len(test_predict)], y_test.flatten(), label='Test Actual')
plt.plot(data.index[:len(train_predict)], train_predict.flatten(), label='Train Predict')
plt.plot(data.index[len(train_predict):len(train_predict) + len(test_predict)], test_predict.flatten(), label='Test Predict')
plt.legend()
plt.show()

# Calculate RMSE
from sklearn.metrics import mean_squared_error
import math
train_rmse = math.sqrt(mean_squared_error(y_train[0], train_predict[:,0]))
test_rmse = math.sqrt(mean_squared_error(y_test[0], test_predict[:,0]))
print(f"Train RMSE: {train_rmse}")
print(f"Test RMSE: {test_rmse}")
