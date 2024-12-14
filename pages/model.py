import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go

@st.cache_data
def get_data(start, end):
    symbol = 'BTC-USD' 
    data = yf.download(symbol, start=start, end=end)
    data.reset_index(inplace=True)
    data['Date'] = data['Date'].dt.date
    return data

start_date = '2020-01-01'
end_date = '2024-01-01'
bitcoin_data = get_data(start_date, end_date)


scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(bitcoin_data[['Adj Close']]) 

def create_sequences(data, time_step=60):
    X = []
    y = []
    for i in range(len(data) - time_step):
        X.append(data[i:i + time_step, 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
X, y = create_sequences(scaled_data, time_step)

X = X.reshape(X.shape[0], X.shape[1], 1)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    tf.keras.layers.LSTM(units=50, return_sequences=False),
    tf.keras.layers.Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=10, batch_size=32)

predictions = model.predict(X_test)

predicted_prices = scaler.inverse_transform(predictions)
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))

fig = go.Figure()

fig.add_trace(go.Scatter(x=bitcoin_data['Date'][train_size:], y=actual_prices.flatten(), mode='lines', name='Actual Prices'))

fig.add_trace(go.Scatter(x=bitcoin_data['Date'][train_size:], y=predicted_prices.flatten(), mode='lines', name='Predicted Prices'))

fig.update_layout(title=f'Bitcoin Price Prediction (RMSE: {rmse:.2f})',
                  xaxis_title='Date',
                  yaxis_title='Price (USD)')

st.plotly_chart(fig)

st.write(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
