import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns


@st.cache_data
def get_data(start, end):
    symbol = 'BTC-USD' 
    data = yf.download(symbol, start=start, end=end)

    data.reset_index(inplace=True)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [' '.join(col).strip() for col in data.columns]
    data['Date'] = data['Date'].dt.date

    return data

st.write('## ₿ Bitcoin Price Analysis')

date1, date2 = st.columns(2)

start = date1.date_input('Start Date')
end = date2.date_input('End Date', min_value=start)

bitcoin_data = get_data(start, end)

st.write('\n ### Bitcoin DataFrame \n')

st.dataframe(bitcoin_data)

plotting = bitcoin_data[['Date', 'Adj Close BTC-USD']].copy()

st.write('\n ### Bitcoin Plot \n')

st.line_chart(data=plotting.set_index('Date'), x_label='Date', y_label='Closing Price')

st.write('\n ### Bitcoin Range Analysis \n')

col1, col2 = st.columns(2)

if not bitcoin_data['Date'].empty:
    start_date = col1.select_slider('First Date', options=bitcoin_data['Date'])
    end_date = col2.select_slider('Last Date', options=bitcoin_data['Date'])
    
    start_index = bitcoin_data[bitcoin_data['Date'] == start_date].index[0]  # Find the index for start date
    end_index = bitcoin_data[bitcoin_data['Date'] == end_date].index[0]  # Find the index for end date

    col1, col2, col3 = st.columns(3)

    low_start = bitcoin_data.loc[start_index, 'Low BTC-USD']
    low_end = bitcoin_data.loc[end_index, 'Low BTC-USD']
    low_diff = low_end - low_start
    col1.metric('Low', low_end, delta=str(low_diff), label_visibility="visible")

    high_start = bitcoin_data.loc[start_index, 'High BTC-USD']
    high_end = bitcoin_data.loc[end_index, 'High BTC-USD']
    high_diff = high_end - high_start
    col2.metric('High', high_end, delta=str(high_diff))

    volume_start = bitcoin_data.loc[start_index, 'Volume BTC-USD']
    volume_end = bitcoin_data.loc[end_index, 'Volume BTC-USD']
    volume_diff = volume_end - volume_start
    col3.metric('Volume', volume_end, delta=str(volume_diff))

    price_start = bitcoin_data.loc[start_index, 'Adj Close BTC-USD']
    price_end = bitcoin_data.loc[end_index, 'Adj Close BTC-USD']
    price_diff = price_end - price_start
    price_pct_change = (price_diff / price_start) * 100

    bitcoin_data['Daily Return'] = bitcoin_data['Adj Close BTC-USD'].pct_change()
    volatility = bitcoin_data['Daily Return'].std() * np.sqrt(365)  # Annualized volatility

    st.write(f"### Analysis for the selected date range:")
    st.write(f"- **Price Change**: The Bitcoin price changed by **{price_diff:.2f} USD** ({price_pct_change:.2f}%) from {start_date} to {end_date}.")
    st.write(f"- **Volatility**: The annualized volatility during this period is **{volatility:.2f}%**.")
    
    st.write("### Volume and Price Correlation:")
    correlation = bitcoin_data['Volume BTC-USD'].corr(bitcoin_data['Adj Close BTC-USD'])
    st.write(f"The correlation between price and volume is **{correlation:.2f}**. A positive value indicates that when the price increases, the volume tends to increase as well.")

    if price_pct_change > 0:
        st.write(f"**Bullish Trend**: Bitcoin shows an upward trend during this period.")
    elif price_pct_change < 0:
        st.write(f"**Bearish Trend**: Bitcoin shows a downward trend during this period.")
    else:
        st.write(f"**Neutral Trend**: Bitcoin's price remained relatively stable during this period.")

st.write('## CandleStick Chart')
    
fig = go.Figure(data=[go.Candlestick(x=bitcoin_data['Date'],
                                     open=bitcoin_data['Open BTC-USD'],
                                     high=bitcoin_data['High BTC-USD'],
                                     low=bitcoin_data['Low BTC-USD'],
                                     close=bitcoin_data['Adj Close BTC-USD'])])

fig.update_layout(title='Candlestick Chart for Bitcoin',
                  xaxis_title='Date',
                  yaxis_title='Price (USD)')

st.plotly_chart(fig)

st.write('## Volume-Price Scatter PLot ')

fig = go.Figure(data=go.Scatter(x=bitcoin_data['Date'],
                                y=bitcoin_data['Volume BTC-USD'],
                                mode='markers',
                                marker=dict(color=bitcoin_data['Adj Close BTC-USD'], colorscale='Viridis', size=8)))

fig.update_layout(title='Volume vs Price',
                  xaxis_title='Date',
                  yaxis_title='Volume',
                  coloraxis_colorbar=dict(title="Closing Price (USD)"))

st.plotly_chart(fig)

st.write('## Histogram')

bitcoin_data['Daily Return'] = bitcoin_data['Adj Close BTC-USD'].at()

fig = go.Figure(data=[go.Histogram(x=bitcoin_data['Daily Return'], nbinsx=50)])

fig.update_layout(title='Distribution of Daily Returns',
                  xaxis_title='Daily Return',
                  yaxis_title='Frequency')

st.plotly_chart(fig)

st.write('## Corrolation HeatMap')

corr_matrix = bitcoin_data[['Open BTC-USD', 'High BTC-USD', 'Low BTC-USD', 'Volume BTC-USD', 'Adj Close BTC-USD']].corr()

# Plot heatmap using seaborn (convert to Plotly for integration with Streamlit)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
plt.title('Correlation Heatmap')

# Convert Seaborn plot to Plotly
st.pyplot(fig)