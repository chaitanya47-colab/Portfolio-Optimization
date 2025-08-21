# Portfolio-Optimization
This project focuses on Portfolio Optimization – the process of selecting the best portfolio (asset allocation) to achieve maximum returns while managing risk. The goal is to analyze stock market data, apply different optimization strategies, and visualize results to help investors make informed decisions.

import pandas as pd
import numpy as np
df = pd.read_excel("/content/chaitanya Yahoo Finance.xlsx")
df
# Import necessary libraries
import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# List of stock tickers
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

# Fetch stock data
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period="2y")
    return df['Close']

# Load and prepare stock data
stock_data = {ticker: get_stock_data(ticker) for ticker in tickers}
prices = pd.DataFrame(stock_data)

# Normalize data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = {ticker: scaler.fit_transform(prices[ticker].values.reshape(-1,1)) for ticker in tickers}

# Create sequences for LSTM
def create_sequences(data, time_steps=60):
    X, y = [], []
    for i in range(len(data) - time_steps - 1):
        X.append(data[i:(i + time_steps), 0])
        y.append(data[i + time_steps, 0])
    return np.array(X), np.array(y)

# Train LSTM models
models = {}
predicted_returns = {}

for ticker in tickers:
    print(f"Training LSTM for {ticker}...")
    time_steps = 60
    data = scaled_data[ticker]
    X, y = create_sequences(data, time_steps)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_steps, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=20, batch_size=32, verbose=0)

    models[ticker] = model
    last_60_days = data[-time_steps:].reshape(1, -1, 1)
    predicted_price = model.predict(last_60_days)[0, 0]
    predicted_price = scaler.inverse_transform([[predicted_price]])[0, 0]
    last_actual_price = prices[ticker].iloc[-1]
    predicted_return = (predicted_price - last_actual_price) / last_actual_price
    predicted_returns[ticker] = predicted_return

# Convert to expected returns Series
expected_returns = pd.Series(predicted_returns)
import matplotlib.pyplot as plt

# Step 1: Calculate daily returns and the covariance matrix
returns = prices.pct_change().dropna()
cov_matrix = returns.cov()

# Step 2: Define a function to calculate portfolio return and volatility
def portfolio_stats(weights, expected_returns, cov_matrix):
    portfolio_return = np.sum(weights * expected_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_volatility

# Step 3: Simulate random portfolios to construct the Efficient Frontier
num_portfolios = 10000
results = {
    'returns': [],
    'volatilities': [],
    'sharpe_ratios': [],
    'weights': []
}

for _ in range(num_portfolios):
    weights = np.random.random(len(tickers))
    weights /= np.sum(weights)

    port_return, port_volatility = portfolio_stats(weights, expected_returns, cov_matrix)
    sharpe_ratio = port_return / port_volatility

    results['returns'].append(port_return)
    results['volatilities'].append(port_volatility)
    results['sharpe_ratios'].append(sharpe_ratio)
    results['weights'].append(weights)

# Convert to NumPy arrays
returns_array = np.array(results['returns'])
vol_array = np.array(results['volatilities'])
sharpe_array = np.array(results['sharpe_ratios'])

# Step 4: Find the portfolio with the highest Sharpe Ratio
max_sharpe_idx = np.argmax(sharpe_array)
optimal_weights = results['weights'][max_sharpe_idx]
optimal_return = returns_array[max_sharpe_idx]
optimal_volatility = vol_array[max_sharpe_idx]

# Step 5: Plot Efficient Frontier
plt.figure(figsize=(10, 6))
plt.scatter(vol_array, returns_array, c=sharpe_array, cmap='viridis', alpha=0.7)
plt.colorbar(label='Sharpe Ratio')
plt.scatter(optimal_volatility, optimal_return, color='red', marker='*', s=200, label='Max Sharpe Ratio Portfolio')
plt.title('Efficient Frontier (Based on LSTM-Predicted Returns)')
plt.xlabel('Volatility (Risk)')
plt.ylabel('Expected Return')
plt.legend()
plt.grid(True)
plt.show()



# Step 6: Display Portfolio Metrics
sharpe_ratio = optimal_return / optimal_volatility

print("\n--- Optimal Portfolio Metrics ---")
print(f"Expected Return      : {optimal_return:.4f} or {optimal_return*100:.2f}%")
print(f"Volatility (Risk)    : {optimal_volatility:.4f} or {optimal_volatility*100:.2f}%")
print(f"Sharpe Ratio         : {sharpe_ratio:.4f}")

!pip install deap
!pip install yfinance
import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from deap import base, creator, tools, algorithms
import random


# List of stock tickers
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

# Fetch stock data
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period="2y")
    return df['Close']

# Load and prepare stock data
stock_data = {ticker: get_stock_data(ticker) for ticker in tickers}
prices = pd.DataFrame(stock_data)

# Normalize data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = {ticker: scaler.fit_transform(prices[ticker].values.reshape(-1,1)) for ticker in tickers}

# Create sequences for LSTM
def create_sequences(data, time_steps=60):
    X, y = [], []
    for i in range(len(data) - time_steps - 1):
        X.append(data[i:(i + time_steps), 0])
        y.append(data[i + time_steps, 0])
    return np.array(X), np.array(y)

# Train LSTM models
models = {}
predicted_returns = {}

for ticker in tickers:
    print(f"Training LSTM for {ticker}...")
    time_steps = 60
    data = scaled_data[ticker]
    X, y = create_sequences(data, time_steps)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_steps, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=20, batch_size=32, verbose=0)

    models[ticker] = model
    last_60_days = data[-time_steps:].reshape(1, -1, 1)
    predicted_price = model.predict(last_60_days)[0, 0]
    predicted_price = scaler.inverse_transform([[predicted_price]])[0, 0]
    last_actual_price = prices[ticker].iloc[-1]
    predicted_return = (predicted_price - last_actual_price) / last_actual_price
    predicted_returns[ticker] = predicted_return

# Convert to expected returns Series
expected_returns = pd.Series(predicted_returns)

# Calculate covariance matrix of returns
# Assuming 'prices' DataFrame contains the historical prices of the assets
returns = prices.pct_change().dropna() # Calculate daily returns and drop NaN values
cov_matrix = returns.cov() # Calculate the covariance matrix of returns


# Sharpe ratio function (to maximize)
def negative_sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate=0.02):
    portfolio_return = np.sum(expected_returns * weights)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return - (portfolio_return - risk_free_rate) / portfolio_volatility

# Assuming you have a function named portfolio_stats
def portfolio_stats(weights, expected_returns, cov_matrix):
    portfolio_return = np.sum(expected_returns * weights)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_volatility


# Setup DEAP GA environment
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def create_individual():
    weights = np.random.random(len(tickers))
    weights /= np.sum(weights)
    return weights.tolist()

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    weights = np.array(individual)
    if np.sum(weights) != 1:
        return (1000000,)  # Penalize if weights don’t sum to 1
    return (negative_sharpe_ratio(weights, expected_returns, cov_matrix),)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# Run the Genetic Algorithm
population = toolbox.population(n=100)
algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=10, verbose=True)

# Extract best weights
best_individual = tools.selBest(population, 1)[0]
best_weights = np.array(best_individual)

# Show best result
print("\nBest Portfolio Weights:")
for ticker, weight in zip(tickers, best_weights):
    print(f"{ticker}: {weight:.2f}")

# Final portfolio return & risk
optimal_return, optimal_volatility = portfolio_stats(best_weights, expected_returns, cov_matrix)
print(f"\nOptimal Portfolio Expected Return: {optimal_return:.4f}")
print(f"Optimal Portfolio Volatility (Risk): {optimal_volatility:.4f}")

# Assuming you have necessary imports for plotting (e.g., matplotlib.pyplot as plt)
import matplotlib.pyplot as plt
# Generate random data for portfolio volatilities and returns for demonstration purposes
portfolio_volatilities = np.random.rand(100)
portfolio_returns = np.random.rand(100)

# Plot optimal portfolio on Efficient Frontier
plt.scatter(portfolio_volatilities, portfolio_returns, c=portfolio_returns/portfolio_volatilities, cmap='YlGnBu')
plt.colorbar(label='Sharpe Ratio')
plt.scatter(optimal_volatility, optimal_return, color='red', marker='*', s=200, label="Optimal Portfolio")
plt.title('Efficient Frontier with Optimal Portfolio')
plt.xlabel('Volatility (Risk)')
plt.ylabel('Return')
plt.legend(loc='upper left')
plt.show()

import matplotlib.pyplot as plt

# Calculate daily returns and covariance matrix
returns = prices.pct_change().dropna()
cov_matrix = returns.cov()

# Portfolio statistics function
def portfolio_stats(weights, expected_returns, cov_matrix):
    portfolio_return = np.sum(weights * expected_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_volatility

# Efficient Frontier plotting (for comparison)
portfolio_returns = []
portfolio_volatilities = []

for _ in range(10000):
    weights = np.random.random(len(tickers))
    weights /= np.sum(weights)
    return_, volatility = portfolio_stats(weights, expected_returns, cov_matrix)
    portfolio_returns.append(return_)
    portfolio_volatilities.append(volatility)

portfolio_returns = np.array(portfolio_returns)
portfolio_volatilities = np.array(portfolio_volatilities)



# Step 1: Calculate Sharpe ratios for all portfolios
sharpe_ratios = portfolio_returns / portfolio_volatilities

# Step 2: Find the index of the portfolio with the max Sharpe ratio
max_sharpe_idx = np.argmax(sharpe_ratios)
optimal_return = portfolio_returns[max_sharpe_idx]
optimal_volatility = portfolio_volatilities[max_sharpe_idx]
optimal_sharpe_ratio = sharpe_ratios[max_sharpe_idx]

# Step 3: Plot the Efficient Frontier
plt.figure(figsize=(10, 6))
scatter = plt.scatter(portfolio_volatilities, portfolio_returns, c=sharpe_ratios, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Sharpe Ratio')
plt.scatter(optimal_volatility, optimal_return, color='red', marker='*', s=200, label='Optimal Portfolio')
plt.title('Efficient Frontier with Optimal Portfolio')
plt.xlabel('Volatility (Risk)')
plt.ylabel('Expected Return')
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 4: Print the optimal values
print("\n--- Optimal Portfolio Metrics ---")
print(f"Expected Return      : {optimal_return:.4f} or {optimal_return*100:.2f}%")
print(f"Volatility (Risk)    : {optimal_volatility:.4f} or {optimal_volatility*100:.2f}%")
print(f"Sharpe Ratio         : {optimal_sharpe_ratio:.4f}")

# Step 5 (Optional): Visualize metrics in a bar chart
metrics = ['Expected Return', 'Volatility (Risk)', 'Sharpe Ratio']
values = [optimal_return, optimal_volatility, optimal_sharpe_ratio]

plt.figure(figsize=(6, 4))
bars = plt.bar(metrics, values, color=['green', 'orange', 'blue'])
plt.title('Optimal Portfolio Performance Metrics')
plt.ylabel('Value')
plt.ylim(0, max(values) * 1.2)

# Add value labels on top
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f'{yval:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

