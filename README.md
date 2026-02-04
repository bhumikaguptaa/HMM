# Gold Price Regime Detection using Hidden Markov Models (HMM)

This project applies a **Gaussian Hidden Markov Model (HMM)** to historical gold price data to detect "hidden" market regimes (e.g., Low Volatility vs. High Volatility states). 

Unlike traditional technical analysis that relies on lagging indicators, this probabilistic approach attempts to decode the underlying market state driving price returns.

### Files
* **`myHMM.py`**: The core Python script that performs data ingestion, log-return calculation, model fitting, and visualization.
* **`goldstock v1.csv`**: Historical daily gold price data containing `Date` and `Close` columns.
 
### Methodology
1. **Data Preprocessing**: Converts daily Close prices into **Log Returns** to ensure stationarity.
2. **Model Fitting**: Trains a 2-Component Gaussian HMM. The model assumes the market switches between two latent states with distinct Gaussian distributions of returns.
3. **Analysis**:
   * **Volatility Comparison**: Calculates mean and variance for each regime.
   * **Persistence**: Measures the longest continuous run (in days) for each state.
   * **Switch Detection**: Identifies specific dates where the market regime flipped.
