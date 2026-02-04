import sys
from matplotlib.patches import Patch
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM

# --------------------------
# 1. LOAD AND CLEAN DATA
# --------------------------
df = pd.read_csv("goldstock v1.csv") 
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# compute log returns
df['Return'] = np.log(df['Close'] / df['Close'].shift(1))
df = df.dropna()

returns = df['Return'].values.reshape(-1, 1) 

# --------------------------
# 2. FIT 2-STATE GAUSSIAN HMM
# --------------------------
model = GaussianHMM(
    n_components=2, 
    covariance_type="full",
    n_iter=200
) 

model.fit(returns) 

hidden_states = model.predict(returns)

# --------------------------
# 3. PRINT PARAMETERS
# --------------------------
print("Transition Matrix:")
print(model.transmat_)

print("\nMeans (per state):")
print(model.means_)

print("\nVariances (per state):")
print(model.covars_)

# --------------------------
# 4. PLOT PRICE WITH STATES
# --------------------------
plt.figure(figsize=(12,6))
colors = ['blue', 'red']

for state in range(2):
    idx = (hidden_states == state)
    plt.plot(df['Date'].iloc[idx], df['Close'].iloc[idx], '.', 
             color=colors[state], label=f"State {state}")

plt.title("Gold Prices Colored by HMM Regimes")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.show()

# --------------------------
# 5. PLOT RETURNS BY STATE
# --------------------------
plt.figure(figsize=(10,5))
for state in range(2):
    plt.hist(df['Return'][hidden_states==state], bins=50, alpha=0.6, 
             label=f"State {state}")
plt.title("Distribution of Returns by Hidden State")
plt.xlabel("Return")
plt.ylabel("Frequency")
plt.legend()
plt.show()


# ============================================================
# ADDITIONAL ANALYSIS TO ANSWER RESEARCH QUESTIONS
# ============================================================

df['State'] = hidden_states 


# --------------------------
# 1. Percentage of time spent in each regime
# --------------------------
print("="*50)
state_counts = df['State'].value_counts(normalize=True) * 100
print("\nPercentage of days in each regime (%):")
print(state_counts) 
print("="*50)

# --------------------------
# 2. Regime statistics (volatility comparison)
# --------------------------
print("="*50)
regime_stats = df.groupby('State')['Return'].agg(['mean','std','var','count'])
print("\nRegime Statistics:")
print(regime_stats)
print("="*50)

# --------------------------
# 3. Detect regime switches (dates)
# --------------------------
print("="*50)
df['Switch'] = df['State'].diff().abs()
switch_dates = df[df['Switch'] == 1]['Date']
print("\nDates when regime switched:")
print(switch_dates.head(20))  # print first 20 switches
print("="*50)

# --------------------------
# 4. Longest continuous periods in each regime
# --------------------------
print("="*50)
df['Regime_Run'] = (df['State'] != df['State'].shift(1)).cumsum()
run_lengths = df.groupby(['State','Regime_Run']).size()

print("\nLongest continuous runs per regime:")
print(run_lengths.groupby(level=0).max())
print("="*50)

# --------------------------
# 5. Summary Table for Presentation
# --------------------------
print("="*50)
summary_df = pd.DataFrame({
    "Mean Return (μ)": model.means_.flatten(),
    "Variance (σ²)": [c[0][0] for c in model.covars_],
    "Days in State (%)": state_counts.sort_index().values
})
print("\nSUMMARY TABLE (Copy This Into Slides):")
print(summary_df)
