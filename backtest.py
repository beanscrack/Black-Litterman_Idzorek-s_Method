import pandas as pd
import numpy as np


df = pd.read_excel("mktclosingprice.xlsx", index_col=0)  # Tickers are the index
df = df.T  # Transpose so dates become the index
df.index = pd.to_datetime(df.index)
df = df.sort_index()

# =daily return
returns = df.pct_change().dropna()
tickers = returns.columns
n_assets = len(tickers)
target_weights = np.array([1.38773517e-03, 9.57340088e-04, 8.31553216e-04, 4.84138068e-01,
 1.54748649e-01, 5.35784084e-04, 4.77310186e-04, 1.87404812e-01,
 4.24275721e-04, 1.69094473e-01])  
initial_capital = 100_000
equal_weights = np.ones(n_assets) / n_assets

initial_capital = 100_000

# --- Prepare tracking variables ---
portfolio_target = pd.Series(index=returns.index, dtype=float)
portfolio_equal = pd.Series(index=returns.index, dtype=float)

weights_target = pd.DataFrame(index=returns.index, columns=tickers, dtype=float)
weights_equal = pd.DataFrame(index=returns.index, columns=tickers, dtype=float)

# --- Initialize both portfolios ---
value_target = initial_capital * target_weights
value_equal = initial_capital * equal_weights

# --- Backtest loop ---
for i, date in enumerate(returns.index):
    daily_ret = returns.loc[date].values

    # Update values
    value_target *= (1 + daily_ret)
    value_equal *= (1 + daily_ret)

    nav_target = value_target.sum()
    nav_equal = value_equal.sum()

    # Store results
    portfolio_target[date] = nav_target
    portfolio_equal[date] = nav_equal

    weights_target.loc[date] = value_target / nav_target
    weights_equal.loc[date] = value_equal / nav_equal

    # Monthly rebalance
    if i < len(returns) - 1:
        next_date = returns.index[i + 1]
        if next_date.to_period('M') != date.to_period('M'):
            value_target = nav_target * target_weights
            value_equal = nav_equal * equal_weights

# --- Combine NAVs for comparison ---
nav_df = pd.DataFrame({
    'Target_Weighted_NAV': portfolio_target,
    'Equal_Weighted_NAV': portfolio_equal
})

# --- Save to Excel ---
with pd.ExcelWriter("portfolio_comparison.xlsx") as writer:
    nav_df.to_excel(writer, sheet_name="NAVs")
    weights_target.to_excel(writer, sheet_name="Weights_Target")
    weights_equal.to_excel(writer, sheet_name="Weights_Equal")
