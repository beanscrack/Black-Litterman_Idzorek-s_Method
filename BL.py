import numpy as np
import pandas as pd
from numpy.linalg import inv
import cvxpy as cp
from scipy.optimize import minimize_scalar


df = pd.read_excel("mktclosingprice.xlsx").dropna()
df.set_index("Ticker", inplace=True)
price_df = df.T
price_df.index = pd.to_datetime(price_df.index)


weights_df = pd.read_excel("weights.xlsx").dropna()
weights_df.set_index("Ticker", inplace=True)
weights_df = weights_df.loc[price_df.columns]
weights = weights_df["Weight"].values
weights = weights / weights.sum()

tickers = price_df.columns.tolist()


returns = price_df.pct_change().dropna()
rf_year = 0.015
rf = (1 + rf_year) ** (1/252) - 1
excess_returns = returns - rf
cov = excess_returns.cov().values


weighted_returns = returns.dot(weights)
total_return = (weighted_returns + 1).prod() - 1
num_days = (weighted_returns.index[-1] - weighted_returns.index[0]).days
num_years = num_days / 365
annualized_return = (1 + total_return) ** (1 / num_years) - 1
port_var = weighted_returns.var() * 252
lambda_ = (annualized_return - rf_year) / port_var
pi = lambda_ * cov @ weights


k = 4
P = np.zeros((k, len(tickers)))
Q = np.array([])
idx = {t: i for i, t in enumerate(tickers)}
P[0, idx['600036.SH']] = 1; Q = np.append(Q, 0.7)
P[1, idx['300059.SZ']] = 1; Q = np.append(Q, 0.5)
P[2, idx['002594.SZ']] = 1; Q = np.append(Q, 0.4)
P[3, idx['000333.SZ']] = 1; Q = np.append(Q, 0.3)


view_count = P.shape[0]
omega_diag = []
tau = 0.025
confidences = np.array([0.8, 0.7, 0.9, 0.65])

for k in range(view_count):
    P_k = P[k:k+1, :]
    Q_k = Q[k] - rf_year

    
    er100 = pi + (tau * cov @ P_k.T) @ inv(P_k @ (tau * cov) @ P_k.T) @ (Q_k - P_k @ pi)
    w100 = inv(lambda_ * cov) @ er100
    D_k = w100.flatten() - weights
    weight_conf = np.zeros_like(D_k)
    weight_conf[P_k.flatten() != 0] = confidences[k]
    w_target = weights + D_k * weight_conf

    
    def objective(log_omega_k):
        omega_k = np.exp(log_omega_k)
        Omega_k = np.array([[omega_k]])
        
        ER_k = pi + (tau * cov @ P_k.T) @ inv(P_k @ (tau * cov) @ P_k.T + Omega_k) @ (Q_k - P_k @ pi)
        w_k = inv(lambda_ * cov) @ ER_k
        return np.sum((w_k.flatten() - w_target) ** 2)

    res = minimize_scalar(objective, bounds=(-10, 10), method='bounded')
    omega_diag.append(np.exp(res.x))

omega = np.diag(omega_diag)


mu = inv(inv(tau * cov) + P.T @ inv(omega) @ P) @ (inv(tau * cov) @ pi + P.T @ inv(omega) @ Q)
w_bl = inv(lambda_ * cov) @ mu
w_bl = w_bl / w_bl.sum()

print(w_bl)  




















'''
import numpy as np
import pandas as pd
from numpy.linalg import inv
import cvxpy as cp


df = pd.read_excel("mktclosingprice.xlsx").dropna()
df.set_index("Ticker", inplace=True)
price_df = df.T
price_df.index = pd.to_datetime(price_df.index)


weights_df = pd.read_excel("weights.xlsx").dropna()
weights_df.set_index("Ticker", inplace=True)
weights_df = weights_df.loc[price_df.columns]
weights = weights_df["Weight"].values
weights = weights / weights.sum()



tickers = price_df.columns.tolist()

returns = price_df.pct_change().dropna()
rf_year = 0.015
rf = (1 + rf_year) ** (1/252) - 1
daily_excess_return = returns - rf
cov = daily_excess_return.cov().values
weighted_returns = returns.dot(weights)
total_return = (weighted_returns + 1).prod() - 1
num_days = (weighted_returns.index[-1] - weighted_returns.index[0]).days
num_years = num_days / 365
annualized_return = (1 + total_return) ** (1 / num_years) - 1

var_excess = daily_excess_return.var()
port_var = weighted_returns.var() * 252
lambda_ = (annualized_return - rf_year) / port_var
pi = lambda_ * cov @ weights 

k = 4
P = np.zeros((4, len(tickers)))
Q = np.array([])
idx = {t: i for i, t in enumerate(tickers)}
P[0, idx['600036.SH']] = 1 # 80%
Q = np.append(Q, 0.7)
P[1, idx['300059.SZ']] = 1 # 70%
Q = np.append(Q, 0.5)
P[2, idx['002594.SZ']] = 1 # 90%
Q = np.append(Q, 0.4)
P[3, idx['000333.SZ']] = 1 # 65%
Q = np.append(Q, 0.3)


view_count = P.shape[0]
N = len(tickers)
omega_diag = []
tau = 0.025 # doesnt matter cancel out in optimization

for k in range(view_count):
    P_k = P[k:k+1, :]

    Q_k = np.array(Q[k]) - rf_year

    
    er100 = pi + ((tau * cov) @ P_k.T) @ inv(P_k @ (tau * cov) @ P_k.T)  @ (Q_k - P_k @ pi)

    w_100 = inv(lambda_ * cov) @ er100
    D_k = w_100.flatten() - weights
    confidence = np.array([0.8, 0.7, 0.9, 0.65])[k]


    w_target = weights + D_k * confidence
    def objective(log_omega_k):
        omega_k = np.exp(log_omega_k)
        Omega_k = np.array([[omega_k]])
        ER_k = pi + (tau * cov @ P_k.T) @ inv(P_k @ (tau * cov) @ P_k.T) @ (Q_k - (P_k @ pi))
        w_k = inv(cov) @ ER_k / lambda_
        return np.sum((w_k.flatten() - w_target) ** 2)
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(objective, bounds = (-10, 10), method = 'bounded')

    omega_diag.append(np.exp(res.x))

omega = np.diag(omega_diag)

mu = inv(inv(tau * cov) + P.T @ inv(omega) @ P) @ (inv(tau * cov) @ pi + P.T @ inv(omega) @ Q)
'''









'''
# === 5. Define Views ===
# Example views:
# View 1: 600519.SH outperforms 300750.SZ by 2%
# View 2: 601318.SH has absolute return of 5%
k = 4
P = np.zeros((k, len(tickers)))
Q = np.array([])
idx = {t: i for i, t in enumerate(tickers)}
P[0, idx['688169.SH']] =  1 #tech stock perform better then bank by 1.5%
P[0, idx['601288.SH']] = -1
Q = np.append(Q, 0.015)
P[1, idx['600276.SH']] = 1 #healthcare delivers 6% abs return
Q = np.append(Q, 0.1)
P[2, idx['000333.SZ']] = 0.5    # 3% higher
P[2, idx['000858.SZ']] = 0.5
Q = np.append(Q, 0.03)
P[3, idx['601186.SH']] = 0.5 #1% lower
P[3, idx['600018.SH']] = 0.5
Q = np.append(Q, -0.01)



tau = 0.025
omega = np.diag(np.diag(P @ (tau * cov) @ P.T))
'''
'''
middle = inv(tau * cov)
posterior_mean = inv(middle + P.T @ inv(omega) @ P) @ (middle @ pi + P.T @ inv(omega) @ Q)


w = cp.Variable(len(tickers))

target_te = 0.02
constraints = [cp.sum(w) == 1,
               w >= 0,
               cp.quad_form(w - weights, cov) <= target_te**2]

obj = cp.Maximize(w @ posterior_mean)
prob = cp.Problem(obj, constraints)
prob.solve()
opt_weights = w.value
opt_weights = np.maximum(opt_weights, 0)
opt_weights = opt_weights / opt_weights.sum()


result = pd.DataFrame({
    'Ticker': tickers,
    'Benchmark_Weight': weights,
    'BL_Weight': opt_weights
}).set_index('Ticker')
result['Change'] = result['BL_Weight'] - result['Benchmark_Weight']
pd.set_option('display.max_rows', None)
print(result.sort_values('BL_Weight', ascending=False))
'''