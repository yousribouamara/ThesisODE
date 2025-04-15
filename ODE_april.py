import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.interpolate import interp1d

""" ---------------------------------------------
# Model Purpose:
# Simulate cancer cell growth (C) influenced by macrophage activity (M1 and M2)
# ---------------------------------------------
# Model: dC/dt = (r_basal + r_M1(A_M1) + r_M2(A_M2)) * C
# - C: cancer cell count
# - A_M1, A_M2: activity levels from qPCR (normalized expression)
# - r_M1, r_M2: logistic functions describing macrophage influence
# ---------------------------------------------
"""
# Step 1: Simulated qPCR activity data (Replace with your qPCR-based relative expression)
timepoints = np.array([1, 3, 5])  # days
A_M1_values = np.array([1.0, 1.8, 2.5])  # example M1 activity (e.g., TNF-α expression)
A_M2_values = np.array([1.2, 2.2, 2.8])  # example M2 activity (e.g., ARG1 expression)

# Interpolate activity as continuous functions of time
A_M1_interp = interp1d(timepoints, A_M1_values, kind='linear', fill_value="extrapolate")
A_M2_interp = interp1d(timepoints, A_M2_values, kind='linear', fill_value="extrapolate")

# Step 2: Define logistic response functions for macrophage influence
def r_M1(A, Rmax, k, theta):
    """M1 reduces cancer growth; logistic inhibition."""
    return -Rmax / (1 + np.exp(-k * (A - theta)))

def r_M2(A, Rmax, k, theta):
    """M2 promotes cancer growth; logistic stimulation."""
    return Rmax / (1 + np.exp(-k * (A - theta)))

# Step 3: Cancer growth ODE
def cancer_growth(C, t, params):
    r_basal, Rmax1, k1, theta1, Rmax2, k2, theta2 = params
    A1 = A_M1_interp(t)
    A2 = A_M2_interp(t)
    r1 = r_M1(A1, Rmax1, k1, theta1)
    r2 = r_M2(A2, Rmax2, k2, theta2)
    dCdt = (r_basal + r1 + r2) * C
    return dCdt

# Step 4: Simulate true cancer cell dynamics for generating synthetic data
C0 = 50000  # initial cell count
t_full = np.linspace(0, 5, 100)  # time in days
true_params = [0.2, 0.5, 3, 1.0, 0.4, 3, 1.0]  # true parameter values
C_simulated = odeint(cancer_growth, C0, t_full, args=(true_params,))

# Sample at measurement points (simulate qPCR-based observation of cancer count)
C_obs = odeint(cancer_growth, C0, timepoints, args=(true_params,)).flatten()
np.random.seed(42)
C_obs_noisy = C_obs * (1 + 0.05 * np.random.randn(len(C_obs)))  # add 5% noise

# Step 5: Define model fitting functions
def model_prediction(params, t_obs):
    return odeint(cancer_growth, C0, t_obs, args=(params,)).flatten()

def objective_function(params, t_obs, C_measured):
    C_pred = model_prediction(params, t_obs)
    return np.sum((C_pred - C_measured) ** 2)

# Step 6: Optimize parameters using least squares fit
initial_guess = [0.1, 0.3, 2, 1.2, 0.3, 2, 1.2]
result = minimize(objective_function, initial_guess, args=(timepoints, C_obs_noisy), method='Nelder-Mead')
fitted_params = result.x

# Step 7: Simulate model with fitted parameters
C_fit = odeint(cancer_growth, C0, t_full, args=(fitted_params,))

# Step 8: Plot results
plt.figure(figsize=(10, 6))
plt.plot(t_full, C_fit, label="Fitted Model", color="blue")
plt.scatter(timepoints, C_obs_noisy, label="Observed Data", color="red")
plt.plot(t_full, C_simulated, '--', label="True Model", color="gray")
plt.xlabel("Time (days)")
plt.ylabel("Cancer Cell Count")
plt.title("Cancer Cell Dynamics under M1/M2 Macrophage Influence")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 9: Print fitted parameters
param_labels = ["r_basal", "Rmax_M1", "k1", "θ1", "Rmax_M2", "k2", "θ2"]
print("\nFitted Parameters:")
for name, val in zip(param_labels, fitted_params):
    print(f"{name:10s} = {val:.4f}")
