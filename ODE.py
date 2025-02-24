"""
Thesis project for cancer cell growth: BCC cells given by growth, attack from M1 and support form M2
"""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


#BCC equation
def bcc_equation(BCC, t, alpha_BCC, gamma_M1, beta_M2, M1, M2):
    # Differential equation for Breast Cancer Cells
    dBCC_dt = alpha_BCC * BCC - gamma_M1 * M1 * BCC + beta_M2 * M2 * BCC
    return dBCC_dt


# Initial Conditions
BCC_0 = 100  #Initial number of breast cancer cells, have to count?

# Time Vector (from 0 to 100 days)
time = np.linspace(0, 28, 1000)

# Parameter Values (adjust later))
alpha_BCC = 0.369  #Cancer cell growth rate
gamma_M1 = 0.03  #Killing rate by M1 macrophages
beta_M2 = 0.02  #Growth promotion by M2 macrophages

#M1 and M2 Macrophages
M1_values = [10, 20, 30]  # Different levels of M1 macrophages
M2_values = [5, 10, 15]  # Different levels of M2 macrophages

# Plot results for different M1 and M2 scenarios
plt.figure(figsize=(12, 8))

for M1, M2 in zip(M1_values, M2_values):
    # Solve ODE
    BCC = odeint(bcc_equation, BCC_0, time, args=(alpha_BCC, gamma_M1, beta_M2, M1, M2))
    BCC = BCC.flatten()  #Flatten the array for easier plotting

    #Plot
    plt.plot(time, BCC, label=f'M1={M1}, M2={M2}')

# Plot settings
plt.title('Simulation of Breast Cancer Cell Dynamics')
plt.xlabel('Time (days)')
plt.ylabel('Number of Cancer Cells')
plt.legend()
plt.grid(True)
plt.show()
