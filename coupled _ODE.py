
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


"""Let's assume three cell types BCC, M1 and M2. We model BCC proliferation, and since macrophages
 won't  proliferate, we model their activity, based on important cytokines. 
 This gives us a coupled system of three equations

Variables:
BCC - Breast cancer cell population
A_M1 - M1 macrophage activity (modeled via TNF-α)
A_M2 - M2 macrophage activity (modeled via IL-10)
alpha_BCC - Baseline BCC proliferation rate
gamma_M1 - Suppression strength of BCC by M1 (treated as a constant)
gamma_M2 - Stimulation strength of BCC by M2 (treated as a constant)
alpha_M1 - Baseline TNF-α secretion rate
beta_BCC - Suppression strength of M1 by BCC
K_M1 - Maximum possible M1 activity
alpha_M2 - Baseline IL-10 secretion rate 
beta_BCC_M2 - Activation of M2 by BCC
K_M2 - Maximum possible M2 activity"""


def system(y, t, alpha_BCC, gamma_M1, gamma_M2, alpha_M1, beta_BCC, K_M1, alpha_M2, beta_BCC_M2, K_M2):
    BCC, A_M1, A_M2 = y  # Unpack variables

    # Equation 1: BCC Growth
    dBCC_dt = alpha_BCC * BCC - gamma_M1 * A_M1 * BCC + gamma_M2 * A_M2 * BCC

    # Equation 2: M1 Activity (TNF-α Secretion)
    dA_M1_dt = alpha_M1 * (1 - A_M1 / K_M1) - beta_BCC * BCC * A_M1

    # Equation 3: M2 Activity (IL-10 Secretion)
    dA_M2_dt = alpha_M2 * (1 - A_M2 / K_M2) + beta_BCC_M2 * BCC * (1 - A_M2 / K_M2)

    return [dBCC_dt, dA_M1_dt, dA_M2_dt]


# Initial conditions
BCC_0 = 100  # Initial BCC population
A_M1_0 = 10  # Initial M1 activity
A_M2_0 = 5  # Initial M2 activity

y0 = [BCC_0, A_M1_0, A_M2_0]

# Time points
t = np.linspace(0, 100, 1000)

# Parameter values (placeholders, should be calibrated with real data)
params = (0.3, 0.02, 0.01, 0.05, 0.01, 50, 0.02, 0.02, 40)

# Solve ODE system
solution = odeint(system, y0, t, args=params)
BCC_sol, A_M1_sol, A_M2_sol = solution.T

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(t, BCC_sol, label='BCC (Cancer Cells)', color='red')
plt.plot(t, A_M1_sol, label='M1 Activity (TNF-α)', color='blue')
plt.plot(t, A_M2_sol, label='M2 Activity (IL-10)', color='green')
plt.xlabel('Time (days)')
plt.ylabel('Cell Count / Activity Level')
plt.title('Coupled System of BCC, M1, and M2')
plt.legend()
plt.grid()
plt.show()
