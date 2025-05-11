import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


'''
This script only predicts further evolutions of existing data in their own categories,
 not what would happen in combined systems
'''
# Logistic influence functions
def r_M1(A, Rmax, k, theta):
    return -Rmax / (1 + np.exp(-k * (A - theta)))

def r_M2(A, Rmax, k, theta):
    return Rmax / (1 + np.exp(-k * (A - theta)))

# Cancer growth ODE
def dCdt(C, t, r_basal, A_M1, A_M2, params):
    Rmax1, k1, theta1, Rmax2, k2, theta2 = params
    return (r_basal + r_M1(A_M1, Rmax1, k1, theta1) + r_M2(A_M2, Rmax2, k2, theta2)) * C

# Best fit parameters (from model calibration)
best_fit = [0.419, 0.598, 5.312, 0.412, 0.394, 5.612, 0.528]

# Experimental conditions and macrophage activity scores
conditions = [
    "231 alone", "231 + M1", "231 + M2",
    "M1 alone CM", "M2 alone CM",
    "231 + M1 CM", "231 + M1 CC", "231 + M2 CC"
]

activity_map = {
    "231 alone":      (0.0, 0.0),
    "231 + M1":       (0.7, 0.0),
    "231 + M2":       (0.0, 0.7),
    "M1 alone CM":    (0.5, 0.0),
    "M2 alone CM":    (0.0, 0.5),
    "231 + M1 CM":    (0.6, 0.0),
    "231 + M1 CC":    (0.9, 0.0),
    "231 + M2 CC":    (0.0, 0.9)
}

# Simulate growth C(t) over 72 hours
t_range = np.linspace(0, 72, 300)
C0 = 1.0
time_courses = {}

for cond in conditions:
    A_M1, A_M2 = activity_map[cond]
    C_t = odeint(dCdt, C0, t_range, args=(best_fit[0], A_M1, A_M2, best_fit[1:])).flatten()
    time_courses[cond] = C_t

# Plot all C(t) curves
plt.figure(figsize=(12, 7))
for cond, C_t in time_courses.items():
    plt.plot(t_range, C_t, label=cond)

plt.xlabel("Time (hours)")
plt.ylabel("C(t): Simulated Confluence / Growth")
plt.title("Cancer Cell Growth C(t) Under Macrophage Influence")
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.grid(True)
plt.tight_layout()
plt.show()
