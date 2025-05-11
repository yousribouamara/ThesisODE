import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# Logistic function
def logistic(t, r, K, C0):
    return K / (1 + (K - C0) / C0 * np.exp(-r * t))

# Data path on your PC
data_path = "/Users/yousribouamara/Downloads/data varady"

# File names and conditions
files = {
    "231 alone": "231alone2X.csv",
    "231 + M1 CM": "231coM1_CM.csv",
    "231 + M1 CC": "231coM1.csv",
    "231 + M2 CM": "231coM2_CM.csv",
    "231 + M2 CC": "231coM2.csv",
    "M1 alone CM": "M1aloneCM.csv",
    "M2 alone CM": "M2aloneCM.csv"
}

# Simulate over this time range
t_sim = np.linspace(0, 72, 300)

# Store fitted parameters
fit_results = {}

# Plot
plt.figure(figsize=(12, 7))

for label, filename in files.items():
    df = pd.read_csv(os.path.join(data_path, filename))
    t = df.iloc[:, 0].values
    C = df.iloc[:, 1].values

    # Initial guess: r, K, C0
    p0 = [0.05, max(C), C[0]]
    try:
        popt, _ = curve_fit(logistic, t, C, p0=p0, maxfev=10000)
        r_fit, K_fit, C0_fit = popt
        fit_results[label] = {"r": r_fit, "K": K_fit, "C0": C0_fit}
        C_fit = logistic(t_sim, *popt)
        plt.plot(t_sim, C_fit, label=f"{label}")
    except RuntimeError:
        print(f"⚠️ Fit failed for: {label}")

# Show results
plt.xlabel("Time (hours)")
plt.ylabel("Confluence C(t)")
plt.title("Logistic Fit of Cancer Cell Growth")
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.grid(True)
plt.tight_layout()
plt.show()

# Print fitted parameters
print("\nFitted Logistic Parameters:")
for label, params in fit_results.items():
    print(f"{label}: r = {params['r']:.4f}, K = {params['K']:.4f}, C0 = {params['C0']:.4f}")
