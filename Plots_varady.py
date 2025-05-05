import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.integrate import odeint

# Set path to CSV files
data_path = "/Users/yousribouamara/Downloads/data varady"


# File names and labels
files = {
    "231 alone": "231alone2X.csv",
    "231 + M1 CM": "231coM1_CM.csv",
    "231 + M1 CC": "231coM1.csv",
    "231 + M2 CM": "231coM2_CM.csv",
    "231 + M2 CC": "231coM2.csv",
    "M1 alone CM": "M1aloneCM.csv",
    "M2 alone CM": "M2aloneCM.csv"
}

# Define markers and colors
markers = ['o', 's', '^', 'D', 'v', '>', '<']
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']

# Create the plot
plt.figure(figsize=(12, 7))

for (label, filename), marker, color in zip(files.items(), markers, colors):
    df = pd.read_csv(os.path.join(data_path, filename))
    x = df.iloc[:, 0]
    y = df.iloc[:, 1]
    plt.plot(x, y, label=label, marker=marker, color=color, linewidth=2, markersize=6)

plt.xlabel("Time (hours)")
plt.ylabel("Normalized Confluence")
plt.title("Cancer Cell Growth Over Time Under Various Macrophage Conditions")
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.grid(True)
plt.tight_layout()
plt.show()

# Dictionary to store rates
growth_rates = {}

# Clean label names to variable-safe format
def label_to_varname(label):
    return "rate_" + label.lower().replace(" ", "_").replace("+", "plus").replace("/", "_").replace(",", "")

print("Estimated average growth rates (ΔC/Δt):")

for (label, filename) in files.items():
    df = pd.read_csv(os.path.join(data_path, filename))
    x = df.iloc[:, 0].values.reshape(-1, 1)  # Time
    y = df.iloc[:, 1].values  # Confluence

    model = LinearRegression().fit(x, y)
    slope = model.coef_[0]

    varname = label_to_varname(label)
    growth_rates[varname] = slope

    print(f"{varname} = {slope:.4f}")

# We now have variables like:
# rate_231_alone, rate_231_plus_m1_cm, etc.
print("\nAccess a rate:")
print("M2 CC growth rate:", growth_rates["rate_231_plus_m2_cc"])


# Now let's get the growth rates by subtracting the basal rate (231 alone) from the combined ones.
# Define r values
r_basal = growth_rates["rate_231_alone"]
r_M1_CM = growth_rates["rate_231_plus_m1_cm"] - r_basal
r_M1_CC = growth_rates["rate_231_plus_m1_cc"] - r_basal
r_M2_CM = growth_rates["rate_231_plus_m2_cm"] - r_basal
r_M2_CC = growth_rates["rate_231_plus_m2_cc"] - r_basal

# Build a dict of total r for each condition
rates = {
    "231 alone": r_basal,
    "231 + M1 CM": r_basal + r_M1_CM,
    "231 + M1 CC": r_basal + r_M1_CC,
    "231 + M2 CM": r_basal + r_M2_CM,
    "231 + M2 CC": r_basal + r_M2_CC
}

# ODE model: simple exponential growth
def dCdt(C, t, r):
    return r * C

# Simulation time
t = np.linspace(0, 72, 200)
C0 = 1.0  # normalized starting confluence

# Plotting
plt.figure(figsize=(12, 7))

for label, r in rates.items():
    C = odeint(dCdt, C0, t, args=(r,)).flatten()
    plt.plot(t, C, label=label)

plt.xlabel("Time (hours)")
plt.ylabel("Simulated Confluence C(t)")
plt.title("ODE Simulation of Cancer Cell Growth Under Macrophage Influence")
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.grid(True)
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Define r values (use your actual calculated values here)
r_basal = growth_rates["rate_231_alone"]
r_M1_CM = growth_rates["rate_231_plus_m1_cm"] - r_basal
r_M1_CC = growth_rates["rate_231_plus_m1_cc"] - r_basal
r_M2_CM = growth_rates["rate_231_plus_m2_cm"] - r_basal
r_M2_CC = growth_rates["rate_231_plus_m2_cc"] - r_basal

# Build a dict of total r for each condition
rates = {
    "231 alone": r_basal,
    "231 + M1 CM": r_basal + r_M1_CM,
    "231 + M1 CC": r_basal + r_M1_CC,
    "231 + M2 CM": r_basal + r_M2_CM,
    "231 + M2 CC": r_basal + r_M2_CC
}

# ODE model: simple exponential growth
def dCdt(C, t, r):
    return r * C

# Simulation time
t = np.linspace(0, 72, 200)
C0 = 1.0  # normalized starting confluence

# Plotting
plt.figure(figsize=(12, 7))

for label, r in rates.items():
    C = odeint(dCdt, C0, t, args=(r,)).flatten()
    plt.plot(t, C, label=label)

plt.xlabel("Time (hours)")
plt.ylabel("Simulated Confluence C(t)")
plt.title("ODE Simulation of Cancer Cell Growth Under Macrophage Influence")
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.grid(True)
plt.tight_layout()
plt.show()
