import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.integrate import odeint
from scipy.optimize import curve_fit

# --- Set path to CSV files ---
data_path = "/Users/yousribouamara/Downloads/data varady"  # Change to your local path

# --- Define all files ---
files = {
    "231 alone":     "231alone_2.csv",
    "231 + M1 CM":   "231M1CM.csv",
    "231 + M1 CC":   "231M1CC.csv",
    "231 + M2 CM":   "231M2CM.csv",
    "231 + M2 CC":   "231M2CC.csv",
    "M1 alone CM":   "M1aloneCM.csv",
    "M2 alone CM":   "M2aloneCM.csv",
}

# --- Map labels to variable-style keys ---
label_to_var = {
    "231 alone":     "r0",
    "M1 alone CM":   "rM1_aloneCM",
    "M2 alone CM":   "rM2_aloneCM",
    "231 + M1 CM":   "rM1_CM",
    "231 + M2 CM":   "rM2_CM",
    "231 + M1 CC":   "rM1_CC",
    "231 + M2 CC":   "rM2_CC",
}

# --- ODE model and simulation wrapper ---
def exp_model(C, t, r):
    return r * C

def simulate(t, r, C0=1.0):
    sol = odeint(exp_model, C0, t, args=(r,))
    return sol.ravel()

# --- Estimate growth rates using curve fitting ---
growth_rates = {}

print("\n📈 Estimated Growth Rates:")
print("-" * 35)

for label, filename in files.items():
    df = pd.read_csv(os.path.join(data_path, filename), header=None, names=["Time", "FoldIncrease"])
    t_data = df["Time"].values
    y_data = df["FoldIncrease"].values

    if (y_data <= 0).any():
        print(f"⚠️ Skipping {label} due to zero or negative values.")
        continue

    try:
        params, _ = curve_fit(lambda t, r: simulate(t, r), t_data, y_data, p0=[0.05])
        r_fit = params[0]
        varname = label_to_var[label]
        growth_rates[varname] = r_fit
        print(f"{varname:15s}: r = {r_fit:.4f} per hour")
    except RuntimeError:
        print(f"❌ Fit failed for {label}")

# --- Adjustable M1/M2 weighting (20% M1, 80% M2 for now) ---
wM1 = 0.2
wM2 = 0.8

# --- Time range for simulation ---
t_sim = np.linspace(0, 72, 100)
C0 = 1.0

# --- Category-based simulation structure ---
categories = {
    "CM only (M1/M2 alone)": [("r0", "rM1_aloneCM", "rM2_aloneCM")],
    "Partial Co-culture (231 + Mφ CM)": [("r0", "rM1_CM", "rM2_CM")],
    "Full Co-culture (231 + Mφ CC)": [("r0", "rM1_CC", "rM2_CC")],
}

# --- Plot each category separately ---
for category, keys in categories.items():
    plt.figure(figsize=(10, 6))
    for (r0_key, rM1_key, rM2_key) in keys:
        if all(k in growth_rates for k in [r0_key, rM1_key, rM2_key]):
            r0 = growth_rates[r0_key]
            rM1 = r0 - growth_rates[rM1_key]
            rM2 = r0 - growth_rates[rM2_key]
            r_weighted = r0 - (wM1 * rM1 + wM2 * rM2)

            C_sim = odeint(exp_model, C0, t_sim, args=(r_weighted,)).ravel()
            plt.plot(t_sim, C_sim,
                     label=f"Simulated: {int(wM1*100)}% M1 + {int(wM2*100)}% M2",
                     color="black", linewidth=2, linestyle='--')

    # Plot original data curves for each category
    for label in files:
        show = False
        if category == "CM only (M1/M2 alone)" and label in ["231 alone", "M1 alone CM", "M2 alone CM"]:
            show = True
        elif category == "Partial Co-culture (231 + Mφ CM)" and label in ["231 alone", "231 + M1 CM", "231 + M2 CM"]:
            show = True
        elif category == "Full Co-culture (231 + Mφ CC)" and label in ["231 alone", "231 + M1 CC", "231 + M2 CC"]:
            show = True

        if show:
            df = pd.read_csv(os.path.join(data_path, files[label]), header=None, names=["Time", "FoldIncrease"])
            plt.plot(df["Time"], df["FoldIncrease"], label=label)

    plt.xlabel("Time (hours)")
    plt.ylabel("Normalized Fold Increase (ϕ)")
    plt.title(f"Cancer Cell Growth: {category}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
