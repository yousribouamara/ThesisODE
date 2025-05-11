import pandas as pd
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit


"""
This script quantifies the effect of contact between CM alone and Partial/Full coculture
on CC fold increase. This is done in terms of estimated growth rates.
"""


# --- Set path to CSV files ---
data_path = "/Users/yousribouamara/Downloads/data varady"  # Change as needed

# --- Files and variable-style keys ---
files = {
    "rM1_aloneCM": "M1aloneCM.csv",
    "rM2_aloneCM": "M2aloneCM.csv",
    "rM1_CM":      "231M1CM.csv",
    "rM2_CM":      "231M2CM.csv",
    "rM1_CC":      "231M1CC.csv",
    "rM2_CC":      "231M2CC.csv",
}

# --- ODE model for exponential growth ---
def exp_model(C, t, r):
    return r * C

def simulate(t, r, C0=1.0):
    sol = odeint(exp_model, C0, t, args=(r,))
    return sol.ravel()

# --- Estimate growth rates from files ---
growth_rates = {}
print("\n📈 Estimated Growth Rates:")
print("-" * 35)

for varname, filename in files.items():
    df = pd.read_csv(f"{data_path}/{filename}", header=None, names=["Time", "FoldIncrease"])
    t_data = df["Time"].values
    y_data = df["FoldIncrease"].values

    if (y_data <= 0).any():
        print(f"⚠️ Skipping {varname} due to zero or negative values.")
        continue

    try:
        params, _ = curve_fit(lambda t, r: simulate(t, r), t_data, y_data, p0=[0.05])
        growth_rates[varname] = params[0]
        print(f"{varname:15s}: r = {params[0]:.4f} per hour")
    except RuntimeError:
        print(f"❌ Fit failed for {varname}")

# --- Quantify effects ---
delta_contact_M1     = growth_rates["rM1_CC"] - growth_rates["rM1_aloneCM"]
delta_contact_M2     = growth_rates["rM2_CC"] - growth_rates["rM2_aloneCM"]
delta_partialCM_M1   = growth_rates["rM1_CM"] - growth_rates["rM1_aloneCM"]
delta_partialCM_M2   = growth_rates["rM2_CM"] - growth_rates["rM2_aloneCM"]

# --- Print summary ---
print("\n📊 Quantified Effects (units: 1/hour)")
print("-" * 40)
print(f"Contact effect (M1):       Δ = {delta_contact_M1:+.4f}")
print(f"Contact effect (M2):       Δ = {delta_contact_M2:+.4f}")
print()
print(f"Partial-CM effect (M1):    Δ = {delta_partialCM_M1:+.4f}")
print(f"Partial-CM effect (M2):    Δ = {delta_partialCM_M2:+.4f}")

#or store results in a dictionary ---
effects = {
    "delta_contact_M1": delta_contact_M1,
    "delta_contact_M2": delta_contact_M2,
    "delta_partialCM_M1": delta_partialCM_M1,
    "delta_partialCM_M2": delta_partialCM_M2,
}
