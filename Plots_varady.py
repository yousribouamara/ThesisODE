import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.integrate import odeint

## Set path to CSV files
data_path = "/Users/yousribouamara/Downloads/data varady"

# File names and labels
files = {
    "231 alone":    "231alone_2.csv",
    "231 + M1 CM":  "231M1CM.csv",
    "231 + M1 CC":  "231M1CC.csv",
    "231 + M2 CM":  "231M2CM.csv",
    "231 + M2 CC":  "231M2CC.csv",
    "M1 alone CM":  "M1aloneCM.csv",
    "M2 alone CM":  "M2aloneCM.csv"
}


# Define markers and colors
markers = ['o', 's', '^', 'D', 'v', '>', '<']
colors  = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']

plt.figure(figsize=(12, 7))

for (label, filename), marker, color in zip(files.items(), markers, colors):
    # read with no header so the first row (0,1.0…) becomes data
    df = pd.read_csv(
        os.path.join(data_path, filename),
        header=None,
        names=["Time", "Confluence"]
    )
    x = df["Time"]
    y = df["Confluence"]
    plt.plot(x, y,
             label=label,
             marker=marker,
             color=color,
             linewidth=2,
             markersize=6)

plt.xlabel("Time (hours)")
plt.ylabel("Normalized confluence")
plt.title("Cancer cell growth in time with different Macrophage Conditions")
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.grid(True)
plt.tight_layout()
plt.show()

