import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "legend.fontsize": 16,
})

# ======== USER INPUT ========
THEORY_CSV = Path(r"C:\Users\User\Desktop\Free beam size plot real\plot.csv")
REAL_CSV   = Path(r"C:\Users\User\Desktop\Free beam size plot\real_propagation_gaussfit.csv")

# ======== LOAD DATA ========
theory = pd.read_csv(THEORY_CSV)
real = pd.read_csv(REAL_CSV)

# ======== EXTRACT ========

# THEORY
z_theory = theory["z_m"]
mean_theory = theory["Gauss_diameter_mean_1e2_mm"]
err_theory = theory["Gauss_diameter_error_1e2_mm"]

# REAL (angepasst!)
z_real = real["z_m"]
mean_real = real["Gauss_diameter_1e2_px"]

# ======== PLOT ========
plt.figure(figsize=(10,6))

plt.errorbar(
    z_theory,
    mean_theory,
    yerr=err_theory,
    fmt="o-",
    capsize=4,
    label="Theory (Gaussian fit 1/e²)",
)

plt.plot(
    z_real,
    mean_real,
    "s-",
    linewidth=2,
    markersize=8,
    label="Measurement (Gaussian fit 1/e²)"
)

plt.xlabel("Propagation distance z [m]")
plt.ylabel("Beam diameter (1/e²) [pixels]")
plt.title("Comparison: Theory vs Measurement (Gaussian-fit beam size)")

plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig("comparison_theory_vs_real.png", dpi=300)
plt.show()