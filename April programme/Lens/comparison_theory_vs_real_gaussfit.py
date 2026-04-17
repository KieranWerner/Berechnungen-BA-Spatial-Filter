import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    "font.size": 16,
})

# ======== USER INPUT ========
THEORY_CSV = Path(r"C:\Users\User\Desktop\999 fertige Dateien\theory_lens_propagation_simple_gaussfit.csv")
REAL_CSV   = Path(r"C:\Users\User\Desktop\999 fertige Dateien\beam_propagation_mean.csv")

# ======== LOAD DATA ========
theory = pd.read_csv(THEORY_CSV)
real = pd.read_csv(REAL_CSV)

# ======== EXTRACT ========
# THEORY
z_theory = theory["z_m"]
mean_theory = theory["Gauss_diameter_mean_1e2_mm"]
err_theory = theory["Gauss_diameter_error_1e2_mm"]

# REAL
z_real = real["z_cm"] / 100.0
mean_real = real["diameter_mean_mm"]
err_real = real["error_mm"]

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

plt.errorbar(
    z_real,
    mean_real,
    yerr=err_real,
    fmt="s-",
    capsize=4,
    label="Measurement (Gaussian fit 1/e²)",
)

plt.xlabel("Propagation distance z [m]")
plt.ylabel("Beam size (1/e² diameter) [mm]")
plt.title("Comparison: Theory vs Measurement (Gaussian-fit beam size)")
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig("comparison_theory_vs_real.png", dpi=200)
plt.show()
