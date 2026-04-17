from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# FILE PATHS
# ============================================================
MEAS_CSV = Path(r"C:\Users\User\Desktop\2D Beamprofiler background substracted\beam_size_vs_position_plot_data.csv")
THEORY_CSV = Path(r"C:\Users\User\Desktop\Theory free propagation\beam_size_vs_z_plot_data.csv")

# ============================================================
# SETTINGS
# ============================================================
MEAS_METRIC = "D4sigma_avg_mm"
THEORY_METRIC = "D4sigma_avg_mm_with_phase"

# ============================================================
# LOAD DATA
# ============================================================
df_meas = pd.read_csv(MEAS_CSV).sort_values("position_m")
df_theory = pd.read_csv(THEORY_CSV).sort_values("z_m")

# ============================================================
# EXTRACT CURVES
# ============================================================
z_meas = df_meas["position_m"].to_numpy()
meas_vals = df_meas[MEAS_METRIC].to_numpy()

z_theory = df_theory["z_m"].to_numpy()
theory_vals = df_theory[THEORY_METRIC].to_numpy()

# ============================================================
# INTERPOLATE THEORY TO MEASUREMENT POSITIONS
# ============================================================
theory_interp = np.interp(z_meas, z_theory, theory_vals)

# ============================================================
# ERROR METRICS
# ============================================================
diff = meas_vals - theory_interp

mae = np.mean(np.abs(diff))
rmse = np.sqrt(np.mean(diff**2))

# ============================================================
# OUTPUT
# ============================================================
print("Simple comparison without optimization")
print(f"Measurement metric: {MEAS_METRIC}")
print(f"Theory metric:      {THEORY_METRIC}")
print()
print(f"MAE  = {mae:.6f} mm")
print(f"RMSE = {rmse:.6f} mm")