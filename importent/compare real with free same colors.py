from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# GLOBAL PLOT STYLE (LARGE LABELS + LEGEND)
# ============================================================
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "figure.titlesize": 24,
})

# ============================================================
# FILE PATHS
# ============================================================
MEAS_CSV = Path(r"C:\Users\User\Desktop\2D Beamprofiler background substracted\beam_size_vs_position_plot_data.csv")
THEORY_CSV = Path(r"C:\Users\User\Desktop\Theory free propagation\beam_size_vs_z_plot_data.csv")
OUTPUT_DIR = Path(r"C:\Users\User\Desktop\comparison_plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THEORY_MODE = "with_phase"

METRICS = [
    ("D4sigma_avg_mm", "D4σ", "blue"),
    ("D_EE50_mm", "EE50", "orange"),
    ("D_EE86_mm", "EE86", "green"),
    ("D_area_50pct_mm", "Area50%", "red"),
    ("D_area_13p5pct_mm", "Area13.5%", "purple"),
    ("FWHM_mm", "FWHM", "brown"),
]


def main():
    df_meas = pd.read_csv(MEAS_CSV).sort_values("position_m")
    df_theory = pd.read_csv(THEORY_CSV).sort_values("z_m")

    diff_rows = []

    # ========================================================
    # PLOT 1: OVERLAY (Theory + Measurement)
    # ========================================================
    plt.figure(figsize=(14, 9))

    for metric, label, color in METRICS:
        theory_col = f"{metric}_{THEORY_MODE}"

        if metric not in df_meas.columns or theory_col not in df_theory.columns:
            print(f"Skipping: {metric} or {theory_col} not found.")
            continue

        z_meas = df_meas["position_m"].to_numpy()
        meas_vals = df_meas[metric].to_numpy()

        z_theory = df_theory["z_m"].to_numpy()
        theory_vals = df_theory[theory_col].to_numpy()

        # Interpolation
        theory_interp = np.interp(z_meas, z_theory, theory_vals)

        # Store difference
        diff = meas_vals - theory_interp
        for i in range(len(z_meas)):
            diff_rows.append({
                "metric": metric,
                "z_m": z_meas[i],
                "measurement_mm": meas_vals[i],
                "theory_mm": theory_interp[i],
                "difference_mm": diff[i],
            })

        # Theory (dashed)
        plt.plot(
            z_theory,
            theory_vals,
            linestyle="--",
            linewidth=2.5,
            color=color,
            label=f"{label} Theory",
        )

        # Measurement (solid)
        plt.plot(
            z_meas,
            meas_vals,
            linestyle="-",
            linewidth=2.5,
            marker="o",
            markersize=6,
            color=color,
            label=f"{label} Measurement",
        )

    plt.xlabel("Position / z [m]")
    plt.ylabel("Beam size [mm]")
    plt.title("Comparison: Theory and Measurement (Overlay)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plot_overlay.png", dpi=200)
    plt.close()

    # ========================================================
    # PLOT 2: MEASUREMENT ONLY
    # ========================================================
    plt.figure(figsize=(14, 9))

    for metric, label, color in METRICS:
        if metric not in df_meas.columns:
            continue

        plt.plot(
            df_meas["position_m"],
            df_meas[metric],
            linestyle="-",
            linewidth=2.5,
            marker="o",
            markersize=6,
            color=color,
            label=label,
        )

    plt.xlabel("Position / z [m]")
    plt.ylabel("Beam size [mm]")
    plt.title("Measurement (all metrics)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plot_measurement_only.png", dpi=200)
    plt.close()

    # ========================================================
    # PLOT 3: THEORY ONLY
    # ========================================================
    plt.figure(figsize=(14, 9))

    for metric, label, color in METRICS:
        theory_col = f"{metric}_{THEORY_MODE}"

        if theory_col not in df_theory.columns:
            continue

        plt.plot(
            df_theory["z_m"],
            df_theory[theory_col],
            linestyle="--",
            linewidth=2.5,
            color=color,
            label=label,
        )

    plt.xlabel("Position / z [m]")
    plt.ylabel("Beam size [mm]")
    plt.title("Theory (all metrics)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plot_theory_only.png", dpi=200)
    plt.close()

    # ========================================================
    # PLOT 4: DIFFERENCE (Measurement - Theory)
    # ========================================================
    plt.figure(figsize=(14, 8))

    for metric, label, color in METRICS:
        rows = [r for r in diff_rows if r["metric"] == metric]
        if not rows:
            continue

        z = [r["z_m"] for r in rows]
        diff = [r["difference_mm"] for r in rows]

        plt.plot(
            z,
            diff,
            marker="o",
            linestyle="-",
            linewidth=2.5,
            markersize=6,
            color=color,
            label=f"{label} (Meas - Theory)",
        )

    plt.axhline(0, color="black", linestyle="--", linewidth=1.5)
    plt.xlabel("Position / z [m]")
    plt.ylabel("Difference [mm]")
    plt.title("Difference: Measurement − Theory")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plot_difference.png", dpi=200)
    plt.close()

    # ========================================================
    # SAVE CSV
    # ========================================================
    pd.DataFrame(diff_rows).to_csv(OUTPUT_DIR / "difference_data.csv", index=False)

    print("Done.")
    print("All plots saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()