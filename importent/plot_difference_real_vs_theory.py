#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vergleicht zwei bereits berechnete z-Verläufe und plottet die Differenz:

Differenz = Messung - Theorie

Gedacht für genau diese zwei Dateien:
1) Real lens propagation:
   beam_size_vs_z.csv
2) Theory lens 5m:
   theory_lens_5m_beam_size_vs_z.csv

Einfach in VS Code öffnen und Run drücken.
Keine Kommandozeilen-Argumente nötig.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# HIER EINSTELLEN
# ============================================================

REAL_CSV = Path(r"C:\Users\User\Desktop\Real lens propagation\beam_size_vs_z.csv")
THEORY_CSV = Path(r"C:\Users\User\Desktop\Theory_lens_5m\theory_lens_5m_beam_size_vs_z.csv")

OUTPUT_DIR = Path(r"C:\Users\User\Desktop\Difference_real_minus_theory")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Welche Metriken vergleichen?
PLOT_D4SIGMA = True
PLOT_EE50 = True
PLOT_EE86 = True
PLOT_AREA_50 = True
PLOT_AREA_13P5 = True
PLOT_FWHM = True

# Falls True: Theorie wird auf die z-Punkte der Messung interpoliert
INTERPOLATE_THEORY_TO_REAL_Z = True


# ============================================================
# HILFSFUNKTIONEN
# ============================================================

def metric_config():
    return [
        ("D4sigma_avg_mm", "D4σ"),
        ("D_EE50_mm", "EE50"),
        ("D_EE86_mm", "EE86"),
        ("D_area_50pct_mm", "Area50%"),
        ("D_area_13p5pct_mm", "Area13.5%"),
        ("FWHM_mm", "FWHM"),
    ]


def metric_enabled(metric_name: str) -> bool:
    enabled = {
        "D4sigma_avg_mm": PLOT_D4SIGMA,
        "D_EE50_mm": PLOT_EE50,
        "D_EE86_mm": PLOT_EE86,
        "D_area_50pct_mm": PLOT_AREA_50,
        "D_area_13p5pct_mm": PLOT_AREA_13P5,
        "FWHM_mm": PLOT_FWHM,
    }
    return enabled.get(metric_name, False)


def prepare_real_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # z-Spalte vereinheitlichen
    if "z_m" in df.columns:
        pass
    elif "z_cm" in df.columns:
        df["z_m"] = df["z_cm"] / 100.0
    else:
        raise ValueError("REAL_CSV braucht 'z_m' oder 'z_cm'.")

    return df.sort_values("z_m").reset_index(drop=True)


def prepare_theory_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "z_m" not in df.columns:
        raise ValueError("THEORY_CSV braucht die Spalte 'z_m'.")
    return df.sort_values("z_m").reset_index(drop=True)


def calculate_error_stats(diff_values_mm: np.ndarray) -> dict:
    diff_values_mm = np.asarray(diff_values_mm, dtype=float)
    diff_values_mm = diff_values_mm[np.isfinite(diff_values_mm)]

    if diff_values_mm.size == 0:
        return {
            "mean_signed_diff_mm": np.nan,
            "mae_mm": np.nan,
            "rmse_mm": np.nan,
            "max_abs_diff_mm": np.nan,
            "n_points": 0,
        }

    return {
        "mean_signed_diff_mm": float(np.mean(diff_values_mm)),
        "mae_mm": float(np.mean(np.abs(diff_values_mm))),
        "rmse_mm": float(np.sqrt(np.mean(diff_values_mm ** 2))),
        "max_abs_diff_mm": float(np.max(np.abs(diff_values_mm))),
        "n_points": int(diff_values_mm.size),
    }


# ============================================================
# HAUPTPROGRAMM
# ============================================================

def main() -> None:
    if not REAL_CSV.exists():
        raise SystemExit(f"REAL_CSV nicht gefunden:\n{REAL_CSV}")

    if not THEORY_CSV.exists():
        raise SystemExit(f"THEORY_CSV nicht gefunden:\n{THEORY_CSV}")

    df_real = prepare_real_df(pd.read_csv(REAL_CSV))
    df_theory = prepare_theory_df(pd.read_csv(THEORY_CSV))

    active_metrics = [(m, label) for m, label in metric_config() if metric_enabled(m)]
    if not active_metrics:
        raise SystemExit("Keine Metrik zum Plotten aktiviert.")

    z_real = df_real["z_m"].to_numpy(dtype=float)
    z_theory = df_theory["z_m"].to_numpy(dtype=float)

    diff_rows = []
    summary_rows = []

    # --------------------------------------------------------
    # Vergleich pro Metrik
    # --------------------------------------------------------
    for metric, label in active_metrics:
        if metric not in df_real.columns:
            print(f"Übersprungen (fehlt in Real): {metric}")
            continue
        if metric not in df_theory.columns:
            print(f"Übersprungen (fehlt in Theorie): {metric}")
            continue

        y_real = df_real[metric].to_numpy(dtype=float)
        y_theory = df_theory[metric].to_numpy(dtype=float)

        if INTERPOLATE_THEORY_TO_REAL_Z:
            y_theory_on_real = np.interp(z_real, z_theory, y_theory)
            z_use = z_real
            y_real_use = y_real
            y_theory_use = y_theory_on_real
        else:
            z_common = np.intersect1d(z_real, z_theory)
            if z_common.size == 0:
                print(f"Keine gemeinsamen z-Werte für {metric}")
                continue

            real_map = {float(z): float(v) for z, v in zip(z_real, y_real)}
            theory_map = {float(z): float(v) for z, v in zip(z_theory, y_theory)}

            z_use = z_common
            y_real_use = np.array([real_map[float(z)] for z in z_common], dtype=float)
            y_theory_use = np.array([theory_map[float(z)] for z in z_common], dtype=float)

        diff = y_real_use - y_theory_use

        for z, r, t, d in zip(z_use, y_real_use, y_theory_use, diff):
            diff_rows.append({
                "metric": metric,
                "metric_label": label,
                "z_m": float(z),
                "real_mm": float(r),
                "theory_mm": float(t),
                "difference_mm": float(d),
            })

        stats = calculate_error_stats(diff)
        summary_rows.append({
            "metric": metric,
            "metric_label": label,
            **stats,
        })

    if not diff_rows:
        raise SystemExit("Es konnten keine Vergleichsdaten erzeugt werden.")

    df_diff = pd.DataFrame(diff_rows)
    df_summary = pd.DataFrame(summary_rows)

    # --------------------------------------------------------
    # CSV + JSON speichern
    # --------------------------------------------------------
    diff_csv = OUTPUT_DIR / "difference_real_minus_theory.csv"
    diff_json = OUTPUT_DIR / "difference_real_minus_theory.json"
    summary_csv = OUTPUT_DIR / "difference_summary.csv"
    summary_json = OUTPUT_DIR / "difference_summary.json"

    df_diff.to_csv(diff_csv, index=False)
    df_summary.to_csv(summary_csv, index=False)

    with open(diff_json, "w", encoding="utf-8") as f:
        json.dump(df_diff.to_dict(orient="records"), f, indent=2)

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(df_summary.to_dict(orient="records"), f, indent=2)

    # --------------------------------------------------------
    # Plot 1: Kurven Real vs Theorie
    # --------------------------------------------------------
    plt.figure(figsize=(12, 8))
    for metric, label in active_metrics:
        sub = df_diff[df_diff["metric"] == metric].sort_values("z_m")
        if sub.empty:
            continue

        plt.plot(sub["z_m"], sub["real_mm"], marker="o", linestyle="--", label=f"{label} Real")
        plt.plot(sub["z_m"], sub["theory_mm"], marker="o", linestyle="-", label=f"{label} Theorie")

    plt.xlabel("z [m]")
    plt.ylabel("Beam size [mm]")
    plt.title("Real vs Theorie")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9, ncol=2)
    plt.tight_layout()
    plot_compare = OUTPUT_DIR / "real_vs_theory.png"
    plt.savefig(plot_compare, dpi=220)
    plt.close()

    # --------------------------------------------------------
    # Plot 2: Differenz = Real - Theorie
    # --------------------------------------------------------
    plt.figure(figsize=(12, 7))
    for metric, label in active_metrics:
        sub = df_diff[df_diff["metric"] == metric].sort_values("z_m")
        if sub.empty:
            continue

        plt.plot(sub["z_m"], sub["difference_mm"], marker="o", linewidth=2, label=label)

    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("z [m]")
    plt.ylabel("Differenz [mm]")
    plt.title("Differenz: Real - Theorie")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plot_diff = OUTPUT_DIR / "difference_real_minus_theory.png"
    plt.savefig(plot_diff, dpi=220)
    plt.close()

    # --------------------------------------------------------
    # Plot 3: Absolutbetrag der Differenz
    # --------------------------------------------------------
    plt.figure(figsize=(12, 7))
    for metric, label in active_metrics:
        sub = df_diff[df_diff["metric"] == metric].sort_values("z_m")
        if sub.empty:
            continue

        plt.plot(sub["z_m"], np.abs(sub["difference_mm"]), marker="o", linewidth=2, label=label)

    plt.xlabel("z [m]")
    plt.ylabel("|Differenz| [mm]")
    plt.title("Absolutbetrag der Differenz: |Real - Theorie|")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plot_absdiff = OUTPUT_DIR / "abs_difference_real_minus_theory.png"
    plt.savefig(plot_absdiff, dpi=220)
    plt.close()

    print("=" * 80)
    print("FERTIG")
    print("=" * 80)
    print(f"Real CSV:          {REAL_CSV}")
    print(f"Theory CSV:        {THEORY_CSV}")
    print(f"Output-Ordner:     {OUTPUT_DIR}")
    print(f"Vergleichsplot:    {plot_compare}")
    print(f"Differenzplot:     {plot_diff}")
    print(f"|Differenz|-Plot:  {plot_absdiff}")
    print(f"Differenz CSV:     {diff_csv}")
    print(f"Summary CSV:       {summary_csv}")
    print()
    print("Zusammenfassung:")
    if not df_summary.empty:
        print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()
