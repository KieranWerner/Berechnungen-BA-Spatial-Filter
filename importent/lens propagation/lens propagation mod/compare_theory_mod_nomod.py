from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# KONFIGURATION
# ============================================================

CFG = {
    "INPUT_DIR": Path.home() / "Desktop" / "lens propagation",
    "REAL_CSV": "theory_lens_propagation_opt_all_except_defocus.csv",
    "THEORY_CSV": "theory_lens_propagation.csv",
    "OUTPUT_PNG": "comparison_lens_propagation.png",
}


# ============================================================
# HILFSFUNKTIONEN
# ============================================================

def load_csv_rows(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {path}")

    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = {}
            for k, v in raw.items():
                if v is None or v == "":
                    row[k] = np.nan
                else:
                    try:
                        row[k] = float(v)
                    except ValueError:
                        row[k] = v
            rows.append(row)

    return rows


def get_z_axis(rows: list[dict[str, float]]) -> np.ndarray:
    if "z_m" in rows[0]:
        return np.array([r["z_m"] for r in rows], dtype=float), "z [m]"
    if "z_cm" in rows[0]:
        return np.array([r["z_cm"] for r in rows], dtype=float), "z [cm]"
    raise KeyError("Keine Spalte 'z_m' oder 'z_cm' gefunden.")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    input_dir = Path(CFG["INPUT_DIR"])
    real_csv = input_dir / CFG["REAL_CSV"]
    theory_csv = input_dir / CFG["THEORY_CSV"]
    output_png = input_dir / CFG["OUTPUT_PNG"]

    real_rows = load_csv_rows(real_csv)
    theory_rows = load_csv_rows(theory_csv)

    z_real, x_label_real = get_z_axis(real_rows)
    z_theory, x_label_theory = get_z_axis(theory_rows)

    if x_label_real != x_label_theory:
        raise ValueError("Real- und Theory-CSV verwenden unterschiedliche z-Einheiten.")

    x_label = x_label_real

    plot_defs = [
        ("D4sigma_avg_mm", "D4σ"),
        ("D_EE50_mm", "EE50"),
        ("D_EE86_mm", "EE86"),
        ("D_area_50pct_mm", "Area50%"),
        ("D_area_13p5pct_mm", "Area13.5%"),
        ("FWHM_mm", "FWHM"),
    ]

    plt.figure(figsize=(12, 8))

    for key, label in plot_defs:
        if key in real_rows[0]:
            y_real = np.array([r[key] for r in real_rows], dtype=float)
            plt.plot(z_real, y_real, marker="o", linestyle="--", linewidth=1.8, label=f"{label} real")

        if key in theory_rows[0]:
            y_theory = np.array([r[key] for r in theory_rows], dtype=float)
            plt.plot(z_theory, y_theory, marker="o", linewidth=2.2, label=f"{label} theory")

    plt.xlabel(x_label)
    plt.ylabel("Beam size [mm]")
    plt.title("Theory and real lens propagation")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(output_png, dpi=220)
    plt.show()

    print("Vergleichsplot gespeichert unter:")
    print(output_png)


if __name__ == "__main__":
    main()