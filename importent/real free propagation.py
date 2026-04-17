import math
from pathlib import Path
import json
import csv

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ============================================================
# AUSWAHL: Welche Definitionen sollen geplottet werden?
# ============================================================
PLOT_D4SIGMA_AVG = True
PLOT_D_EE50 = True
PLOT_D_EE86 = True
PLOT_D_AREA_50PCT = True
PLOT_D_AREA_13P5PCT = True
PLOT_FWHM = True

# ============================================================
# EINSTELLUNGEN
# ============================================================
DATASET_NAME = "beamprofiler_front"

PIXEL_SIZE_MM = 0.011 * 2.1
UNIT_LABEL = "mm"

MEASUREMENTS = [
    (0.53, "messung 53cm referenz_bg_subtracted.tiff"),
    (2.00, "messung 200 cm_bg_subtracted.tiff"),
    (3.50, "messung 350cm_bg_subtracted.tiff"),
    (5.00, "messung 500cm_bg_subtracted.tiff"),
    (7.50, "messung 750cm_bg_subtracted.tiff"),
    (12.0, "messung ende_bg_subtracted.tiff"),
]

BASE_DIR = Path(r"C:\Users\User\Desktop\2D Beamprofiler background substracted")


# ============================================================
# HILFSFUNKTIONEN
# ============================================================
def load_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path), dtype=np.float64)


def centroid(img: np.ndarray) -> tuple[float, float]:
    total = img.sum()
    if total <= 0:
        raise ValueError("Bild enthält keine positive Gesamtintensität.")
    y, x = np.indices(img.shape)
    cx = float((img * x).sum() / total)
    cy = float((img * y).sum() / total)
    return cx, cy


def px_to_mm(value_px: float) -> float:
    return value_px * PIXEL_SIZE_MM


# ============================================================
# SIZE-DEFINITIONEN
# ============================================================
def d4sigma_avg(img: np.ndarray) -> float:
    total = img.sum()
    if total <= 0:
        return np.nan

    y, x = np.indices(img.shape)
    cx, cy = centroid(img)

    sigma_x = math.sqrt(float((img * (x - cx) ** 2).sum() / total))
    sigma_y = math.sqrt(float((img * (y - cy) ** 2).sum() / total))

    dx = 4.0 * sigma_x
    dy = 4.0 * sigma_y
    return 0.5 * (dx + dy)


def encircled_energy_diameter(img: np.ndarray, fraction: float) -> float:
    total = img.sum()
    if total <= 0:
        return np.nan

    cx, cy = centroid(img)
    y, x = np.indices(img.shape)

    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).ravel()
    inten = img.ravel()

    order = np.argsort(r)
    r_sorted = r[order]
    i_sorted = inten[order]

    cum = np.cumsum(i_sorted)
    target = fraction * total

    idx = np.searchsorted(cum, target)
    idx = min(idx, len(r_sorted) - 1)
    radius = float(r_sorted[idx])

    return 2.0 * radius


def fwhm_profile(img: np.ndarray) -> float:
    total = img.sum()
    if total <= 0:
        return np.nan

    cx, cy = centroid(img)

    cx_i = int(round(cx))
    cy_i = int(round(cy))

    cx_i = max(0, min(cx_i, img.shape[1] - 1))
    cy_i = max(0, min(cy_i, img.shape[0] - 1))

    profile_x = img[cy_i, :]
    profile_y = img[:, cx_i]

    def single_fwhm(profile: np.ndarray) -> float:
        peak = float(profile.max())
        if peak <= 0:
            return np.nan

        half = peak / 2.0
        indices = np.where(profile >= half)[0]

        if len(indices) < 2:
            return np.nan

        return float(indices[-1] - indices[0])

    fwhm_x = single_fwhm(profile_x)
    fwhm_y = single_fwhm(profile_y)

    if np.isnan(fwhm_x) and np.isnan(fwhm_y):
        return np.nan
    if np.isnan(fwhm_x):
        return fwhm_y
    if np.isnan(fwhm_y):
        return fwhm_x

    return 0.5 * (fwhm_x + fwhm_y)


def area_equivalent_diameter(img: np.ndarray, rel_threshold: float) -> float:
    peak = img.max()
    if peak <= 0:
        return np.nan

    mask = img >= (rel_threshold * peak)
    area_px = float(mask.sum())

    if area_px <= 0:
        return np.nan

    return math.sqrt(4.0 * area_px / math.pi)


# ============================================================
# AUSWERTUNG
# ============================================================
def main() -> None:
    positions_m = []

    curves = {
        "D4sigma_avg": [],
        "D_EE50": [],
        "D_EE86": [],
        "D_area_50pct": [],
        "D_area_13p5pct": [],
        "FWHM": [],
    }

    full_rows = []

    print("Auswertung Beam-Größe über Position")
    print("=" * 100)

    for position_m, filename in MEASUREMENTS:
        path = BASE_DIR / filename

        if not path.exists():
            print(f"WARNUNG: Datei nicht gefunden: {path}")
            continue

        img = load_image(path)
        positions_m.append(position_m)

        d4 = px_to_mm(d4sigma_avg(img))
        ee50 = px_to_mm(encircled_energy_diameter(img, 0.50))
        ee86 = px_to_mm(encircled_energy_diameter(img, 0.86))
        area50 = px_to_mm(area_equivalent_diameter(img, 0.50))
        area135 = px_to_mm(area_equivalent_diameter(img, 0.135))
        fwhm = px_to_mm(fwhm_profile(img))

        curves["D4sigma_avg"].append(d4)
        curves["D_EE50"].append(ee50)
        curves["D_EE86"].append(ee86)
        curves["D_area_50pct"].append(area50)
        curves["D_area_13p5pct"].append(area135)
        curves["FWHM"].append(fwhm)

        full_rows.append({
            "dataset": DATASET_NAME,
            "position_m": position_m,
            "filename": filename,
            "D4sigma_avg_mm": d4,
            "D_EE50_mm": ee50,
            "D_EE86_mm": ee86,
            "D_area_50pct_mm": area50,
            "D_area_13p5pct_mm": area135,
            "FWHM_mm": fwhm,
        })

        print(
            f"{filename:35s} | Position = {position_m:5.2f} m | "
            f"D4sigma_avg = {d4:7.3f} mm | "
            f"D_EE50 = {ee50:7.3f} mm | "
            f"D_EE86 = {ee86:7.3f} mm | "
            f"D_area_50pct = {area50:7.3f} mm | "
            f"D_area_13p5pct = {area135:7.3f} mm | "
            f"FWHM = {fwhm:7.3f} mm"
        )

    if not positions_m:
        raise RuntimeError("Keine gültigen Bilddateien gefunden.")

    # ========================================================
    # Plot-Daten speichern
    # ========================================================
    plot_rows = []
    for i, pos in enumerate(positions_m):
        plot_rows.append({
            "dataset": DATASET_NAME,
            "position_m": pos,
            "D4sigma_avg_mm": curves["D4sigma_avg"][i],
            "D_EE50_mm": curves["D_EE50"][i],
            "D_EE86_mm": curves["D_EE86"][i],
            "D_area_50pct_mm": curves["D_area_50pct"][i],
            "D_area_13p5pct_mm": curves["D_area_13p5pct"][i],
            "FWHM_mm": curves["FWHM"][i],
        })

    plot_csv_path = BASE_DIR / "beam_size_vs_position_plot_data.csv"
    with open(plot_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(plot_rows[0].keys()))
        writer.writeheader()
        writer.writerows(plot_rows)

    plot_json_path = BASE_DIR / "beam_size_vs_position_plot_data.json"
    with open(plot_json_path, "w", encoding="utf-8") as f:
        json.dump(plot_rows, f, indent=2)

    full_csv_path = BASE_DIR / "beam_size_vs_position_full_results.csv"
    with open(full_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(full_rows[0].keys()))
        writer.writeheader()
        writer.writerows(full_rows)

    # ========================================================
    # Plot
    # ========================================================
    plt.figure(figsize=(10, 6))

    if PLOT_D4SIGMA_AVG:
        plt.plot(positions_m, curves["D4sigma_avg"], marker="o", label="D4sigma_avg")
    if PLOT_D_EE50:
        plt.plot(positions_m, curves["D_EE50"], marker="o", label="D_EE50")
    if PLOT_D_EE86:
        plt.plot(positions_m, curves["D_EE86"], marker="o", label="D_EE86")
    if PLOT_D_AREA_50PCT:
        plt.plot(positions_m, curves["D_area_50pct"], marker="o", label="D_area_50pct")
    if PLOT_D_AREA_13P5PCT:
        plt.plot(positions_m, curves["D_area_13p5pct"], marker="o", label="D_area_13p5pct")
    if PLOT_FWHM:
        plt.plot(positions_m, curves["FWHM"], marker="o", label="FWHM")

    plt.xlabel("Position [m]")
    plt.ylabel(f"Größe [{UNIT_LABEL}]")
    plt.title("Beam-Größe über Position")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = BASE_DIR / "beam_size_vs_position.png"
    plt.savefig(out_path, dpi=200)
    plt.show()

    print(f"\nPlot gespeichert unter: {out_path}")
    print(f"Plot-CSV gespeichert unter: {plot_csv_path}")
    print(f"Plot-JSON gespeichert unter: {plot_json_path}")
    print(f"Vollständige CSV gespeichert unter: {full_csv_path}")


if __name__ == "__main__":
    main()