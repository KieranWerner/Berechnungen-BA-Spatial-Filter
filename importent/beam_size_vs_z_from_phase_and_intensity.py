#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VS-Code-Skript:
- liest Intensität und Phase aus CSV
- baut daraus das komplexe Feld bei START_Z_CM
- propagiert von START_Z_CM bis END_Z_CM
- berechnet für jede z-Position den Strahlverlauf
- speichert am Ende:
    1) beam_size_vs_z.csv
    2) beam_size_vs_z.json
    3) beam_size_vs_z.png

Kein Kommandozeilen-Aufruf nötig.
Einfach in VS Code öffnen und Run drücken.

WICHTIG:
- Die Phase muss in Radiant vorliegen.
- DX / DY müssen zur Eingangsdatei passen.
- Wenn USE_LENS = True, wird eine dünne Linse in die Propagation eingefügt.
"""

from pathlib import Path
import json
import math
import csv

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# HIER EINSTELLEN
# ============================================================

INTENSITY_PATH = Path(r"C:\Users\User\Desktop\combined INT front csv\average_INT no bg substract.csv")
PHASE_PATH = Path(r"C:\Users\User\Desktop\combined PHA front csv\average_PHA no bg substract.csv")

OUTPUT_DIR = Path(r"C:\Users\User\Desktop\Propagation_350_to_1000cm")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Optische Parameter
WAVELENGTH_M = 532e-9
DX_M = 7.4e-6
DY_M = DX_M

# Start- und Endebene
START_Z_CM = 350.0
END_Z_CM = 1000.0
STEP_Z_CM = 10.0

# Optionale Linse
USE_LENS = False
LENS_F_M = 5.0
LENS_Z_CM = 500.0   # absolute Position der Linse entlang der z-Achse, z. B. 500 cm

# Welche Kurven geplottet werden sollen
PLOT_D4SIGMA = True
PLOT_EE50 = False
PLOT_EE86 = False
PLOT_FWHM = False

# CSV-Import
CSV_DELIMITER = ","
SKIPROWS = 0


# ============================================================
# HILFSFUNKTIONEN
# ============================================================

def load_csv_auto(path: Path) -> np.ndarray:
    """Versucht CSV mit mehreren Trennzeichen einzulesen."""
    for delim in [CSV_DELIMITER, ",", ";", "\t"]:
        try:
            arr = np.loadtxt(path, delimiter=delim, skiprows=SKIPROWS)
            if arr.ndim == 2 and arr.size > 0:
                return arr.astype(np.float64)
        except Exception:
            pass
    raise ValueError(f"Konnte Datei nicht als 2D-CSV lesen: {path}")


def save_json(path: Path, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def wrap_phase(phi: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * phi))


def angular_spectrum(field: np.ndarray, wavelength: float, dx: float, dy: float, z: float, pad_factor: int = 4) -> np.ndarray:
    """
    Angular-Spectrum-Propagation mit Zero-Padding.
    """
    ny, nx = field.shape

    py = int((pad_factor - 1) * ny / 2)
    px = int((pad_factor - 1) * nx / 2)

    U = np.pad(field, ((py, py), (px, px)), mode="constant")
    Ny, Nx = U.shape

    fx = np.fft.fftfreq(Nx, d=dx)
    fy = np.fft.fftfreq(Ny, d=dy)
    FX, FY = np.meshgrid(fx, fy)

    k = 2.0 * np.pi / wavelength
    kx = 2.0 * np.pi * FX
    ky = 2.0 * np.pi * FY

    kz_sq = k**2 - kx**2 - ky**2
    kz = np.sqrt(np.maximum(kz_sq, 0.0))

    H = np.exp(1j * kz * z)
    H[kz_sq < 0] = 0.0

    Uz = np.fft.ifft2(np.fft.fft2(U) * H)
    return Uz[py:py + ny, px:px + nx]


def thin_lens_phase(shape: tuple[int, int], wavelength: float, dx: float, dy: float, focal_length: float) -> np.ndarray:
    """
    Dünne Linsenphase exp(-i k/(2f) (x^2+y^2)).
    """
    ny, nx = shape
    k = 2.0 * np.pi / wavelength

    x = (np.arange(nx) - nx / 2) * dx
    y = (np.arange(ny) - ny / 2) * dy
    X, Y = np.meshgrid(x, y)

    phi = -(k / (2.0 * focal_length)) * (X**2 + Y**2)
    return np.exp(1j * phi)


def propagate_from_start_to_absolute_z(field_start: np.ndarray, z_abs_cm: float) -> np.ndarray:
    """
    Propagiert vom Startfeld bei START_Z_CM zur absoluten Position z_abs_cm.
    Wenn USE_LENS=True und die Linse zwischen Start und Ziel liegt, wird sie berücksichtigt.
    """
    z0_m = START_Z_CM / 100.0
    zt_m = z_abs_cm / 100.0

    if zt_m < z0_m:
        raise ValueError(f"z_abs_cm={z_abs_cm} liegt vor der Startebene {START_Z_CM} cm.")

    if not USE_LENS:
        return angular_spectrum(field_start, WAVELENGTH_M, DX_M, DY_M, zt_m - z0_m)

    lens_m = LENS_Z_CM / 100.0

    if lens_m <= z0_m or lens_m >= zt_m:
        return angular_spectrum(field_start, WAVELENGTH_M, DX_M, DY_M, zt_m - z0_m)

    # 1) bis zur Linse
    field_at_lens = angular_spectrum(field_start, WAVELENGTH_M, DX_M, DY_M, lens_m - z0_m)

    # 2) Linsenphase
    lens = thin_lens_phase(field_start.shape, WAVELENGTH_M, DX_M, DY_M, LENS_F_M)
    field_after_lens = field_at_lens * lens

    # 3) weiter bis Ziel
    return angular_spectrum(field_after_lens, WAVELENGTH_M, DX_M, DY_M, zt_m - lens_m)


# ============================================================
# STRAHLMETRIKEN
# ============================================================

def centroid(I: np.ndarray) -> tuple[float, float]:
    I = np.clip(np.asarray(I, float), 0, None)
    total = I.sum()
    if total <= 0:
        return np.nan, np.nan

    y, x = np.indices(I.shape)
    cx = float((I * x).sum() / total)
    cy = float((I * y).sum() / total)
    return cx, cy


def d4sigma_avg_px(I: np.ndarray) -> float:
    I = np.clip(np.asarray(I, float), 0, None)
    total = I.sum()
    if total <= 0:
        return np.nan

    y, x = np.indices(I.shape)
    cx, cy = centroid(I)

    sigma_x = math.sqrt(float((I * (x - cx) ** 2).sum() / total))
    sigma_y = math.sqrt(float((I * (y - cy) ** 2).sum() / total))

    dx = 4.0 * sigma_x
    dy = 4.0 * sigma_y
    return 0.5 * (dx + dy)


def encircled_energy_diameter_px(I: np.ndarray, fraction: float) -> float:
    I = np.clip(np.asarray(I, float), 0, None)
    total = I.sum()
    if total <= 0:
        return np.nan

    cx, cy = centroid(I)
    y, x = np.indices(I.shape)
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    order = np.argsort(r.ravel())
    r_sorted = r.ravel()[order]
    i_sorted = I.ravel()[order]

    cdf = np.cumsum(i_sorted) / total
    idx = min(np.searchsorted(cdf, fraction), len(r_sorted) - 1)

    return 2.0 * float(r_sorted[idx])


def fwhm_profile_px(I: np.ndarray) -> float:
    I = np.clip(np.asarray(I, float), 0, None)
    total = I.sum()
    if total <= 0:
        return np.nan

    cx, cy = centroid(I)
    cx_i = int(round(cx))
    cy_i = int(round(cy))

    cx_i = max(0, min(cx_i, I.shape[1] - 1))
    cy_i = max(0, min(cy_i, I.shape[0] - 1))

    profile_x = I[cy_i, :]
    profile_y = I[:, cx_i]

    def single_fwhm(profile: np.ndarray) -> float:
        peak = float(profile.max())
        if peak <= 0:
            return np.nan

        half = 0.5 * peak
        idx = np.where(profile >= half)[0]
        if idx.size < 2:
            return np.nan

        return float(idx[-1] - idx[0])

    fx = single_fwhm(profile_x)
    fy = single_fwhm(profile_y)

    if np.isnan(fx) and np.isnan(fy):
        return np.nan
    if np.isnan(fx):
        return fy
    if np.isnan(fy):
        return fx
    return 0.5 * (fx + fy)


def px_to_mm(value_px: float) -> float:
    return value_px * DX_M * 1e3


# ============================================================
# HAUPTPROGRAMM
# ============================================================

def main():
    print("Lade Intensität:")
    print(INTENSITY_PATH)
    print("Lade Phase:")
    print(PHASE_PATH)

    intensity = load_csv_auto(INTENSITY_PATH)
    phase = load_csv_auto(PHASE_PATH)

    if intensity.shape != phase.shape:
        raise ValueError(f"Shape-Mismatch: Intensität {intensity.shape}, Phase {phase.shape}")

    intensity = np.clip(intensity, 0.0, None)
    amplitude = np.sqrt(intensity)

    # Komplexes Startfeld bei 350 cm
    field_start = amplitude * np.exp(1j * phase)

    z_values_cm = np.arange(START_Z_CM, END_Z_CM + 0.5 * STEP_Z_CM, STEP_Z_CM)

    rows = []

    print("\nStarte Auswertung über z ...")
    for z_cm in z_values_cm:
        field_z = propagate_from_start_to_absolute_z(field_start, float(z_cm))
        I_z = np.abs(field_z) ** 2

        d4_mm = px_to_mm(d4sigma_avg_px(I_z))
        ee50_mm = px_to_mm(encircled_energy_diameter_px(I_z, 0.50))
        ee86_mm = px_to_mm(encircled_energy_diameter_px(I_z, 0.86))
        fwhm_mm = px_to_mm(fwhm_profile_px(I_z))

        rows.append({
            "z_cm": float(z_cm),
            "z_m": float(z_cm / 100.0),
            "D4sigma_avg_mm": float(d4_mm),
            "D_EE50_mm": float(ee50_mm),
            "D_EE86_mm": float(ee86_mm),
            "FWHM_mm": float(fwhm_mm),
        })

        print(
            f"z = {z_cm:6.1f} cm | "
            f"D4sigma = {d4_mm:8.4f} mm | "
            f"EE50 = {ee50_mm:8.4f} mm | "
            f"EE86 = {ee86_mm:8.4f} mm | "
            f"FWHM = {fwhm_mm:8.4f} mm"
        )

    # CSV speichern
    csv_path = OUTPUT_DIR / "beam_size_vs_z.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # JSON speichern
    json_path = OUTPUT_DIR / "beam_size_vs_z.json"
    save_json(json_path, {
        "intensity_path": str(INTENSITY_PATH),
        "phase_path": str(PHASE_PATH),
        "wavelength_m": WAVELENGTH_M,
        "dx_m": DX_M,
        "dy_m": DY_M,
        "start_z_cm": START_Z_CM,
        "end_z_cm": END_Z_CM,
        "step_z_cm": STEP_Z_CM,
        "use_lens": USE_LENS,
        "lens_f_m": LENS_F_M,
        "lens_z_cm": LENS_Z_CM,
        "rows": rows,
    })

    # Plot
    z_m = [r["z_m"] for r in rows]

    plt.figure(figsize=(10, 6))

    if PLOT_D4SIGMA:
        plt.plot(z_m, [r["D4sigma_avg_mm"] for r in rows], marker="o", label="D4sigma")
    if PLOT_EE50:
        plt.plot(z_m, [r["D_EE50_mm"] for r in rows], marker="o", label="EE50")
    if PLOT_EE86:
        plt.plot(z_m, [r["D_EE86_mm"] for r in rows], marker="o", label="EE86")
    if PLOT_FWHM:
        plt.plot(z_m, [r["FWHM_mm"] for r in rows], marker="o", label="FWHM")

    plt.xlabel("z [m]")
    plt.ylabel("Strahlgröße [mm]")
    plt.title("Propagation: Strahlverlauf über z")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plot_path = OUTPUT_DIR / "beam_size_vs_z.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print("\nFertig.")
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")
    print(f"PNG:  {plot_path}")


if __name__ == "__main__":
    main()
