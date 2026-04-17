#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Einfaches VS-Code-Skript:
- liest Intensität aus einer CSV
- liest Phase aus einem Ordner (ebenfalls CSV)
- baut das komplexe Feld U = sqrt(I) * exp(i*phi)
- nimmt an, dass dieses Feld bei START_Z_CM = 350 cm liegt
- propagiert das Feld von 350 cm bis 1000 cm
- speichert für jede Ebene Intensität und Phase als CSV
- speichert zusätzlich PNG-Bilder

WICHTIG:
1) Dieses Skript ist absichtlich OHNE Kommandozeilen-Argumente.
   Du kannst es direkt in VS Code mit Run starten.

2) Oben im Bereich "HIER EINSTELLEN" trägst du deine Pfade ein.

3) Die Phase wird aus einem ORDNER geladen:
   - PHASE_FOLDER = Ordner
   - PHASE_FILENAME = Dateiname in diesem Ordner

4) Die Propagation erfolgt relativ zum Startfeld bei 350 cm.
   Das heißt:
   - Bei 350 cm wird das gemessene komplexe Feld verwendet.
   - Danach wird bis 1000 cm weiterpropagiert.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# HIER EINSTELLEN
# ============================================================

# Intensität
INTENSITY_PATH = r"C:\Users\User\Desktop\combined INT front csv\average_INT no bg substract.csv"

# Phase aus einem Ordner laden
PHASE_FOLDER = r"C:\Users\User\Desktop\combined PHA front csv"
PHASE_FILENAME = "average_PHA no bg substract.csv"

# Optische Parameter
WAVELENGTH = 532e-9   # Meter
DX = 7.4e-6           # Meter
DY = DX

# Start- und Endposition
START_Z_CM = 350.0
END_Z_CM = 1000.0
STEP_Z_CM = 10.0      # z.B. alle 10 cm speichern

# Ausgabeordner
OUTDIR = Path("results_350_to_1000cm")

# CSV-Format
CSV_DELIMITER = ","
SKIPROWS = 0

# PNG-Bilder speichern?
SAVE_PNG = True

# ============================================================
# FUNKTIONEN
# ============================================================

def load_csv(path, delimiter=",", skiprows=0):
    arr = np.loadtxt(path, delimiter=delimiter, skiprows=skiprows)
    if arr.ndim != 2:
        raise ValueError(f"Datei ist kein 2D-Array: {path}")
    return arr.astype(np.float64)

def save_csv(path, arr):
    np.savetxt(path, arr, delimiter=",")

def wrap_phase(phi):
    return np.angle(np.exp(1j * phi))

def angular_spectrum(field, wavelength, dx, dy, z):
    """
    Propagiert ein komplexes Feld um die Distanz z [m]
    mit dem Angular-Spectrum-Verfahren.
    """
    ny, nx = field.shape
    k = 2.0 * np.pi / wavelength

    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dy)
    FX, FY = np.meshgrid(fx, fy)

    kx = 2.0 * np.pi * FX
    ky = 2.0 * np.pi * FY

    kz_sq = k**2 - kx**2 - ky**2
    kz = np.sqrt(kz_sq.astype(np.complex128))

    H = np.exp(1j * kz * z)

    spectrum = np.fft.fft2(field)
    field_out = np.fft.ifft2(spectrum * H)
    return field_out

def save_plot(intensity, phase, title, png_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    im0 = axes[0].imshow(intensity, origin="lower")
    axes[0].set_title("Intensität")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(phase, origin="lower", vmin=-np.pi, vmax=np.pi)
    axes[1].set_title("Phase [rad]")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(png_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# HAUPTTEIL
# ============================================================

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    phase_path = Path(PHASE_FOLDER) / PHASE_FILENAME

    print("Lade Intensität von:")
    print(INTENSITY_PATH)

    print("Lade Phase von:")
    print(phase_path)

    intensity_350 = load_csv(INTENSITY_PATH, delimiter=CSV_DELIMITER, skiprows=SKIPROWS)
    phase_350 = load_csv(phase_path, delimiter=CSV_DELIMITER, skiprows=SKIPROWS)

    if intensity_350.shape != phase_350.shape:
        raise ValueError(
            f"Shape-Mismatch: Intensität {intensity_350.shape}, Phase {phase_350.shape}"
        )

    # Komplexes Feld bei 350 cm
    intensity_350 = np.clip(intensity_350, 0.0, None)
    amplitude_350 = np.sqrt(intensity_350)
    field_350 = amplitude_350 * np.exp(1j * phase_350)

    # Startfeld speichern
    save_csv(OUTDIR / "z_350cm_intensity.csv", np.abs(field_350)**2)
    save_csv(OUTDIR / "z_350cm_phase_wrapped.csv", wrap_phase(np.angle(field_350)))
    save_csv(OUTDIR / "z_350cm_real.csv", np.real(field_350))
    save_csv(OUTDIR / "z_350cm_imag.csv", np.imag(field_350))

    if SAVE_PNG:
        save_plot(
            np.abs(field_350)**2,
            wrap_phase(np.angle(field_350)),
            "Startfeld bei 350 cm",
            OUTDIR / "z_350cm.png"
        )

    z_values_cm = np.arange(START_Z_CM, END_Z_CM + 0.5 * STEP_Z_CM, STEP_Z_CM)

    print("\nStarte Propagation ...")
    for z_cm in z_values_cm:
        dz_m = (z_cm - START_Z_CM) / 100.0

        if abs(dz_m) < 1e-15:
            field_z = field_350
        else:
            field_z = angular_spectrum(field_350, WAVELENGTH, DX, DY, dz_m)

        intensity_z = np.abs(field_z)**2
        phase_z = wrap_phase(np.angle(field_z))

        tag = f"z_{z_cm:.0f}cm"

        save_csv(OUTDIR / f"{tag}_intensity.csv", intensity_z)
        save_csv(OUTDIR / f"{tag}_phase_wrapped.csv", phase_z)
        save_csv(OUTDIR / f"{tag}_real.csv", np.real(field_z))
        save_csv(OUTDIR / f"{tag}_imag.csv", np.imag(field_z))

        if SAVE_PNG:
            save_plot(
                intensity_z,
                phase_z,
                f"Propagiertes Feld bei {z_cm:.0f} cm",
                OUTDIR / f"{tag}.png"
            )

        print(f"gespeichert: {tag}")

    print("\nFertig.")
    print("Ergebnisse liegen in:")
    print(OUTDIR.resolve())


if __name__ == "__main__":
    main()
