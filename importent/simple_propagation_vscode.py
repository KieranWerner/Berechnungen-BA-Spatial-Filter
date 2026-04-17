#!/usr/bin/env python3
# Einfaches Skript: direkt in VS Code starten (ohne CLI)

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# 🔧 HIER EINSTELLEN
# =========================

INTENSITY_PATH = r"C:\Users\User\Desktop\combined INT front csv\average_INT no bg substract.csv"
PHASE_PATH = r"C:\Users\User\Desktop\combined PHA front csv\average_PHA no bg substract.csv"

WAVELENGTH = 532e-9     # Meter
DX = 7.4e-6             # Pixelgröße (Meter)
DY = DX

Z_PROP = 3.5            # Zielabstand (Meter)

USE_LENS = True
LENS_F = 5.0            # 5 Meter Brennweite
LENS_Z = 2.85           # Position der Linse (Meter)

OUTDIR = Path("results_simple")
OUTDIR.mkdir(exist_ok=True)

# =========================
# Funktionen
# =========================

def load_csv(path):
    return np.loadtxt(path, delimiter=",")

def angular_spectrum(field, wavelength, dx, dy, z):
    ny, nx = field.shape
    k = 2*np.pi / wavelength

    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dy)
    FX, FY = np.meshgrid(fx, fy)

    kx = 2*np.pi*FX
    ky = 2*np.pi*FY

    kz = np.sqrt((k**2 - kx**2 - ky**2).astype(complex))

    H = np.exp(1j * kz * z)

    F = np.fft.fft2(field)
    return np.fft.ifft2(F * H)

def lens_phase(shape, wavelength, dx, dy, f):
    ny, nx = shape
    k = 2*np.pi / wavelength

    x = (np.arange(nx) - nx/2)*dx
    y = (np.arange(ny) - ny/2)*dy
    X, Y = np.meshgrid(x, y)

    return np.exp(-1j * k/(2*f) * (X**2 + Y**2))

# =========================
# LOAD DATA
# =========================

I = load_csv(INTENSITY_PATH)
phi = load_csv(PHASE_PATH)

U0 = np.sqrt(I) * np.exp(1j*phi)

# =========================
# PROPAGATION
# =========================

if USE_LENS:
    U_lens = angular_spectrum(U0, WAVELENGTH, DX, DY, LENS_Z)
    U_lens *= lens_phase(U0.shape, WAVELENGTH, DX, DY, LENS_F)
    U = angular_spectrum(U_lens, WAVELENGTH, DX, DY, Z_PROP - LENS_Z)
else:
    U = angular_spectrum(U0, WAVELENGTH, DX, DY, Z_PROP)

# =========================
# RESULTS
# =========================

I_out = np.abs(U)**2
phi_out = np.angle(U)

np.savetxt(OUTDIR/"intensity.csv", I_out, delimiter=",")
np.savetxt(OUTDIR/"phase.csv", phi_out, delimiter=",")

# Plot
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(I_out)
plt.title("Intensity")

plt.subplot(1,2,2)
plt.imshow(phi_out, vmin=-np.pi, vmax=np.pi)
plt.title("Phase")

plt.tight_layout()
plt.savefig(OUTDIR/"result.png")
plt.show()

print("Fertig! Ergebnisse in:", OUTDIR.resolve())
