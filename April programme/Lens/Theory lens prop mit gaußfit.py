import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import csv

# ================= CONFIG =================
INT_FILE = Path("theory_INT.csv")
PHA_FILE = Path("theory_PHA.csv")

OUTPUT_CSV = Path("theory_propagation.csv")

WAVELENGTH = 800e-9
PIXEL_SIZE = 10e-6  # m

Z_LIST = np.linspace(0, 10, 20)

# =========================================

def load_csv_auto(path):
    for delim in [",", ";", "\t"]:
        try:
            arr = np.loadtxt(path, delimiter=delim)
            if arr.ndim == 2:
                return arr.astype(float)
        except Exception:
            pass
    raise ValueError(f"Cannot read file: {path}")

def convert_phase_to_radians(phase, unit="waves"):
    if unit == "waves":
        return 2*np.pi * phase
    elif unit == "rad":
        return phase
    else:
        raise ValueError("Unsupported phase unit")

def build_field(intensity, phase):
    intensity = np.clip(intensity, 0, None)
    intensity /= np.max(intensity)
    amplitude = np.sqrt(intensity)
    return amplitude * np.exp(1j * phase)

def angular_spectrum(U0, dx, wavelength, z):
    ny, nx = U0.shape

    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dx)
    FX, FY = np.meshgrid(fx, fy)

    k = 2*np.pi / wavelength
    kx = 2*np.pi * FX
    ky = 2*np.pi * FY

    kz_sq = k**2 - kx**2 - ky**2
    kz = np.sqrt(np.maximum(kz_sq, 0))

    H = np.exp(1j * kz * z)
    H[kz_sq < 0] = 0  # evanescent cutoff

    return np.fft.ifft2(np.fft.fft2(U0) * H)

def beam_size(I):
    I = np.clip(I, 0, None)
    y, x = np.indices(I.shape)

    P = I.sum()
    if P <= 0:
        return 0

    x0 = (I*x).sum() / P
    y0 = (I*y).sum() / P

    sigma_x = np.sqrt(((x-x0)**2 * I).sum() / P)
    sigma_y = np.sqrt(((y-y0)**2 * I).sum() / P)

    return 4 * 0.5 * (sigma_x + sigma_y)  # D4σ mean

# ================= MAIN =================

print("Loading theory data...")

intensity = load_csv_auto(INT_FILE)
phase_raw = load_csv_auto(PHA_FILE)

if intensity.shape != phase_raw.shape:
    raise ValueError("Shape mismatch INT vs PHA")

phase = convert_phase_to_radians(phase_raw, unit="waves")

U0 = build_field(intensity, phase)

results = []

print("Starting propagation...")

for z in Z_LIST:
    Uz = angular_spectrum(U0, PIXEL_SIZE, WAVELENGTH, z)
    Iz = np.abs(Uz)**2

    size = beam_size(Iz)

    results.append({
        "z_m": z,
        "beam_size": size
    })

# ================= SAVE =================

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["z_m", "beam_size"])
    writer.writeheader()
    writer.writerows(results)

print("Theory propagation finished.")
print(f"Saved to: {OUTPUT_CSV}")