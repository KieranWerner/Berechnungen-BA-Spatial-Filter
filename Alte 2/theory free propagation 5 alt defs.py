from pathlib import Path
import json
import csv
import numpy as np
import matplotlib.pyplot as plt

BASE = Path(r"C:\Users\User\Desktop\Theory free propagation")
PHA_FILE = BASE / r"average_PHA.csv"
INT_FILE = BASE / r"average_INT.csv"
OUTPUT_DIR = BASE / "beam_size_vs_z_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# Bekannter Eingangsdurchmesser zur Kalibrierung der Pixelgröße
INPUT_D4SIGMA_MM = 11.0

# Wellenlänge
WAVELENGTH_M = 800e-9

# Vorzeichen der Phase
PHASE_SIGN = -1.0

# Propagationsabstände
Z_LIST = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 8.1, 10.0, 12.0, 15.0]


def load_csv_auto(path):
    for delim in [",", ";", "\t"]:
        try:
            arr = np.loadtxt(path, delimiter=delim)
            if arr.ndim == 2 and arr.size > 0:
                return arr
        except Exception:
            pass
    raise ValueError(f"Could not parse {path}")


def angular_spectrum(U0, dx, dy, wavelength, z, pad_factor=4):
    ny, nx = U0.shape

    py = int((pad_factor - 1) * ny / 2)
    px = int((pad_factor - 1) * nx / 2)

    U = np.pad(U0, ((py, py), (px, px)), mode="constant")
    Ny, Nx = U.shape

    fx = np.fft.fftfreq(Nx, d=dx)
    fy = np.fft.fftfreq(Ny, d=dy)
    FX, FY = np.meshgrid(fx, fy)

    k = 2 * np.pi / wavelength
    kx = 2 * np.pi * FX
    ky = 2 * np.pi * FY

    kz_sq = k**2 - kx**2 - ky**2
    kz = np.sqrt(np.maximum(kz_sq, 0.0))

    H = np.exp(1j * kz * z)
    H[kz_sq < 0] = 0.0

    Uz = np.fft.ifft2(np.fft.fft2(U) * H)
    return Uz[py:py + ny, px:px + nx]


def beam_metrics_definitions(I, dx, dy):
    """
    Liefert die gewünschten Definitionen:

    D4sigma  : Mittelwert aus D4sigma_x und D4sigma_y
    EE50     : Durchmesser des Kreises mit 50% eingeschlossener Energie
    EE86     : Durchmesser des Kreises mit 86% eingeschlossener Energie
    Deq50    : äquivalenter Durchmesser für 50% Energie
               (hier identisch zu EE50 bei kreisförmiger Definition)
    EEq13.8  : Durchmesser des Kreises mit 13.8% eingeschlossener Energie
    """

    I = np.clip(np.asarray(I, float), 0, None)
    P = I.sum()
    if P <= 0:
        raise ValueError("Intensity sum is zero; metrics cannot be computed.")

    y = np.arange(I.shape[0])
    x = np.arange(I.shape[1])
    X, Y = np.meshgrid(x, y)

    x0_px = float((I * X).sum() / P)
    y0_px = float((I * Y).sum() / P)

    Xc = (X - x0_px) * dx
    Yc = (Y - y0_px) * dy
    R = np.sqrt(Xc**2 + Yc**2)

    sigma_x = float(np.sqrt((I * Xc**2).sum() / P))
    sigma_y = float(np.sqrt((I * Yc**2).sum() / P))

    d4sigma_x = 4.0 * sigma_x
    d4sigma_y = 4.0 * sigma_y
    d4sigma = 0.5 * (d4sigma_x + d4sigma_y)

    order = np.argsort(R.ravel())
    r_sorted = R.ravel()[order]
    i_sorted = I.ravel()[order]
    cdf = np.cumsum(i_sorted) / P

    def encircled_energy_radius(frac):
        idx = min(np.searchsorted(cdf, frac), len(r_sorted) - 1)
        return float(r_sorted[idx])

    r_ee50 = encircled_energy_radius(0.50)
    r_ee86 = encircled_energy_radius(0.86)
    r_ee138 = encircled_energy_radius(0.138)

    ee50 = 2.0 * r_ee50
    ee86 = 2.0 * r_ee86
    eeq13_8 = 2.0 * r_ee138

    # Bei kreisförmiger EE-Definition ist der äquivalente Durchmesser:
    # D_eq = 2 * sqrt(A/pi), mit A = pi*r^2 => D_eq = 2*r
    deq50 = 2.0 * np.sqrt((np.pi * r_ee50**2) / np.pi)

    return {
        "centroid_x": x0_px * dx,
        "centroid_y": y0_px * dy,
        "D4sigma_x": d4sigma_x,
        "D4sigma_y": d4sigma_y,
        "D4sigma": d4sigma,
        "EE50": ee50,
        "EE86": ee86,
        "Deq50": deq50,
        "EEq13.8": eeq13_8,
    }


def build_field_from_measurement(intensity, phase_waves=None, use_phase=True, phase_sign=-1.0):
    intensity = np.asarray(intensity, dtype=float)

    valid_int = np.isfinite(intensity) & (intensity > 0)
    if not np.any(valid_int):
        raise ValueError("No valid positive intensity values found.")

    int_norm = np.zeros_like(intensity, dtype=float)
    int_norm[valid_int] = intensity[valid_int] / np.nanmax(intensity[valid_int])

    A = np.sqrt(int_norm)

    if use_phase:
        if phase_waves is None:
            raise ValueError("phase_waves is required when use_phase=True")

        phase_waves = np.asarray(phase_waves, dtype=float)
        valid_phase = np.isfinite(phase_waves)

        phase_waves_clean = np.zeros_like(phase_waves, dtype=float)
        phase_waves_clean[valid_phase] = phase_waves[valid_phase]

        phase_rad = phase_sign * 2.0 * np.pi * phase_waves_clean
        U0 = A * np.exp(1j * phase_rad)
    else:
        U0 = A.astype(np.complex128)

    return U0, int_norm


def propagate_and_analyze(label, U0, dx_m, dy_m, wavelength_m, z_list, output_dir):
    rows = []

    for z in z_list:
        Uz = U0 if z == 0 else angular_spectrum(U0, dx_m, dy_m, wavelength_m, z, pad_factor=4)
        Iz = np.abs(Uz) ** 2
        m = beam_metrics_definitions(Iz, dx_m, dy_m)

        rows.append({
            "mode": label,
            "z_m": z,
            "centroid_x_mm": m["centroid_x"] * 1e3,
            "centroid_y_mm": m["centroid_y"] * 1e3,
            "D4sigma_x_mm": m["D4sigma_x"] * 1e3,
            "D4sigma_y_mm": m["D4sigma_y"] * 1e3,
            "D4sigma_mm": m["D4sigma"] * 1e3,
            "EE50_mm": m["EE50"] * 1e3,
            "EE86_mm": m["EE86"] * 1e3,
            "Deq50_mm": m["Deq50"] * 1e3,
            "EEq13_8_mm": m["EEq13.8"] * 1e3,
        })

    csv_path = output_dir / f"beam_metrics_vs_z_{label}.csv"
    json_path = output_dir / f"beam_metrics_vs_z_{label}.json"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "mode": label,
            "pixel_size_um": dx_m * 1e6,
            "wavelength_nm": wavelength_m * 1e9,
            "rows": rows
        }, f, indent=2)

    return rows


# -------------------------------------------------------------------------
# Daten laden
# -------------------------------------------------------------------------
phase_waves = load_csv_auto(PHA_FILE)
intensity = load_csv_auto(INT_FILE)

if phase_waves.shape != intensity.shape:
    raise ValueError(f"Shape mismatch: phase {phase_waves.shape} vs intensity {intensity.shape}")

# -------------------------------------------------------------------------
# Pixelgröße über D4sigma kalibrieren
# -------------------------------------------------------------------------
valid_int = np.isfinite(intensity) & (intensity > 0)
int_norm_for_scale = np.zeros_like(intensity, dtype=float)
int_norm_for_scale[valid_int] = intensity[valid_int] / np.nanmax(intensity[valid_int])

tmp_px = beam_metrics_definitions(int_norm_for_scale, 1.0, 1.0)
input_d4_px = tmp_px["D4sigma"]

dx_m = (INPUT_D4SIGMA_MM / input_d4_px) * 1e-3
dy_m = dx_m

# -------------------------------------------------------------------------
# Startfelder
# -------------------------------------------------------------------------
U0_with_phase, int_norm = build_field_from_measurement(
    intensity=intensity,
    phase_waves=phase_waves,
    use_phase=True,
    phase_sign=PHASE_SIGN,
)

U0_without_phase, _ = build_field_from_measurement(
    intensity=intensity,
    phase_waves=None,
    use_phase=False,
    phase_sign=PHASE_SIGN,
)

# -------------------------------------------------------------------------
# Propagation
# -------------------------------------------------------------------------
rows_with_phase = propagate_and_analyze(
    label="with_phase",
    U0=U0_with_phase,
    dx_m=dx_m,
    dy_m=dy_m,
    wavelength_m=WAVELENGTH_M,
    z_list=Z_LIST,
    output_dir=OUTPUT_DIR,
)

rows_without_phase = propagate_and_analyze(
    label="without_phase",
    U0=U0_without_phase,
    dx_m=dx_m,
    dy_m=dy_m,
    wavelength_m=WAVELENGTH_M,
    z_list=Z_LIST,
    output_dir=OUTPUT_DIR,
)

# -------------------------------------------------------------------------
# Kombinierte CSV
# -------------------------------------------------------------------------
all_rows = rows_with_phase + rows_without_phase
with open(OUTPUT_DIR / "beam_metrics_vs_z_comparison.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
    writer.writeheader()
    writer.writerows(all_rows)

# -------------------------------------------------------------------------
# Plot
# -------------------------------------------------------------------------
z_with = [r["z_m"] for r in rows_with_phase]
z_without = [r["z_m"] for r in rows_without_phase]

plt.figure(figsize=(10, 6))
plt.plot(z_with, [r["D4sigma_mm"] for r in rows_with_phase], marker="o", label="D4sigma (with phase)")
plt.plot(z_with, [r["EE50_mm"] for r in rows_with_phase], marker="s", label="EE50 (with phase)")
plt.plot(z_with, [r["EE86_mm"] for r in rows_with_phase], marker="^", label="EE86 (with phase)")
plt.plot(z_with, [r["Deq50_mm"] for r in rows_with_phase], marker="d", label="Deq50 (with phase)")
plt.plot(z_with, [r["EEq13_8_mm"] for r in rows_with_phase], marker="v", label="EEq13.8 (with phase)")

#plt.plot(z_without, [r["D4sigma_mm"] for r in rows_without_phase], marker="o", linestyle="--", label="D4sigma (without phase)")
#plt.plot(z_without, [r["EE50_mm"] for r in rows_without_phase], marker="s", linestyle="--", label="EE50 (without phase)")
#plt.plot(z_without, [r["EE86_mm"] for r in rows_without_phase], marker="^", linestyle="--", label="EE86 (without phase)")
#plt.plot(z_without, [r["Deq50_mm"] for r in rows_without_phase], marker="d", linestyle="--", label="Deq50 (without phase)")
#plt.plot(z_without, [r["EEq13_8_mm"] for r in rows_without_phase], marker="v", linestyle="--", label="EEq13.8 (without phase)")

plt.xlabel("Propagation distance z [m]")
plt.ylabel("Beam diameter [mm]")
plt.title("Beam metrics versus propagation distance")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "beam_metrics_vs_z_comparison.png", dpi=180)
plt.close()

# -------------------------------------------------------------------------
# Eingangsplots
# -------------------------------------------------------------------------
plt.figure(figsize=(6, 5))
plt.imshow(int_norm, origin="lower")
plt.colorbar(label="Normalized intensity")
plt.title("Input normalized intensity")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "input_intensity.png", dpi=180)
plt.close()

plt.figure(figsize=(6, 5))
plt.imshow(phase_waves, origin="lower")
plt.colorbar(label="Phase [waves at 800 nm]")
plt.title("Input phase map")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "input_phase_waves.png", dpi=180)
plt.close()

meta = {
    "pixel_size_um": dx_m * 1e6,
    "input_d4sigma_mm": INPUT_D4SIGMA_MM,
    "wavelength_nm": WAVELENGTH_M * 1e9,
    "phase_unit": "waves at 800 nm",
    "phase_to_radians": "phi_rad = PHASE_SIGN * 2*pi*phase_waves",
    "phase_sign": PHASE_SIGN,
    "definitions": {
        "D4sigma": "mean of D4sigma_x and D4sigma_y",
        "EE50": "diameter enclosing 50% of total energy",
        "EE86": "diameter enclosing 86% of total energy",
        "Deq50": "equivalent diameter for 50% enclosed energy",
        "EEq13.8": "diameter enclosing 13.8% of total energy"
    }
}
with open(OUTPUT_DIR / "run_metadata.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print("Done.")
print(f"Pixel size: {dx_m*1e6:.3f} um")
print(f"Results written to: {OUTPUT_DIR}")