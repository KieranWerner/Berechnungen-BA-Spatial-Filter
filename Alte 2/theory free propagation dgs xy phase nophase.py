from pathlib import Path
import json
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(r"C:\Users\User\Desktop\Theory free propagation")
PHA_FILE = BASE / r"average_PHA.csv"
INT_FILE = BASE / r"average_INT.csv"
OUTPUT_DIR = BASE / "beam_size_vs_z_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# Bekannter Eingangsdurchmesser (zur Kalibrierung der Pixelgröße)
INPUT_D4SIGMA_MM = 11.0

# Wellenlänge der Messung / Propagation
WAVELENGTH_M = 800e-9

# Vorzeichen der Phase, falls das Messsystem die Konvention andersherum liefert
PHASE_SIGN = -1.0

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
    H[kz_sq < 0] = 0.0  # evaneszente Anteile abschneiden

    Uz = np.fft.ifft2(np.fft.fft2(U) * H)
    return Uz[py:py + ny, px:px + nx]


def beam_metrics_units(I, dx, dy):
    I = np.clip(np.asarray(I, float), 0, None)
    P = I.sum()
    if P <= 0:
        raise ValueError("Intensity sum is zero; metrics cannot be computed.")

    y = np.arange(I.shape[0])
    x = np.arange(I.shape[1])
    X, Y = np.meshgrid(x, y)

    x0 = float((I * X).sum() / P)
    y0 = float((I * Y).sum() / P)

    Xc = (X - x0) * dx
    Yc = (Y - y0) * dy
    R = np.sqrt(Xc**2 + Yc**2)

    sigma_x = float(np.sqrt((I * Xc**2).sum() / P))
    sigma_y = float(np.sqrt((I * Yc**2).sum() / P))

    order = np.argsort(R.ravel())
    r_sorted = R.ravel()[order]
    i_sorted = I.ravel()[order]
    cdf = np.cumsum(i_sorted) / P

    def qrad(frac):
        idx = min(np.searchsorted(cdf, frac), len(r_sorted) - 1)
        return float(r_sorted[idx])

    return {
        "centroid_x": x0 * dx,
        "centroid_y": y0 * dy,
        "d4sigma_x": 4 * sigma_x,
        "d4sigma_y": 4 * sigma_y,
        "d4sigma_mean": 0.5 * (4 * sigma_x + 4 * sigma_y),
        "mean_radius": float(np.sqrt((I * R**2).sum() / P)),
        "r50": qrad(0.50),
        "r86": qrad(0.86),
    }


def build_field_from_measurement(intensity, phase_waves=None, use_phase=True, phase_sign=-1.0):
    """
    intensity   : gemessene Intensität
    phase_waves : gemessene Phase in Einheiten von Wellenlängen (bei 800 nm)
    use_phase   : True -> mit gemessener Phase, False -> ebene Phase
    """

    intensity = np.asarray(intensity, dtype=float)
    A = np.zeros_like(intensity, dtype=float)

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

        # Phase ist in Wellenlängen angegeben -> Umrechnung in Radiant:
        # phi_rad = 2*pi*phi_lambda
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
        m = beam_metrics_units(Iz, dx_m, dy_m)

        rows.append({
            "mode": label,
            "z_m": z,
            "centroid_x_mm": m["centroid_x"] * 1e3,
            "centroid_y_mm": m["centroid_y"] * 1e3,
            "d4sigma_x_mm": m["d4sigma_x"] * 1e3,
            "d4sigma_y_mm": m["d4sigma_y"] * 1e3,
            "d4sigma_mean_mm": m["d4sigma_mean"] * 1e3,
            "mean_radius_mm": m["mean_radius"] * 1e3,
            "r50_mm": m["r50"] * 1e3,
            "r86_mm": m["r86"] * 1e3,
        })

    csv_path = output_dir / f"beam_size_vs_z_{label}.csv"
    json_path = output_dir / f"beam_size_vs_z_{label}.json"

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
phase_waves = load_csv_auto(PHA_FILE)   # Phase in Wellenlängen bei 800 nm
intensity = load_csv_auto(INT_FILE)

if phase_waves.shape != intensity.shape:
    raise ValueError(
        f"Shape mismatch: phase {phase_waves.shape} vs intensity {intensity.shape}"
    )

# -------------------------------------------------------------------------
# Pixelgröße aus bekanntem Eingangsdurchmesser kalibrieren
# -------------------------------------------------------------------------
valid_int = np.isfinite(intensity) & (intensity > 0)
int_norm_for_scale = np.zeros_like(intensity, dtype=float)
int_norm_for_scale[valid_int] = intensity[valid_int] / np.nanmax(intensity[valid_int])

tmp_px = beam_metrics_units(int_norm_for_scale, 1.0, 1.0)
input_d4_px = tmp_px["d4sigma_mean"]

dx_m = (INPUT_D4SIGMA_MM / input_d4_px) * 1e-3
dy_m = dx_m

# -------------------------------------------------------------------------
# Startfelder erzeugen
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
# Propagation: mit und ohne Phase
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
with open(OUTPUT_DIR / "beam_size_vs_z_comparison.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
    writer.writeheader()
    writer.writerows(all_rows)

# -------------------------------------------------------------------------
# Plot: Vergleich mit / ohne Phase
# -------------------------------------------------------------------------
z_with = [r["z_m"] for r in rows_with_phase]
z_without = [r["z_m"] for r in rows_without_phase]

plt.figure(figsize=(9, 6))
plt.plot(z_with, [r["d4sigma_mean_mm"] for r in rows_with_phase], marker="o", label="Mean D4σ (with phase)")
plt.plot(z_with, [r["d4sigma_x_mm"] for r in rows_with_phase], marker="s", label="D4σ x (with phase)")
plt.plot(z_with, [r["d4sigma_y_mm"] for r in rows_with_phase], marker="^", label="D4σ y (with phase)")

plt.plot(z_without, [r["d4sigma_mean_mm"] for r in rows_without_phase], marker="o", linestyle="--", label="Mean D4σ (without phase)")
plt.plot(z_without, [r["d4sigma_x_mm"] for r in rows_without_phase], marker="s", linestyle="--", label="D4σ x (without phase)")
plt.plot(z_without, [r["d4sigma_y_mm"] for r in rows_without_phase], marker="^", linestyle="--", label="D4σ y (without phase)")

plt.xlabel("Propagation distance z [m]")
plt.ylabel("Beam size [mm]")
plt.title("Beam size versus propagation distance: with phase vs without phase")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "beam_size_vs_z_comparison.png", dpi=180)
plt.close()

# -------------------------------------------------------------------------
# Optional: Startintensität und Phase speichern/plotten
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
}
with open(OUTPUT_DIR / "run_metadata.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print("Done.")
print(f"Pixel size: {dx_m*1e6:.3f} um")
print(f"Results written to: {OUTPUT_DIR}")