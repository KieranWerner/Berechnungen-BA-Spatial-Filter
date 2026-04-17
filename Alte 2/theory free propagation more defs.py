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

INPUT_D4SIGMA_MM = 11.74
WAVELENGTH_M = 800e-9
PHASE_SIGN = -1.0
Z_LIST = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 8.1, 10.0, 12.0, 15.0]

# Maskierung der gültigen Apertur
INTENSITY_THRESHOLD_REL = 0.02   # 2 % vom Maximum
USE_CIRCLE_MASK = True
CIRCLE_RADIUS_PX = 180           # anpassen falls nötig
CIRCLE_CENTER_X = None           # None -> Bildmitte
CIRCLE_CENTER_Y = None           # None -> Bildmitte


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


def make_valid_mask(intensity, threshold_rel=0.02, use_circle=True,
                    circle_radius_px=180, cx=None, cy=None):
    intensity = np.asarray(intensity, dtype=float)
    thr = threshold_rel * np.nanmax(intensity)

    valid = np.isfinite(intensity) & (intensity > thr)

    ny, nx = intensity.shape
    Y, X = np.indices((ny, nx))

    if cx is None:
        cx = (nx - 1) / 2.0
    if cy is None:
        cy = (ny - 1) / 2.0

    if use_circle:
        circle = (X - cx)**2 + (Y - cy)**2 <= circle_radius_px**2
        valid = valid & circle

    return valid, float(thr), float(cx), float(cy)


def build_field_from_measurement(intensity, valid_mask, phase_waves=None,
                                 use_phase=True, phase_sign=-1.0):
    intensity = np.asarray(intensity, dtype=float)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    if not np.any(valid_mask):
        raise ValueError("Valid mask is empty.")

    int_norm = np.zeros_like(intensity, dtype=float)
    int_norm[valid_mask] = intensity[valid_mask] / np.nanmax(intensity[valid_mask])

    A = np.zeros_like(intensity, dtype=float)
    A[valid_mask] = np.sqrt(int_norm[valid_mask])

    if use_phase:
        if phase_waves is None:
            raise ValueError("phase_waves is required when use_phase=True")

        phase_waves = np.asarray(phase_waves, dtype=float)
        phase_waves_clean = np.zeros_like(phase_waves, dtype=float)
        phase_waves_clean[valid_mask] = phase_waves[valid_mask]

        # SID4-Phase ist in Wellenlängen angegeben -> Radiant
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
            "d4sigma_mean_mm": m["d4sigma_mean"] * 1e3,
            "diam_mean_radius_mm": 2.0 * m["mean_radius"] * 1e3,
            "d50_mm": 2.0 * m["r50"] * 1e3,
            "d86_mm": 2.0 * m["r86"] * 1e3,
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
phase_waves = load_csv_auto(PHA_FILE)
intensity = load_csv_auto(INT_FILE)

if phase_waves.shape != intensity.shape:
    raise ValueError(f"Shape mismatch: phase {phase_waves.shape} vs intensity {intensity.shape}")

# -------------------------------------------------------------------------
# Gültige Aperturmaske aus Intensität + optionaler Kreisapertur
# -------------------------------------------------------------------------
valid_mask, threshold_abs, cx, cy = make_valid_mask(
    intensity=intensity,
    threshold_rel=INTENSITY_THRESHOLD_REL,
    use_circle=USE_CIRCLE_MASK,
    circle_radius_px=CIRCLE_RADIUS_PX,
    cx=CIRCLE_CENTER_X,
    cy=CIRCLE_CENTER_Y,
)

# -------------------------------------------------------------------------
# Pixelgröße aus bekanntem Eingangsdurchmesser kalibrieren
# -------------------------------------------------------------------------
int_norm_for_scale = np.zeros_like(intensity, dtype=float)
int_norm_for_scale[valid_mask] = intensity[valid_mask] / np.nanmax(intensity[valid_mask])

tmp_px = beam_metrics_units(int_norm_for_scale, 1.0, 1.0)
input_d4_px = tmp_px["d4sigma_mean"]

dx_m = (INPUT_D4SIGMA_MM / input_d4_px) * 1e-3
dy_m = dx_m

# -------------------------------------------------------------------------
# Startfelder erzeugen: mit und ohne Phase
# -------------------------------------------------------------------------
U0_with_phase, int_norm = build_field_from_measurement(
    intensity=intensity,
    valid_mask=valid_mask,
    phase_waves=phase_waves,
    use_phase=True,
    phase_sign=PHASE_SIGN,
)

U0_without_phase, _ = build_field_from_measurement(
    intensity=intensity,
    valid_mask=valid_mask,
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
with open(OUTPUT_DIR / "beam_size_vs_z_comparison.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
    writer.writeheader()
    writer.writerows(all_rows)

# -------------------------------------------------------------------------
# Plot: alternative Beam-Size-Definitionen
# -------------------------------------------------------------------------
z_with = [r["z_m"] for r in rows_with_phase]
z_without = [r["z_m"] for r in rows_without_phase]

plt.figure(figsize=(10, 6))

plt.plot(z_with, [r["d4sigma_mean_mm"] for r in rows_with_phase],
         marker="o", label="D4σ mean (with phase)")
plt.plot(z_with, [r["diam_mean_radius_mm"] for r in rows_with_phase],
         marker="s", label="2·mean radius (with phase)")
plt.plot(z_with, [r["d50_mm"] for r in rows_with_phase],
         marker="^", label="D50 (with phase)")
plt.plot(z_with, [r["d86_mm"] for r in rows_with_phase],
         marker="d", label="D86 (with phase)")

plt.plot(z_without, [r["d4sigma_mean_mm"] for r in rows_without_phase],
         marker="o", linestyle="--", label="D4σ mean (without phase)")
plt.plot(z_without, [r["diam_mean_radius_mm"] for r in rows_without_phase],
         marker="s", linestyle="--", label="2·mean radius (without phase)")
plt.plot(z_without, [r["d50_mm"] for r in rows_without_phase],
         marker="^", linestyle="--", label="D50 (without phase)")
plt.plot(z_without, [r["d86_mm"] for r in rows_without_phase],
         marker="d", linestyle="--", label="D86 (without phase)")

plt.xlabel("Propagation distance z [m]")
plt.ylabel("Beam size [mm]")
plt.title("Beam size versus propagation distance using different size definitions")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "beam_size_vs_z_alt_definitions.png", dpi=180)
plt.close()

# -------------------------------------------------------------------------
# Diagnoseplots
# -------------------------------------------------------------------------
plt.figure(figsize=(6, 5))
plt.imshow(int_norm, origin="lower")
plt.colorbar(label="Normalized intensity")
plt.title("Input normalized intensity (masked)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "input_intensity_masked.png", dpi=180)
plt.close()

plt.figure(figsize=(6, 5))
phase_plot = np.full_like(phase_waves, np.nan, dtype=float)
phase_plot[valid_mask] = phase_waves[valid_mask]
plt.imshow(phase_plot, origin="lower")
plt.colorbar(label="Phase [waves at 800 nm]")
plt.title("Input phase map (masked)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "input_phase_waves_masked.png", dpi=180)
plt.close()

plt.figure(figsize=(6, 5))
plt.imshow(valid_mask.astype(float), origin="lower")
plt.colorbar(label="Valid mask")
plt.title("Aperture mask")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "aperture_mask.png", dpi=180)
plt.close()

# -------------------------------------------------------------------------
# Metadaten
# -------------------------------------------------------------------------
meta = {
    "pixel_size_um": dx_m * 1e6,
    "input_d4sigma_mm": INPUT_D4SIGMA_MM,
    "wavelength_nm": WAVELENGTH_M * 1e9,
    "phase_unit": "waves at 800 nm",
    "phase_to_radians": "phi_rad = PHASE_SIGN * 2*pi*phase_waves",
    "phase_sign": PHASE_SIGN,
    "intensity_threshold_rel": INTENSITY_THRESHOLD_REL,
    "intensity_threshold_abs": threshold_abs,
    "use_circle_mask": USE_CIRCLE_MASK,
    "circle_radius_px": CIRCLE_RADIUS_PX,
    "circle_center_x_px": cx,
    "circle_center_y_px": cy,
}
with open(OUTPUT_DIR / "run_metadata.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print("Done.")
print(f"Pixel size: {dx_m*1e6:.3f} um")
print(f"Threshold abs.: {threshold_abs:.6g}")
print(f"Valid pixels: {int(valid_mask.sum())}")
print(f"Results written to: {OUTPUT_DIR}")