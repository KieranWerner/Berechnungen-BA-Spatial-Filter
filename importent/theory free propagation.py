from pathlib import Path
import json
import csv
import numpy as np
import matplotlib.pyplot as plt

PHA_FILE = Path(r"C:\Users\User\Desktop\combined PHA front csv\average_PHA no bg substract.csv")
INT_FILE = Path(r"C:\Users\User\Desktop\combined average INT front bg substracted\average_INT front bg substracted.csv")
OUTPUT_DIR = Path(r"C:\Users\User\Desktop\Theory free propagation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_D4SIGMA_MM = 13.7
WAVELENGTH_M = 800e-9
PHASE_SIGN = -1.0
Z_LIST = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 8.1, 10.0, 12.0, 13.50]

# Maskierung der gültigen Apertur
INTENSITY_THRESHOLD_REL = 0.02
USE_CIRCLE_MASK = True
CIRCLE_RADIUS_PX = 180
CIRCLE_CENTER_X = None
CIRCLE_CENTER_Y = None


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

        phase_rad = phase_sign * 2.0 * np.pi * phase_waves_clean
        U0 = A * np.exp(1j * phase_rad)
    else:
        U0 = A.astype(np.complex128)

    return U0, int_norm


def _centroid_pixels(I):
    I = np.clip(np.asarray(I, float), 0, None)
    P = I.sum()
    if P <= 0:
        raise ValueError("Intensity sum is zero; metrics cannot be computed.")

    y = np.arange(I.shape[0])
    x = np.arange(I.shape[1])
    X, Y = np.meshgrid(x, y)

    x0 = float((I * X).sum() / P)
    y0 = float((I * Y).sum() / P)
    return x0, y0, P, X, Y


def _encircled_energy_radius(I, R, frac, P):
    order = np.argsort(R.ravel())
    r_sorted = R.ravel()[order]
    i_sorted = I.ravel()[order]
    cdf = np.cumsum(i_sorted) / P
    idx = min(np.searchsorted(cdf, frac), len(r_sorted) - 1)
    return float(r_sorted[idx])


def _area_equivalent_diameter(I, dx, dy, rel_threshold):
    peak = float(np.max(I))
    if peak <= 0:
        return np.nan

    mask = I >= (rel_threshold * peak)
    area_m2 = float(mask.sum()) * dx * dy
    if area_m2 <= 0:
        return np.nan

    return float(np.sqrt(4.0 * area_m2 / np.pi))


def _fwhm_1d(axis_coords, profile):
    profile = np.asarray(profile, dtype=float)
    if profile.size < 2:
        return np.nan

    peak = float(np.max(profile))
    if peak <= 0:
        return np.nan

    half = 0.5 * peak
    idx = np.where(profile >= half)[0]
    if idx.size < 2:
        return np.nan

    i_left = int(idx[0])
    i_right = int(idx[-1])

    # lineare Interpolation links
    if i_left > 0:
        x1, x2 = axis_coords[i_left - 1], axis_coords[i_left]
        y1, y2 = profile[i_left - 1], profile[i_left]
        if y2 != y1:
            left = x1 + (half - y1) * (x2 - x1) / (y2 - y1)
        else:
            left = axis_coords[i_left]
    else:
        left = axis_coords[i_left]

    # lineare Interpolation rechts
    if i_right < profile.size - 1:
        x1, x2 = axis_coords[i_right], axis_coords[i_right + 1]
        y1, y2 = profile[i_right], profile[i_right + 1]
        if y2 != y1:
            right = x1 + (half - y1) * (x2 - x1) / (y2 - y1)
        else:
            right = axis_coords[i_right]
    else:
        right = axis_coords[i_right]

    return float(right - left)


def beam_metrics_units(I, dx, dy):
    I = np.clip(np.asarray(I, float), 0, None)
    x0_px, y0_px, P, X, Y = _centroid_pixels(I)

    Xc = (X - x0_px) * dx
    Yc = (Y - y0_px) * dy
    R = np.sqrt(Xc**2 + Yc**2)

    sigma_x = float(np.sqrt((I * Xc**2).sum() / P))
    sigma_y = float(np.sqrt((I * Yc**2).sum() / P))

    d4sigma_x = 4.0 * sigma_x
    d4sigma_y = 4.0 * sigma_y
    d4sigma_avg = 0.5 * (d4sigma_x + d4sigma_y)

    r50 = _encircled_energy_radius(I, R, 0.50, P)
    r86 = _encircled_energy_radius(I, R, 0.86, P)

    d_ee50 = 2.0 * r50
    d_ee86 = 2.0 * r86

    d_area_50pct = _area_equivalent_diameter(I, dx, dy, 0.50)
    d_area_13p5pct = _area_equivalent_diameter(I, dx, dy, 0.135)

    # Profilbasiertes FWHM durch Schwerpunkt
    x_axis = np.arange(I.shape[1]) * dx
    y_axis = np.arange(I.shape[0]) * dy

    cx_i = int(round(x0_px))
    cy_i = int(round(y0_px))

    cx_i = np.clip(cx_i, 0, I.shape[1] - 1)
    cy_i = np.clip(cy_i, 0, I.shape[0] - 1)

    profile_x = I[cy_i, :]
    profile_y = I[:, cx_i]

    fwhm_x = _fwhm_1d(x_axis, profile_x)
    fwhm_y = _fwhm_1d(y_axis, profile_y)
    fwhm_avg = 0.5 * (fwhm_x + fwhm_y)

    return {
        "centroid_x": x0_px * dx,
        "centroid_y": y0_px * dy,
        "d4sigma_x": d4sigma_x,
        "d4sigma_y": d4sigma_y,
        "d4sigma_avg": d4sigma_avg,
        "d_ee50": d_ee50,
        "d_ee86": d_ee86,
        "d_area_50pct": d_area_50pct,
        "d_area_13p5pct": d_area_13p5pct,
        "fwhm_x": fwhm_x,
        "fwhm_y": fwhm_y,
        "fwhm_avg": fwhm_avg,
    }


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
            "D4sigma_avg_mm": m["d4sigma_avg"] * 1e3,
            "D_EE50_mm": m["d_ee50"] * 1e3,
            "D_EE86_mm": m["d_ee86"] * 1e3,
            "D_area_50pct_mm": m["d_area_50pct"] * 1e3,
            "D_area_13p5pct_mm": m["d_area_13p5pct"] * 1e3,
            "FWHM_mm": m["fwhm_avg"] * 1e3,
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
input_d4_px = tmp_px["d4sigma_avg"]

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
# Plot: 6 Beam-Size-Definitionen
# -------------------------------------------------------------------------
z_with = [r["z_m"] for r in rows_with_phase]
z_without = [r["z_m"] for r in rows_without_phase]

plt.figure(figsize=(11, 7))

# with phase
plt.plot(z_with, [r["D4sigma_avg_mm"] for r in rows_with_phase],
         marker="o", label="D4sigma_avg (with phase)")
plt.plot(z_with, [r["D_EE50_mm"] for r in rows_with_phase],
         marker="s", label="D_EE50 (with phase)")
plt.plot(z_with, [r["D_EE86_mm"] for r in rows_with_phase],
         marker="^", label="D_EE86 (with phase)")
plt.plot(z_with, [r["D_area_50pct_mm"] for r in rows_with_phase],
         marker="d", label="D_area_50pct (with phase)")
plt.plot(z_with, [r["D_area_13p5pct_mm"] for r in rows_with_phase],
         marker="v", label="D_area_13p5pct (with phase)")
plt.plot(z_with, [r["FWHM_mm"] for r in rows_with_phase],
         marker="P", label="FWHM (with phase)")
"""
# without phase
plt.plot(z_without, [r["D4sigma_avg_mm"] for r in rows_without_phase],
         marker="o", linestyle="--", label="D4sigma_avg (without phase)")
plt.plot(z_without, [r["D_EE50_mm"] for r in rows_without_phase],
         marker="s", linestyle="--", label="D_EE50 (without phase)")
plt.plot(z_without, [r["D_EE86_mm"] for r in rows_without_phase],
         marker="^", linestyle="--", label="D_EE86 (without phase)")
plt.plot(z_without, [r["D_area_50pct_mm"] for r in rows_without_phase],
         marker="d", linestyle="--", label="D_area_50pct (without phase)")
plt.plot(z_without, [r["D_area_13p5pct_mm"] for r in rows_without_phase],
         marker="v", linestyle="--", label="D_area_13p5pct (without phase)")
plt.plot(z_without, [r["FWHM_mm"] for r in rows_without_phase],
         marker="P", linestyle="--", label="FWHM (without phase)")
"""

plt.xlabel("Propagation distance z [m]")
plt.ylabel("Beam size [mm]")
plt.title("Beam size versus propagation distance using 6 size definitions")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=9, ncol=2)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "beam_size_vs_z_6_definitions.png", dpi=180)
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


# -------------------------------------------------------------------------
# Plot-Daten separat speichern (direkt reproduzierbar für den PNG-Plot)
# -------------------------------------------------------------------------
plot_csv_path = OUTPUT_DIR / "beam_size_vs_z_plot_data.csv"

plot_rows = []
for i, z in enumerate(z_with):
    row = {
        "z_m": z,
        "D4sigma_avg_mm_with_phase": rows_with_phase[i]["D4sigma_avg_mm"],
        "D_EE50_mm_with_phase": rows_with_phase[i]["D_EE50_mm"],
        "D_EE86_mm_with_phase": rows_with_phase[i]["D_EE86_mm"],
        "D_area_50pct_mm_with_phase": rows_with_phase[i]["D_area_50pct_mm"],
        "D_area_13p5pct_mm_with_phase": rows_with_phase[i]["D_area_13p5pct_mm"],
        "FWHM_mm_with_phase": rows_with_phase[i]["FWHM_mm"],
        "D4sigma_avg_mm_without_phase": rows_without_phase[i]["D4sigma_avg_mm"],
        "D_EE50_mm_without_phase": rows_without_phase[i]["D_EE50_mm"],
        "D_EE86_mm_without_phase": rows_without_phase[i]["D_EE86_mm"],
        "D_area_50pct_mm_without_phase": rows_without_phase[i]["D_area_50pct_mm"],
        "D_area_13p5pct_mm_without_phase": rows_without_phase[i]["D_area_13p5pct_mm"],
        "FWHM_mm_without_phase": rows_without_phase[i]["FWHM_mm"],
    }
    plot_rows.append(row)

with open(plot_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(plot_rows[0].keys()))
    writer.writeheader()
    writer.writerows(plot_rows)

# optional zusätzlich als JSON
with open(OUTPUT_DIR / "beam_size_vs_z_plot_data.json", "w", encoding="utf-8") as f:
    json.dump(plot_rows, f, indent=2)