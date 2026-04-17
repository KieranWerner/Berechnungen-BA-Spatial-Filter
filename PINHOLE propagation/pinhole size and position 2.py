from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

CFG = {
    # --------------------------------------------------------
    # Input files
    # --------------------------------------------------------
    "PHA_FILE": Path(r"C:\Users\User\Desktop\combined PHA front csv\average_PHA no bg substract.csv"),
    "INT_FILE": Path(r"C:\Users\User\Desktop\combined average INT front bg substracted\average_INT front bg substracted.csv"),

    # Output
    "OUTPUT_DIR": Path(r"C:\Users\User\Desktop\PINHOLE_fast_sweep"),

    # Wavelength
    "WAVELENGTH_VACUUM_M": 800e-9,

    # --------------------------------------------------------
    # Initial field scaling
    # --------------------------------------------------------
    "USE_INPUT_D4SIGMA_FOR_SCALING": True,
    "INPUT_D4SIGMA_MM": 13.7,
    "THEORY_PIXEL_SIZE_UM": None,

    # --------------------------------------------------------
    # Input-field mask
    # --------------------------------------------------------
    "INTENSITY_THRESHOLD_REL": 0.02,
    "USE_CIRCLE_MASK": True,
    "CIRCLE_RADIUS_PX": 180,
    "CIRCLE_CENTER_X": None,
    "CIRCLE_CENTER_Y": None,
    "USE_SOFT_MASK": False,
    "SOFT_MASK_SIGMA_PX": 3.0,

    # --------------------------------------------------------
    # Phase
    # --------------------------------------------------------
    "USE_PHASE_MAP": True,
    "PHASE_SIGN": -1.0,

    # --------------------------------------------------------
    # Optical elements at z = 0
    # --------------------------------------------------------
    "APPLY_START_LENS_PHASE": True,
    "START_LENS_FOCAL_LENGTH_M": 5.0,
    "START_LENS_X_OFFSET_M": 0.0,
    "START_LENS_Y_OFFSET_M": 0.0,

    "APPLY_EXTRA_DEFOCUS_PHASE": False,
    "DEFOCUS_COEFF_M_INV": 0.0,

    "APPLY_TILT_PHASE": False,
    "TILT_X_RAD_PER_M": 0.0,
    "TILT_Y_RAD_PER_M": 0.0,

    "APPLY_ASTIGMATISM_PHASE": False,
    "ASTIG_X_M_INV": 0.0,
    "ASTIG_Y_M_INV": 0.0,

    # --------------------------------------------------------
    # Optional inserted lenses
    # --------------------------------------------------------
    "LENS1_ENABLED": False,
    "LENS1_Z_M": 0.0,
    "LENS1_FOCAL_LENGTH_M": 5.0,
    "LENS1_X_OFFSET_M": 0.0,
    "LENS1_Y_OFFSET_M": 0.0,

    "LENS2_ENABLED": True,
    "LENS2_Z_M": 13.0,
    "LENS2_FOCAL_LENGTH_M": 5.4,
    "LENS2_X_OFFSET_M": 0.0,
    "LENS2_Y_OFFSET_M": 0.0,

    # --------------------------------------------------------
    # Pinhole
    # --------------------------------------------------------
    "PINHOLE_ENABLED": True,
    "PINHOLE_X_OFFSET_M": 0.0,
    "PINHOLE_Y_OFFSET_M": 0.0,
    "PINHOLE_USE_SOFT_EDGE": False,
    "PINHOLE_SOFT_EDGE_SIGMA_M": 20e-6,

    # --------------------------------------------------------
    # 1) Search nominal pinhole plane WITHOUT pinhole
    # --------------------------------------------------------
    "PINHOLE_NOMINAL_SEARCH_Z_MIN_M": 4.5,
    "PINHOLE_NOMINAL_SEARCH_Z_MAX_M": 9.0,
    "PINHOLE_NOMINAL_SEARCH_NUM_POINTS": 20,

    # --------------------------------------------------------
    # 2) Local sweep around nominal plane
    # --------------------------------------------------------
    "PINHOLE_Z_LOCAL_HALF_RANGE_M": 3.00,
    "PINHOLE_Z_NUM_POINTS": 20,

    "PINHOLE_DIAMETER_MIN_M": 50e-6,
    "PINHOLE_DIAMETER_MAX_M": 2000e-6,
    "PINHOLE_DIAMETER_NUM_POINTS": 25,

    # --------------------------------------------------------
    # Evaluation after Lens 2
    # fast mode:
    # - either fixed plane
    # - or tiny local refinement
    # --------------------------------------------------------
    "USE_EVAL_MINI_SWEEP": False,
    "GAUSSIAN_QUALITY_AFTER_LENS2_DELTA_Z_M": 0.01,

    "POST_LENS2_EVAL_Z_MIN_REL_M": 0.006,
    "POST_LENS2_EVAL_Z_MAX_REL_M": 0.020,
    "POST_LENS2_EVAL_Z_NUM_POINTS": 5,

    # --------------------------------------------------------
    # Angular Spectrum
    # --------------------------------------------------------
    "PAD_FACTOR": 2,

    # --------------------------------------------------------
    # Optimization weights
    # --------------------------------------------------------
    "WEIGHT_OVERLAP": 0.60,
    "WEIGHT_RMSE": 0.20,
    "WEIGHT_TRANSMISSION": 0.15,
    "WEIGHT_ELLIPTICITY_PENALTY": 0.05,

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------
    "SAVE_BEST_INTENSITY_MAP": True,
    "PRINT_EVERY_N_STEPS": 1,
}


# ============================================================
# I/O
# ============================================================

def load_csv_auto(path: Path) -> np.ndarray:
    for delim in [",", ";", "\t"]:
        try:
            arr = np.loadtxt(path, delimiter=delim)
            if arr.ndim == 2 and arr.size > 0:
                return arr.astype(np.float64)
        except Exception:
            pass
    raise ValueError(f"Could not parse file: {path}")


# ============================================================
# HELPERS
# ============================================================

def timestamp() -> str:
    return time.strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{timestamp()}] {msg}", flush=True)


def apply_soft_mask(mask: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    try:
        from scipy.ndimage import gaussian_filter
        soft = gaussian_filter(mask.astype(float), sigma=sigma)
    except Exception:
        soft = mask.astype(float).copy()
        for _ in range(max(1, int(round(sigma)))):
            soft = (
                np.roll(soft, 1, 0) + np.roll(soft, -1, 0) +
                np.roll(soft, 1, 1) + np.roll(soft, -1, 1) +
                4.0 * soft
            ) / 8.0
    m = np.max(soft)
    if m > 0:
        soft /= m
    return soft


def make_valid_mask(
    intensity: np.ndarray,
    threshold_rel: float,
    use_circle: bool,
    circle_radius_px: float,
    cx: float | None,
    cy: float | None,
) -> tuple[np.ndarray, float, float, float]:
    intensity = np.asarray(intensity, dtype=float)
    thr = threshold_rel * float(np.nanmax(intensity))
    valid = np.isfinite(intensity) & (intensity > thr)

    ny, nx = intensity.shape
    Y, X = np.indices((ny, nx))

    if cx is None:
        cx = 0.5 * (nx - 1)
    if cy is None:
        cy = 0.5 * (ny - 1)

    if use_circle:
        circle = (X - cx) ** 2 + (Y - cy) ** 2 <= circle_radius_px ** 2
        valid &= circle

    return valid, float(thr), float(cx), float(cy)


def centroid_pixels(I: np.ndarray) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    I = np.clip(np.asarray(I, float), 0, None)
    P = float(I.sum())
    if P <= 0:
        raise ValueError("Intensity sum is zero.")
    y = np.arange(I.shape[0])
    x = np.arange(I.shape[1])
    X, Y = np.meshgrid(x, y)
    x0 = float((I * X).sum() / P)
    y0 = float((I * Y).sum() / P)
    return x0, y0, P, X, Y


def make_xy_grids(nx: int, ny: int, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    x = (np.arange(nx) - nx // 2) * dx
    y = (np.arange(ny) - ny // 2) * dy
    return np.meshgrid(x, y)


def safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    a = np.ravel(np.asarray(a, float))
    b = np.ravel(np.asarray(b, float))
    if a.size < 2 or b.size < 2:
        return np.nan
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa <= 0 or sb <= 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def profile_r2_against_gaussian(axis: np.ndarray, profile: np.ndarray) -> float:
    profile = np.clip(np.asarray(profile, float), 0.0, None)
    if profile.size < 3:
        return np.nan

    total = float(np.sum(profile))
    if total <= 0:
        return np.nan

    center = float(np.sum(axis * profile) / total)
    sigma = float(np.sqrt(np.sum(profile * (axis - center) ** 2) / total))
    if sigma <= 0:
        return np.nan

    w = 2.0 * sigma
    gshape = np.exp(-2.0 * ((axis - center) ** 2) / (w ** 2))

    denom = float(np.sum(gshape ** 2))
    if denom <= 0:
        return np.nan

    A = float(np.sum(profile * gshape) / denom)
    fit = A * gshape

    sse = float(np.sum((profile - fit) ** 2))
    sst = float(np.sum((profile - np.mean(profile)) ** 2))
    if sst <= 0:
        return np.nan

    return float(1.0 - sse / sst)


# ============================================================
# PROPAGATION
# ============================================================

def angular_spectrum(U0: np.ndarray, dx: float, dy: float, wavelength: float, z: float, pad_factor: int = 2) -> np.ndarray:
    ny, nx = U0.shape
    py = int((pad_factor - 1) * ny / 2)
    px = int((pad_factor - 1) * nx / 2)

    U = np.pad(U0, ((py, py), (px, px)), mode="constant")
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


def build_initial_field(intensity: np.ndarray, phase_waves: np.ndarray | None, valid_mask: np.ndarray) -> np.ndarray:
    intensity = np.asarray(intensity, dtype=float)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    if not np.any(valid_mask):
        raise ValueError("Valid mask is empty.")

    int_norm = np.zeros_like(intensity, dtype=float)
    int_norm[valid_mask] = intensity[valid_mask] / np.nanmax(intensity[valid_mask])

    A = np.sqrt(np.clip(int_norm, 0, None))

    if CFG["USE_SOFT_MASK"]:
        A *= apply_soft_mask(valid_mask, sigma=float(CFG["SOFT_MASK_SIGMA_PX"]))
    else:
        A[~valid_mask] = 0.0

    if CFG["USE_PHASE_MAP"]:
        if phase_waves is None:
            raise ValueError("USE_PHASE_MAP=True, but no phase file was provided.")
        phi = float(CFG["PHASE_SIGN"]) * 2.0 * np.pi * np.nan_to_num(phase_waves)
        U0 = A * np.exp(1j * phi)
    else:
        U0 = A.astype(np.complex128)

    return U0


def apply_start_plane_phase(U0: np.ndarray, dx: float, dy: float, wavelength: float) -> np.ndarray:
    ny, nx = U0.shape
    X, Y = make_xy_grids(nx, ny, dx, dy)

    k = 2.0 * np.pi / wavelength
    phi = np.zeros_like(X, dtype=float)

    if CFG["APPLY_START_LENS_PHASE"]:
        f = float(CFG["START_LENS_FOCAL_LENGTH_M"])
        x0 = float(CFG["START_LENS_X_OFFSET_M"])
        y0 = float(CFG["START_LENS_Y_OFFSET_M"])
        phi += -k * (((X - x0) ** 2 + (Y - y0) ** 2) / (2.0 * f))

    if CFG["APPLY_EXTRA_DEFOCUS_PHASE"]:
        phi += k * float(CFG["DEFOCUS_COEFF_M_INV"]) * (X**2 + Y**2)

    if CFG["APPLY_TILT_PHASE"]:
        phi += k * (
            float(CFG["TILT_X_RAD_PER_M"]) * X +
            float(CFG["TILT_Y_RAD_PER_M"]) * Y
        )

    if CFG["APPLY_ASTIGMATISM_PHASE"]:
        phi += k * (
            float(CFG["ASTIG_X_M_INV"]) * X**2 +
            float(CFG["ASTIG_Y_M_INV"]) * Y**2
        )

    return U0 * np.exp(1j * phi)


def apply_thin_lens(
    U: np.ndarray,
    dx: float,
    dy: float,
    wavelength: float,
    focal_length_m: float,
    x_offset_m: float = 0.0,
    y_offset_m: float = 0.0,
) -> np.ndarray:
    ny, nx = U.shape
    X, Y = make_xy_grids(nx, ny, dx, dy)
    k = 2.0 * np.pi / wavelength
    phi = -k * (((X - x_offset_m) ** 2 + (Y - y_offset_m) ** 2) / (2.0 * focal_length_m))
    return U * np.exp(1j * phi)


def apply_pinhole(
    U: np.ndarray,
    dx: float,
    dy: float,
    diameter_m: float,
    x_offset_m: float = 0.0,
    y_offset_m: float = 0.0,
    use_soft_edge: bool = False,
    soft_edge_sigma_m: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = U.shape
    X, Y = make_xy_grids(nx, ny, dx, dy)
    radius_m = 0.5 * diameter_m

    if use_soft_edge:
        R = np.sqrt((X - x_offset_m) ** 2 + (Y - y_offset_m) ** 2)
        if soft_edge_sigma_m <= 0:
            aperture = (R <= radius_m).astype(float)
        else:
            aperture = 0.5 * (1.0 - np.tanh((R - radius_m) / soft_edge_sigma_m))
    else:
        aperture = (((X - x_offset_m) ** 2 + (Y - y_offset_m) ** 2) <= radius_m**2).astype(float)

    return U * aperture, aperture


def build_elements_without_pinhole() -> list[dict]:
    elements: list[dict] = []

    if CFG["LENS1_ENABLED"]:
        elements.append({
            "type": "lens",
            "z_m": float(CFG["LENS1_Z_M"]),
            "focal_length_m": float(CFG["LENS1_FOCAL_LENGTH_M"]),
            "x_offset_m": float(CFG["LENS1_X_OFFSET_M"]),
            "y_offset_m": float(CFG["LENS1_Y_OFFSET_M"]),
        })

    if CFG["LENS2_ENABLED"]:
        elements.append({
            "type": "lens",
            "z_m": float(CFG["LENS2_Z_M"]),
            "focal_length_m": float(CFG["LENS2_FOCAL_LENGTH_M"]),
            "x_offset_m": float(CFG["LENS2_X_OFFSET_M"]),
            "y_offset_m": float(CFG["LENS2_Y_OFFSET_M"]),
        })

    elements.sort(key=lambda e: e["z_m"])
    return elements


def propagate_to_z_with_elements(
    U_start: np.ndarray,
    dx: float,
    dy: float,
    wavelength: float,
    z_target: float,
    elements: list[dict],
    pad_factor: int = 2,
) -> np.ndarray:
    if z_target < 0:
        raise ValueError("Negative propagation distances are not supported.")

    U = U_start.copy()
    z_current = 0.0
    tol = 1e-15

    for elem in elements:
        if abs(elem["z_m"] - 0.0) <= tol:
            U = apply_thin_lens(
                U, dx, dy, wavelength,
                focal_length_m=elem["focal_length_m"],
                x_offset_m=elem["x_offset_m"],
                y_offset_m=elem["y_offset_m"],
            )

    for elem in elements:
        z_elem = float(elem["z_m"])
        if z_elem <= tol:
            continue
        if z_elem > z_target + tol:
            break

        dz = z_elem - z_current
        if dz > tol:
            U = angular_spectrum(U, dx, dy, wavelength, dz, pad_factor=pad_factor)
            z_current = z_elem

        U = apply_thin_lens(
            U, dx, dy, wavelength,
            focal_length_m=elem["focal_length_m"],
            x_offset_m=elem["x_offset_m"],
            y_offset_m=elem["y_offset_m"],
        )

    dz_last = z_target - z_current
    if dz_last > tol:
        U = angular_spectrum(U, dx, dy, wavelength, dz_last, pad_factor=pad_factor)

    return U


def propagate_from_plane_to_z(
    U_plane: np.ndarray,
    dx: float,
    dy: float,
    wavelength: float,
    z_plane: float,
    z_target: float,
    elements: list[dict],
    pad_factor: int = 2,
) -> np.ndarray:
    if z_target < z_plane:
        raise ValueError("z_target must be >= z_plane.")

    U = U_plane.copy()
    z_current = z_plane
    tol = 1e-15

    for elem in elements:
        z_elem = float(elem["z_m"])
        if z_elem <= z_plane + tol:
            continue
        if z_elem > z_target + tol:
            break

        dz = z_elem - z_current
        if dz > tol:
            U = angular_spectrum(U, dx, dy, wavelength, dz, pad_factor=pad_factor)
            z_current = z_elem

        U = apply_thin_lens(
            U, dx, dy, wavelength,
            focal_length_m=elem["focal_length_m"],
            x_offset_m=elem["x_offset_m"],
            y_offset_m=elem["y_offset_m"],
        )

    dz_last = z_target - z_current
    if dz_last > tol:
        U = angular_spectrum(U, dx, dy, wavelength, dz_last, pad_factor=pad_factor)

    return U


# ============================================================
# METRICS
# ============================================================

def gaussian_quality_metrics(I: np.ndarray, dx: float, dy: float) -> dict[str, float]:
    I = np.clip(np.asarray(I, dtype=float), 0.0, None)

    power_sum = float(I.sum())
    if power_sum <= 0:
        return {
            "gaussian_overlap_2d": np.nan,
            "gaussian_rmse_rel_2d": np.nan,
            "gaussian_correlation_2d": np.nan,
            "gaussian_r2_integrated_avg": np.nan,
            "gaussian_wx": np.nan,
            "gaussian_wy": np.nan,
            "ellipticity": np.nan,
        }

    x0_px, y0_px, _, Xpx, Ypx = centroid_pixels(I)
    X = (Xpx - x0_px) * dx
    Y = (Ypx - y0_px) * dy

    sigma_x = float(np.sqrt((I * X**2).sum() / power_sum))
    sigma_y = float(np.sqrt((I * Y**2).sum() / power_sum))
    if sigma_x <= 0 or sigma_y <= 0:
        return {
            "gaussian_overlap_2d": np.nan,
            "gaussian_rmse_rel_2d": np.nan,
            "gaussian_correlation_2d": np.nan,
            "gaussian_r2_integrated_avg": np.nan,
            "gaussian_wx": np.nan,
            "gaussian_wy": np.nan,
            "ellipticity": np.nan,
        }

    wx = 2.0 * sigma_x
    wy = 2.0 * sigma_y

    Gshape = np.exp(-2.0 * ((X**2) / (wx**2) + (Y**2) / (wy**2)))

    denom = float((Gshape**2).sum())
    if denom <= 0:
        return {
            "gaussian_overlap_2d": np.nan,
            "gaussian_rmse_rel_2d": np.nan,
            "gaussian_correlation_2d": np.nan,
            "gaussian_r2_integrated_avg": np.nan,
            "gaussian_wx": np.nan,
            "gaussian_wy": np.nan,
            "ellipticity": np.nan,
        }

    A = float((I * Gshape).sum() / denom)
    Ifit = A * Gshape

    num = float((I * Ifit).sum()) ** 2
    den = float((I**2).sum()) * float((Ifit**2).sum())
    overlap_2d = num / den if den > 0 else np.nan

    rms_ref = float(np.sqrt(np.mean(I**2)))
    rmse_rel_2d = float(np.sqrt(np.mean((I - Ifit)**2)) / rms_ref) if rms_ref > 0 else np.nan

    corr_2d = safe_corrcoef(I, Ifit)

    x_axis = (np.arange(I.shape[1]) - x0_px) * dx
    y_axis = (np.arange(I.shape[0]) - y0_px) * dy
    profile_x = I.sum(axis=0)
    profile_y = I.sum(axis=1)

    r2_x = profile_r2_against_gaussian(x_axis, profile_x)
    r2_y = profile_r2_against_gaussian(y_axis, profile_y)
    r2_avg = np.nanmean([r2_x, r2_y])

    ellipticity = max(wx, wy) / min(wx, wy)

    return {
        "gaussian_overlap_2d": float(overlap_2d),
        "gaussian_rmse_rel_2d": float(rmse_rel_2d),
        "gaussian_correlation_2d": float(corr_2d),
        "gaussian_r2_integrated_avg": float(r2_avg),
        "gaussian_wx": float(wx),
        "gaussian_wy": float(wy),
        "ellipticity": float(ellipticity),
    }


def beam_d4sigma(I: np.ndarray, dx: float, dy: float) -> tuple[float, float]:
    I = np.clip(np.asarray(I, float), 0, None)
    P = float(I.sum())
    if P <= 0:
        return np.nan, np.nan

    x0_px, y0_px, _, Xpx, Ypx = centroid_pixels(I)
    X = (Xpx - x0_px) * dx
    Y = (Ypx - y0_px) * dy

    sigma_x = float(np.sqrt((I * X**2).sum() / P))
    sigma_y = float(np.sqrt((I * Y**2).sum() / P))
    return 4.0 * sigma_x, 4.0 * sigma_y


def compute_score(
    overlap: float,
    rmse_rel: float,
    transmission: float,
    wx: float,
    wy: float,
) -> float:
    if not all(np.isfinite(v) for v in [overlap, rmse_rel, transmission, wx, wy]):
        return np.nan
    if wx <= 0 or wy <= 0:
        return np.nan

    ellip_pen = abs(np.log(wx / wy))

    score = (
        float(CFG["WEIGHT_OVERLAP"]) * overlap
        + float(CFG["WEIGHT_RMSE"]) * (1.0 - rmse_rel)
        + float(CFG["WEIGHT_TRANSMISSION"]) * transmission
        - float(CFG["WEIGHT_ELLIPTICITY_PENALTY"]) * ellip_pen
    )
    return float(score)


def determine_dx_dy_m(intensity: np.ndarray, valid_mask: np.ndarray) -> tuple[float, float]:
    if CFG["USE_INPUT_D4SIGMA_FOR_SCALING"]:
        int_norm = np.zeros_like(intensity, dtype=float)
        int_norm[valid_mask] = intensity[valid_mask] / np.nanmax(intensity[valid_mask])

        x0_px, y0_px, P, X, Y = centroid_pixels(int_norm)
        Xc = X - x0_px
        Yc = Y - y0_px
        sigma_x_px = float(np.sqrt((int_norm * Xc**2).sum() / P))
        sigma_y_px = float(np.sqrt((int_norm * Yc**2).sum() / P))
        d4_px = 0.5 * (4.0 * sigma_x_px + 4.0 * sigma_y_px)

        if not np.isfinite(d4_px) or d4_px <= 0:
            raise RuntimeError("Could not determine D4sigma in pixels for scaling.")

        dx_m = (float(CFG["INPUT_D4SIGMA_MM"]) / d4_px) * 1e-3
        dy_m = dx_m
        return dx_m, dy_m

    pitch_um = CFG["THEORY_PIXEL_SIZE_UM"]
    if pitch_um is None:
        raise RuntimeError(
            "Please either set USE_INPUT_D4SIGMA_FOR_SCALING=True "
            "or specify THEORY_PIXEL_SIZE_UM."
        )

    dx_m = float(pitch_um) * 1e-6
    dy_m = dx_m
    return dx_m, dy_m


# ============================================================
# PLOTTING
# ============================================================

def save_metric_heatmap(
    X_mm: np.ndarray,
    Y_um: np.ndarray,
    Z: np.ndarray,
    title: str,
    colorbar_label: str,
    out_png: Path,
    out_pdf: Path,
) -> None:
    plt.figure(figsize=(9, 7))
    plt.imshow(
        Z,
        origin="lower",
        aspect="auto",
        extent=[X_mm.min(), X_mm.max(), Y_um.min(), Y_um.max()],
        cmap="viridis_r",
    )
    plt.xlabel("Pinhole position z [mm]")
    plt.ylabel("Pinhole diameter [µm]")
    plt.title(title)
    plt.colorbar(label=colorbar_label)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.savefig(out_pdf)
    plt.close()


def save_line_plot(
    x,
    y,
    xlabel: str,
    ylabel: str,
    title: str,
    out_png: Path,
    out_pdf: Path,
) -> None:
    plt.figure(figsize=(8.5, 5.5))
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.savefig(out_pdf)
    plt.close()


# ============================================================
# NOMINAL PINHOLE PLANE
# ============================================================

def find_nominal_pinhole_plane(
    U0: np.ndarray,
    dx_m: float,
    dy_m: float,
    wavelength: float,
    pad_factor: int,
) -> tuple[float, list[dict]]:
    log("Suche nominale Pinhole-Ebene ohne Pinhole ...")

    elements = build_elements_without_pinhole()
    z_values = np.linspace(
        float(CFG["PINHOLE_NOMINAL_SEARCH_Z_MIN_M"]),
        float(CFG["PINHOLE_NOMINAL_SEARCH_Z_MAX_M"]),
        int(CFG["PINHOLE_NOMINAL_SEARCH_NUM_POINTS"]),
    )

    scan_rows: list[dict] = []
    best_metric = np.inf
    best_z = None

    n_total = len(z_values)
    t0 = time.perf_counter()

    for i, z in enumerate(z_values, start=1):
        Uz = propagate_to_z_with_elements(
            U_start=U0,
            dx=dx_m,
            dy=dy_m,
            wavelength=wavelength,
            z_target=float(z),
            elements=elements,
            pad_factor=pad_factor,
        )
        Iz = np.abs(Uz) ** 2
        d4x, d4y = beam_d4sigma(Iz, dx_m, dy_m)
        area_metric = d4x * d4y if np.isfinite(d4x) and np.isfinite(d4y) else np.inf

        scan_rows.append({
            "z_m": float(z),
            "d4sigma_x_mm": float(d4x * 1e3) if np.isfinite(d4x) else np.nan,
            "d4sigma_y_mm": float(d4y * 1e3) if np.isfinite(d4y) else np.nan,
            "spot_area_metric_mm2": float(area_metric * 1e6) if np.isfinite(area_metric) else np.nan,
        })

        if np.isfinite(area_metric) and area_metric < best_metric:
            best_metric = area_metric
            best_z = float(z)

        if i % max(1, int(CFG["PRINT_EVERY_N_STEPS"])) == 0 or i == n_total:
            elapsed = time.perf_counter() - t0
            log(f"Nominalsuche: {i}/{n_total}  z={z:.4f} m")

    if best_z is None:
        raise RuntimeError("Could not determine nominal pinhole plane.")

    log(f"Nominale Pinhole-Ebene gefunden bei z = {best_z:.6f} m")
    return best_z, scan_rows


# ============================================================
# MAIN FAST SWEEP
# ============================================================

def fast_pinhole_sweep() -> None:
    total_t0 = time.perf_counter()

    output_dir = Path(CFG["OUTPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)

    log("Lade Eingangsdaten ...")
    phase_waves = load_csv_auto(Path(CFG["PHA_FILE"])) if CFG["USE_PHASE_MAP"] else None
    intensity = load_csv_auto(Path(CFG["INT_FILE"]))

    if phase_waves is not None and phase_waves.shape != intensity.shape:
        raise ValueError(f"Shape mismatch: phase {phase_waves.shape} vs intensity {intensity.shape}")

    log("Erzeuge Maske ...")
    valid_mask, threshold_abs, cx, cy = make_valid_mask(
        intensity=intensity,
        threshold_rel=float(CFG["INTENSITY_THRESHOLD_REL"]),
        use_circle=bool(CFG["USE_CIRCLE_MASK"]),
        circle_radius_px=float(CFG["CIRCLE_RADIUS_PX"]),
        cx=CFG["CIRCLE_CENTER_X"],
        cy=CFG["CIRCLE_CENTER_Y"],
    )

    log("Bestimme Pixelgroesse ...")
    dx_m, dy_m = determine_dx_dy_m(intensity, valid_mask)
    wavelength = float(CFG["WAVELENGTH_VACUUM_M"])
    pad_factor = int(CFG["PAD_FACTOR"])

    if not CFG["PINHOLE_ENABLED"]:
        raise ValueError("PINHOLE_ENABLED must be True.")
    if not CFG["LENS2_ENABLED"]:
        raise ValueError("LENS2_ENABLED must be True.")

    log("Baue Startfeld ...")
    U0 = build_initial_field(
        intensity=intensity,
        phase_waves=phase_waves,
        valid_mask=valid_mask,
    )
    U0 = apply_start_plane_phase(U0, dx_m, dy_m, wavelength)

    # 1) nominal plane
    nominal_z, nominal_scan_rows = find_nominal_pinhole_plane(
        U0=U0,
        dx_m=dx_m,
        dy_m=dy_m,
        wavelength=wavelength,
        pad_factor=pad_factor,
    )

    nominal_csv = output_dir / "nominal_pinhole_plane_scan.csv"
    with open(nominal_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(nominal_scan_rows[0].keys()))
        writer.writeheader()
        writer.writerows(nominal_scan_rows)

    z_nom_scan_mm = np.array([r["z_m"] for r in nominal_scan_rows]) * 1e3
    area_metric_mm2 = np.array([r["spot_area_metric_mm2"] for r in nominal_scan_rows], dtype=float)
    save_line_plot(
        x=z_nom_scan_mm,
        y=area_metric_mm2,
        xlabel="z [mm]",
        ylabel="D4σ area metric [mm²]",
        title="Nominal pinhole plane search without pinhole",
        out_png=output_dir / "nominal_pinhole_plane_search.png",
        out_pdf=output_dir / "nominal_pinhole_plane_search.pdf",
    )

    # 2) local sweep
    pinhole_z_values = np.linspace(
        nominal_z - float(CFG["PINHOLE_Z_LOCAL_HALF_RANGE_M"]),
        nominal_z + float(CFG["PINHOLE_Z_LOCAL_HALF_RANGE_M"]),
        int(CFG["PINHOLE_Z_NUM_POINTS"]),
    )

    pinhole_d_values = np.linspace(
        float(CFG["PINHOLE_DIAMETER_MIN_M"]),
        float(CFG["PINHOLE_DIAMETER_MAX_M"]),
        int(CFG["PINHOLE_DIAMETER_NUM_POINTS"]),
    )

    if CFG["USE_EVAL_MINI_SWEEP"]:
        z_eval_values = np.linspace(
            float(CFG["LENS2_Z_M"]) + float(CFG["POST_LENS2_EVAL_Z_MIN_REL_M"]),
            float(CFG["LENS2_Z_M"]) + float(CFG["POST_LENS2_EVAL_Z_MAX_REL_M"]),
            int(CFG["POST_LENS2_EVAL_Z_NUM_POINTS"]),
        )
        log(f"Benutze kleinen z_eval-Sweep mit {len(z_eval_values)} Punkten.")
    else:
        z_eval_values = np.array([
            float(CFG["LENS2_Z_M"]) + float(CFG["GAUSSIAN_QUALITY_AFTER_LENS2_DELTA_Z_M"])
        ])
        log(f"Benutze feste Auswerteebene z = {z_eval_values[0]:.6f} m")

    nz = len(pinhole_z_values)
    nd = len(pinhole_d_values)
    n_eval = len(z_eval_values)

    overlap_map = np.full((nd, nz), np.nan, dtype=float)
    rmse_map = np.full((nd, nz), np.nan, dtype=float)
    corr_map = np.full((nd, nz), np.nan, dtype=float)
    r2avg_map = np.full((nd, nz), np.nan, dtype=float)
    trans_map = np.full((nd, nz), np.nan, dtype=float)
    ellip_map = np.full((nd, nz), np.nan, dtype=float)
    score_map = np.full((nd, nz), np.nan, dtype=float)
    best_eval_z_map = np.full((nd, nz), np.nan, dtype=float)

    rows: list[dict] = []

    best_score = -np.inf
    best_result: dict | None = None
    best_intensity = None

    elements = build_elements_without_pinhole()

    total_cases = nz * nd
    case_counter = 0
    sweep_t0 = time.perf_counter()

    log("Starte schnellen lokalen Pinhole-Sweep ...")

    for i_z, pinhole_z in enumerate(pinhole_z_values):
        log(f"Propagiere einmal bis zur Pinhole-Ebene z = {pinhole_z:.6f} m ...")

        U_before_pinhole = propagate_to_z_with_elements(
            U_start=U0,
            dx=dx_m,
            dy=dy_m,
            wavelength=wavelength,
            z_target=float(pinhole_z),
            elements=elements,
            pad_factor=pad_factor,
        )

        power_before = float(np.sum(np.abs(U_before_pinhole) ** 2))

        for i_d, pinhole_d in enumerate(pinhole_d_values):
            case_counter += 1

            U_after_pinhole, _ = apply_pinhole(
                U_before_pinhole,
                dx_m,
                dy_m,
                diameter_m=float(pinhole_d),
                x_offset_m=float(CFG["PINHOLE_X_OFFSET_M"]),
                y_offset_m=float(CFG["PINHOLE_Y_OFFSET_M"]),
                use_soft_edge=bool(CFG["PINHOLE_USE_SOFT_EDGE"]),
                soft_edge_sigma_m=float(CFG["PINHOLE_SOFT_EDGE_SIGMA_M"]),
            )

            power_after = float(np.sum(np.abs(U_after_pinhole) ** 2))
            transmission = power_after / power_before if power_before > 0 else np.nan

            best_local_score = -np.inf
            best_local = None
            best_local_I = None

            for z_eval in z_eval_values:
                U_eval = propagate_from_plane_to_z(
                    U_plane=U_after_pinhole,
                    dx=dx_m,
                    dy=dy_m,
                    wavelength=wavelength,
                    z_plane=float(pinhole_z),
                    z_target=float(z_eval),
                    elements=elements,
                    pad_factor=pad_factor,
                )

                I_eval = np.abs(U_eval) ** 2
                gq = gaussian_quality_metrics(I_eval, dx_m, dy_m)

                score = compute_score(
                    overlap=gq["gaussian_overlap_2d"],
                    rmse_rel=gq["gaussian_rmse_rel_2d"],
                    transmission=transmission,
                    wx=gq["gaussian_wx"],
                    wy=gq["gaussian_wy"],
                )

                row = {
                    "nominal_pinhole_z_m": float(nominal_z),
                    "pinhole_z_m": float(pinhole_z),
                    "pinhole_diameter_m": float(pinhole_d),
                    "evaluation_z_m": float(z_eval),
                    "gaussian_overlap_2d": float(gq["gaussian_overlap_2d"]),
                    "gaussian_rmse_rel_2d": float(gq["gaussian_rmse_rel_2d"]),
                    "gaussian_correlation_2d": float(gq["gaussian_correlation_2d"]),
                    "gaussian_r2_integrated_avg": float(gq["gaussian_r2_integrated_avg"]),
                    "transmission": float(transmission),
                    "ellipticity": float(gq["ellipticity"]),
                    "gaussian_wx_mm": float(gq["gaussian_wx"] * 1e3),
                    "gaussian_wy_mm": float(gq["gaussian_wy"] * 1e3),
                    "score": float(score),
                }
                rows.append(row)

                if np.isfinite(score) and score > best_local_score:
                    best_local_score = score
                    best_local = row
                    best_local_I = I_eval.copy()

            if best_local is not None:
                overlap_map[i_d, i_z] = best_local["gaussian_overlap_2d"]
                rmse_map[i_d, i_z] = best_local["gaussian_rmse_rel_2d"]
                corr_map[i_d, i_z] = best_local["gaussian_correlation_2d"]
                r2avg_map[i_d, i_z] = best_local["gaussian_r2_integrated_avg"]
                trans_map[i_d, i_z] = best_local["transmission"]
                ellip_map[i_d, i_z] = best_local["ellipticity"]
                score_map[i_d, i_z] = best_local["score"]
                best_eval_z_map[i_d, i_z] = best_local["evaluation_z_m"] * 1e3

                if np.isfinite(best_local_score) and best_local_score > best_score:
                    best_score = best_local_score
                    best_result = best_local
                    best_intensity = best_local_I

            if case_counter % max(1, int(CFG["PRINT_EVERY_N_STEPS"])) == 0 or case_counter == total_cases:
                elapsed = time.perf_counter() - sweep_t0
                rate = case_counter / elapsed if elapsed > 0 else 0.0
                remaining = (total_cases - case_counter) / rate if rate > 0 else np.nan

                msg = (
                    f"Sweep: {case_counter}/{total_cases}  "
                    f"z={pinhole_z:.6f} m, d={pinhole_d * 1e6:.1f} um, "
                    f"trans={transmission:.4f}"
                )
                if best_local is not None:
                    msg += f", score={best_local['score']:.5f}"
                if np.isfinite(remaining):
                    msg += f", ETA={remaining:.1f} s"
                log(msg)

    # save numeric results
    csv_path = output_dir / "pinhole_size_position_vs_quality_fast.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_dir / "pinhole_size_position_vs_quality_fast.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    # plots
    z_mm = pinhole_z_values * 1e3
    d_um = pinhole_d_values * 1e6

    save_metric_heatmap(
        X_mm=z_mm,
        Y_um=d_um,
        Z=score_map,
        title="Combined score",
        colorbar_label="score",
        out_png=output_dir / "combined_score_vs_pinhole_size_position.png",
        out_pdf=output_dir / "combined_score_vs_pinhole_size_position.pdf",
    )
    save_metric_heatmap(
        X_mm=z_mm,
        Y_um=d_um,
        Z=overlap_map,
        title="Gaussian overlap (2D)",
        colorbar_label="overlap",
        out_png=output_dir / "gaussian_overlap_2d_vs_pinhole_size_position.png",
        out_pdf=output_dir / "gaussian_overlap_2d_vs_pinhole_size_position.pdf",
    )
    save_metric_heatmap(
        X_mm=z_mm,
        Y_um=d_um,
        Z=rmse_map,
        title="Relative Gaussian RMSE (2D)",
        colorbar_label="relative RMSE",
        out_png=output_dir / "gaussian_rmse_rel_2d_vs_pinhole_size_position.png",
        out_pdf=output_dir / "gaussian_rmse_rel_2d_vs_pinhole_size_position.pdf",
    )
    save_metric_heatmap(
        X_mm=z_mm,
        Y_um=d_um,
        Z=trans_map,
        title="Transmission",
        colorbar_label="transmission",
        out_png=output_dir / "transmission_vs_pinhole_size_position.png",
        out_pdf=output_dir / "transmission_vs_pinhole_size_position.pdf",
    )
    save_metric_heatmap(
        X_mm=z_mm,
        Y_um=d_um,
        Z=ellip_map,
        title="Ellipticity",
        colorbar_label="ellipticity",
        out_png=output_dir / "ellipticity_vs_pinhole_size_position.png",
        out_pdf=output_dir / "ellipticity_vs_pinhole_size_position.pdf",
    )

    if CFG["USE_EVAL_MINI_SWEEP"]:
        save_metric_heatmap(
            X_mm=z_mm,
            Y_um=d_um,
            Z=best_eval_z_map,
            title="Best evaluation z after Lens 2",
            colorbar_label="best eval z [mm]",
            out_png=output_dir / "best_eval_z_after_lens2_vs_pinhole_size_position.png",
            out_pdf=output_dir / "best_eval_z_after_lens2_vs_pinhole_size_position.pdf",
        )

    # best intensity map
    if CFG["SAVE_BEST_INTENSITY_MAP"] and best_intensity is not None and best_result is not None:
        Iz = best_intensity / max(np.max(best_intensity), 1e-12)

        plt.figure(figsize=(6.5, 5.5))
        extent = [
            -0.5 * Iz.shape[1] * dx_m * 1e3,
             0.5 * Iz.shape[1] * dx_m * 1e3,
            -0.5 * Iz.shape[0] * dy_m * 1e3,
             0.5 * Iz.shape[0] * dy_m * 1e3,
        ]
        plt.imshow(Iz, origin="lower", extent=extent, aspect="equal")
        plt.xlabel("x [mm]")
        plt.ylabel("y [mm]")
        plt.title(
            "Best result\n"
            f"pinhole z = {best_result['pinhole_z_m']:.6f} m, "
            f"d = {best_result['pinhole_diameter_m'] * 1e6:.2f} µm\n"
            f"eval z = {best_result['evaluation_z_m']:.6f} m"
        )
        plt.colorbar(label="Normalized intensity")
        plt.tight_layout()
        plt.savefig(output_dir / "best_result_intensity_map.png", dpi=220)
        plt.savefig(output_dir / "best_result_intensity_map.pdf")
        plt.close()

    # metadata
    meta = {
        "wavelength_nm": wavelength * 1e9,
        "dx_um": dx_m * 1e6,
        "dy_um": dy_m * 1e6,
        "threshold_abs": threshold_abs,
        "valid_pixels": int(valid_mask.sum()),
        "mask_center_x_px": cx,
        "mask_center_y_px": cy,
        "nominal_pinhole_z_m": nominal_z,
        "pinhole_z_num_points": CFG["PINHOLE_Z_NUM_POINTS"],
        "pinhole_diameter_num_points": CFG["PINHOLE_DIAMETER_NUM_POINTS"],
        "use_eval_mini_sweep": CFG["USE_EVAL_MINI_SWEEP"],
        "pad_factor": CFG["PAD_FACTOR"],
        "optimization_metric": "combined_score",
        "best_result": best_result,
    }
    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    total_elapsed = time.perf_counter() - total_t0

    print()
    print("=" * 80)
    print("FAST PINHOLE SWEEP FINISHED")
    print("=" * 80)
    print(f"Output directory:                    {output_dir}")
    print(f"Pixel size in theory:                {dx_m * 1e6:.6f} um")
    print(f"Nominal pinhole plane:               {nominal_z:.6f} m")
    print(f"Position points:                     {CFG['PINHOLE_Z_NUM_POINTS']}")
    print(f"Diameter points:                     {CFG['PINHOLE_DIAMETER_NUM_POINTS']}")
    print(f"Eval points per configuration:       {len(z_eval_values)}")
    print(f"CSV saved to:                        {csv_path}")
    print(f"JSON saved to:                       {json_path}")
    print(f"Total runtime:                       {total_elapsed:.2f} s")
    print()

    if best_result is None:
        print("No valid optimum could be determined.")
    else:
        print("Optimal result based on combined score:")
        print(f"  Nominal pinhole plane:             {best_result['nominal_pinhole_z_m']:.6f} m")
        print(f"  Optimal pinhole position:          {best_result['pinhole_z_m']:.6f} m")
        print(f"  Optimal pinhole diameter:          {best_result['pinhole_diameter_m'] * 1e6:.3f} um")
        print(f"  Evaluation plane after Lens 2:     {best_result['evaluation_z_m']:.6f} m")
        print(f"  Combined score:                    {best_result['score']:.6f}")
        print(f"  Gaussian overlap (2D):             {best_result['gaussian_overlap_2d']:.6f}")
        print(f"  Relative RMSE (2D):                {best_result['gaussian_rmse_rel_2d']:.6f}")
        print(f"  Gaussian correlation (2D):         {best_result['gaussian_correlation_2d']:.6f}")
        print(f"  Gaussian R² integrated avg:        {best_result['gaussian_r2_integrated_avg']:.6f}")
        print(f"  Transmission:                      {best_result['transmission']:.6f}")
        print(f"  Ellipticity:                       {best_result['ellipticity']:.6f}")
        print(f"  Gaussian wx:                       {best_result['gaussian_wx_mm']:.6f} mm")
        print(f"  Gaussian wy:                       {best_result['gaussian_wy_mm']:.6f} mm")


def main() -> None:
    fast_pinhole_sweep()


if __name__ == "__main__":
    main()