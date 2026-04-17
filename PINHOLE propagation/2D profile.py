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
    "OUTPUT_DIR": Path(r"C:\Users\User\Desktop\PINHOLE_fixed_profiles"),

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
    "APPLY_START_LENS_PHASE": False,
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

    "LENS2_ENABLED": False,
    "LENS2_Z_M": 13.0,
    "LENS2_FOCAL_LENGTH_M": 5.4,
    "LENS2_X_OFFSET_M": 0.0,
    "LENS2_Y_OFFSET_M": 0.0,

    # --------------------------------------------------------
    # Fixed pinhole instead of optimization
    # --------------------------------------------------------
    "PINHOLE_ENABLED": False,
    "PINHOLE_FIXED_Z_M": 7.5658,
    "PINHOLE_FIXED_DIAMETER_M": 1106e-6,
    "PINHOLE_X_OFFSET_M": 0.0,
    "PINHOLE_Y_OFFSET_M": 0.0,
    "PINHOLE_USE_SOFT_EDGE": False,
    "PINHOLE_SOFT_EDGE_SIGMA_M": 20e-6,

    # --------------------------------------------------------
    # Adjustable output positions for radial profiles
    # absolute z positions in meters
    # --------------------------------------------------------
    "PROFILE_Z_POSITIONS_M": [
        1.00,
        4.00,
        6.00,
        8.01,
        13.02,
    ],

    # --------------------------------------------------------
    # z sweep for beam size versus z
    # absolute z positions in meters
    # --------------------------------------------------------
    "SIZE_VS_Z_MIN_M": 0.0,
    "SIZE_VS_Z_MAX_M": 13.10,
    "SIZE_VS_Z_NUM_POINTS": 80,

    # --------------------------------------------------------
    # Angular Spectrum
    # --------------------------------------------------------
    "PAD_FACTOR": 2,

    # --------------------------------------------------------
    # Radial profile settings
    # --------------------------------------------------------
    "RADIAL_PROFILE_NUM_BINS": 250,
    "RADIAL_PROFILE_USE_CENTROID": True,
    "SAVE_INTENSITY_MAPS_AT_PROFILE_Z": True,
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



def radial_profile(I: np.ndarray, dx: float, dy: float, num_bins: int, use_centroid: bool = True) -> tuple[np.ndarray, np.ndarray]:
    I = np.clip(np.asarray(I, float), 0.0, None)
    ny, nx = I.shape

    if use_centroid and float(I.sum()) > 0:
        x0_px, y0_px, _, Xpx, Ypx = centroid_pixels(I)
    else:
        x = np.arange(nx)
        y = np.arange(ny)
        Xpx, Ypx = np.meshgrid(x, y)
        x0_px = 0.5 * (nx - 1)
        y0_px = 0.5 * (ny - 1)

    X = (Xpx - x0_px) * dx
    Y = (Ypx - y0_px) * dy
    R = np.sqrt(X**2 + Y**2)

    r_max = float(np.max(R))
    edges = np.linspace(0.0, r_max, int(num_bins) + 1)
    idx = np.digitize(R.ravel(), edges) - 1
    idx = np.clip(idx, 0, num_bins - 1)

    sums = np.bincount(idx, weights=I.ravel(), minlength=num_bins)
    counts = np.bincount(idx, minlength=num_bins)

    with np.errstate(invalid="ignore", divide="ignore"):
        prof = sums / counts
    prof = np.nan_to_num(prof, nan=0.0, posinf=0.0, neginf=0.0)

    r_centers = 0.5 * (edges[:-1] + edges[1:])
    return r_centers, prof


# ============================================================
# PLOTTING
# ============================================================


def save_line_plot(x, y, xlabel: str, ylabel: str, title: str, out_png: Path, out_pdf: Path, labels: list[str] | None = None) -> None:
    plt.figure(figsize=(8.5, 5.5))

    if isinstance(y, (list, tuple)) and len(y) > 0 and np.ndim(y[0]) > 0:
        for i, yi in enumerate(y):
            label = labels[i] if labels is not None and i < len(labels) else None
            plt.plot(x, yi, label=label)
        if labels is not None:
            plt.legend()
    else:
        plt.plot(x, y)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.savefig(out_pdf)
    plt.close()



def save_heatmap(X_mm: np.ndarray, Y_mm: np.ndarray, Z: np.ndarray, xlabel: str, ylabel: str, title: str, colorbar_label: str, out_png: Path, out_pdf: Path) -> None:
    plt.figure(figsize=(9, 7))
    plt.imshow(
        Z,
        origin="lower",
        aspect="auto",
        extent=[X_mm.min(), X_mm.max(), Y_mm.min(), Y_mm.max()],
        cmap="viridis",
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.colorbar(label=colorbar_label)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.savefig(out_pdf)
    plt.close()



def save_intensity_map(I: np.ndarray, dx_m: float, dy_m: float, title: str, out_png: Path, out_pdf: Path) -> None:
    I = np.asarray(I, float)
    In = I / max(float(np.max(I)), 1e-12)
    extent = [
        -0.5 * I.shape[1] * dx_m * 1e3,
         0.5 * I.shape[1] * dx_m * 1e3,
        -0.5 * I.shape[0] * dy_m * 1e3,
         0.5 * I.shape[0] * dy_m * 1e3,
    ]

    plt.figure(figsize=(6.8, 5.8))
    plt.imshow(In, origin="lower", extent=extent, aspect="equal")
    plt.xlabel("x [mm]")
    plt.ylabel("y [mm]")
    plt.title(title)
    plt.colorbar(label="Normierte Intensität")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.savefig(out_pdf)
    plt.close()


# ============================================================
# MAIN
# ============================================================


def fixed_pinhole_profiles() -> None:
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

    log("Baue Startfeld ...")
    U0 = build_initial_field(
        intensity=intensity,
        phase_waves=phase_waves,
        valid_mask=valid_mask,
    )
    U0 = apply_start_plane_phase(U0, dx_m, dy_m, wavelength)

    elements = build_elements_without_pinhole()

    pinhole_z = float(CFG["PINHOLE_FIXED_Z_M"])
    pinhole_d = float(CFG["PINHOLE_FIXED_DIAMETER_M"])

    if pinhole_z < 0:
        raise ValueError("PINHOLE_FIXED_Z_M must be >= 0.")

    profile_z_values = np.array(sorted(float(z) for z in CFG["PROFILE_Z_POSITIONS_M"]))

    z_size_values = np.linspace(
        float(CFG["SIZE_VS_Z_MIN_M"]),
        float(CFG["SIZE_VS_Z_MAX_M"]),
        int(CFG["SIZE_VS_Z_NUM_POINTS"]),
    )
    if np.any(z_size_values < 0.0):
        raise ValueError("SIZE_VS_Z_MIN_M must be >= 0.")

    log(f"Propagiere bis zur festen Pinhole-Ebene z = {pinhole_z:.6f} m ...")
    U_before_pinhole = propagate_to_z_with_elements(
        U_start=U0,
        dx=dx_m,
        dy=dy_m,
        wavelength=wavelength,
        z_target=pinhole_z,
        elements=elements,
        pad_factor=pad_factor,
    )

    U_after_pinhole, aperture = apply_pinhole(
        U_before_pinhole,
        dx_m,
        dy_m,
        diameter_m=pinhole_d,
        x_offset_m=float(CFG["PINHOLE_X_OFFSET_M"]),
        y_offset_m=float(CFG["PINHOLE_Y_OFFSET_M"]),
        use_soft_edge=bool(CFG["PINHOLE_USE_SOFT_EDGE"]),
        soft_edge_sigma_m=float(CFG["PINHOLE_SOFT_EDGE_SIGMA_M"]),
    )

    power_before = float(np.sum(np.abs(U_before_pinhole) ** 2))
    power_after = float(np.sum(np.abs(U_after_pinhole) ** 2))
    transmission = power_after / power_before if power_before > 0 else np.nan

    save_intensity_map(
        np.abs(U_before_pinhole) ** 2,
        dx_m,
        dy_m,
        title=f"Intensität vor Pinhole bei z = {pinhole_z:.6f} m",
        out_png=output_dir / "intensity_before_pinhole.png",
        out_pdf=output_dir / "intensity_before_pinhole.pdf",
    )
    save_intensity_map(
        aperture,
        dx_m,
        dy_m,
        title=f"Pinhole-Apertur, Durchmesser = {pinhole_d * 1e6:.2f} µm",
        out_png=output_dir / "pinhole_aperture.png",
        out_pdf=output_dir / "pinhole_aperture.pdf",
    )
    save_intensity_map(
        np.abs(U_after_pinhole) ** 2,
        dx_m,
        dy_m,
        title=f"Intensität direkt nach Pinhole bei z = {pinhole_z:.6f} m",
        out_png=output_dir / "intensity_after_pinhole.png",
        out_pdf=output_dir / "intensity_after_pinhole.pdf",
    )

    radial_rows: list[dict] = []
    size_rows: list[dict] = []
    radial_profiles = []
    radial_labels = []
    r_mm_reference = None

    log("Berechne radiale Profile an einstellbaren z-Positionen ...")
    for z_eval in profile_z_values:
        if z_eval < pinhole_z:
            U_eval = propagate_to_z_with_elements(
                U_start=U0,
                dx=dx_m,
                dy=dy_m,
                wavelength=wavelength,
                z_target=float(z_eval),
                elements=elements,
                pad_factor=pad_factor,
            )
        else:
            U_eval = propagate_from_plane_to_z(
                U_plane=U_after_pinhole,
                dx=dx_m,
                dy=dy_m,
                wavelength=wavelength,
                z_plane=pinhole_z,
                z_target=float(z_eval),
                elements=elements,
                pad_factor=pad_factor,
            )
        I_eval = np.abs(U_eval) ** 2
        r_m, prof = radial_profile(
            I_eval,
            dx=dx_m,
            dy=dy_m,
            num_bins=int(CFG["RADIAL_PROFILE_NUM_BINS"]),
            use_centroid=bool(CFG["RADIAL_PROFILE_USE_CENTROID"]),
        )

        prof_norm = prof / max(float(np.max(prof)), 1e-12)
        radial_profiles.append(prof_norm)
        radial_labels.append(f"z = {z_eval:.4f} m")
        r_mm_reference = r_m * 1e3

        for r_val, p_val, p_norm in zip(r_m, prof, prof_norm):
            radial_rows.append({
                "z_m": float(z_eval),
                "radius_m": float(r_val),
                "radius_mm": float(r_val * 1e3),
                "radial_intensity": float(p_val),
                "radial_intensity_norm": float(p_norm),
            })

        if CFG["SAVE_INTENSITY_MAPS_AT_PROFILE_Z"]:
            stem = f"intensity_map_z_{z_eval:.6f}_m".replace(".", "p")
            save_intensity_map(
                I_eval,
                dx_m,
                dy_m,
                title=f"2D-Profil / Intensitätskarte bei z = {z_eval:.6f} m",
                out_png=output_dir / f"{stem}.png",
                out_pdf=output_dir / f"{stem}.pdf",
            )

        radial_stem = f"radial_profile_z_{z_eval:.6f}_m".replace(".", "p")
        save_line_plot(
            x=r_m * 1e3,
            y=prof_norm,
            xlabel="Radius [mm]",
            ylabel="Normierte radiale Intensität",
            title=f"Radiales Profil bei z = {z_eval:.6f} m",
            out_png=output_dir / f"{radial_stem}.png",
            out_pdf=output_dir / f"{radial_stem}.pdf",
        )

    if r_mm_reference is None:
        raise RuntimeError("No radial profiles were generated.")

    save_line_plot(
        x=r_mm_reference,
        y=radial_profiles,
        xlabel="Radius [mm]",
        ylabel="Normierte radiale Intensität",
        title="Radiale Profile für mehrere z-Positionen",
        out_png=output_dir / "radial_profiles_multiple_z.png",
        out_pdf=output_dir / "radial_profiles_multiple_z.pdf",
        labels=radial_labels,
    )

    radial_matrix = np.array(radial_profiles, dtype=float).T
    save_heatmap(
        X_mm=profile_z_values * 1e3,
        Y_mm=r_mm_reference,
        Z=radial_matrix,
        xlabel="z [mm]",
        ylabel="Radius [mm]",
        title="2D-Darstellung der radialen Profile",
        colorbar_label="Normierte radiale Intensität",
        out_png=output_dir / "radial_profile_2d_vs_z.png",
        out_pdf=output_dir / "radial_profile_2d_vs_z.pdf",
    )

    log("Berechne size abhängig von z ...")
    for z_eval in z_size_values:
        if z_eval < pinhole_z:
            U_eval = propagate_to_z_with_elements(
                U_start=U0,
                dx=dx_m,
                dy=dy_m,
                wavelength=wavelength,
                z_target=float(z_eval),
                elements=elements,
                pad_factor=pad_factor,
            )
        else:
            U_eval = propagate_from_plane_to_z(
                U_plane=U_after_pinhole,
                dx=dx_m,
                dy=dy_m,
                wavelength=wavelength,
                z_plane=pinhole_z,
                z_target=float(z_eval),
                elements=elements,
                pad_factor=pad_factor,
            )
        I_eval = np.abs(U_eval) ** 2
        d4x_m, d4y_m = beam_d4sigma(I_eval, dx_m, dy_m)
        mean_d4_m = np.nanmean([d4x_m, d4y_m])
        size_rows.append({
            "z_m": float(z_eval),
            "z_mm": float(z_eval * 1e3),
            "d4sigma_x_m": float(d4x_m),
            "d4sigma_y_m": float(d4y_m),
            "d4sigma_x_mm": float(d4x_m * 1e3),
            "d4sigma_y_mm": float(d4y_m * 1e3),
            "d4sigma_mean_mm": float(mean_d4_m * 1e3),
        })

    z_mm = np.array([row["z_mm"] for row in size_rows], dtype=float)
    d4x_mm = np.array([row["d4sigma_x_mm"] for row in size_rows], dtype=float)
    d4y_mm = np.array([row["d4sigma_y_mm"] for row in size_rows], dtype=float)
    d4m_mm = np.array([row["d4sigma_mean_mm"] for row in size_rows], dtype=float)

    save_line_plot(
        x=z_mm,
        y=[d4x_mm, d4y_mm, d4m_mm],
        xlabel="z [mm]",
        ylabel="D4σ [mm]",
        title="Beam size abhängig von z",
        out_png=output_dir / "beam_size_vs_z.png",
        out_pdf=output_dir / "beam_size_vs_z.pdf",
        labels=["D4σ x", "D4σ y", "D4σ Mittel"],
    )

    with open(output_dir / "radial_profiles.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(radial_rows[0].keys()))
        writer.writeheader()
        writer.writerows(radial_rows)

    with open(output_dir / "beam_size_vs_z.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(size_rows[0].keys()))
        writer.writeheader()
        writer.writerows(size_rows)

    meta = {
        "wavelength_nm": wavelength * 1e9,
        "dx_um": dx_m * 1e6,
        "dy_um": dy_m * 1e6,
        "threshold_abs": threshold_abs,
        "valid_pixels": int(valid_mask.sum()),
        "mask_center_x_px": cx,
        "mask_center_y_px": cy,
        "pinhole_fixed_z_m": pinhole_z,
        "pinhole_fixed_diameter_um": pinhole_d * 1e6,
        "pinhole_x_offset_um": float(CFG["PINHOLE_X_OFFSET_M"]) * 1e6,
        "pinhole_y_offset_um": float(CFG["PINHOLE_Y_OFFSET_M"]) * 1e6,
        "transmission": transmission,
        "profile_z_positions_m": profile_z_values.tolist(),
        "size_vs_z_min_m": float(CFG["SIZE_VS_Z_MIN_M"]),
        "size_vs_z_includes_region_before_pinhole": True,
        "size_vs_z_max_m": float(CFG["SIZE_VS_Z_MAX_M"]),
        "size_vs_z_num_points": int(CFG["SIZE_VS_Z_NUM_POINTS"]),
        "radial_profile_num_bins": int(CFG["RADIAL_PROFILE_NUM_BINS"]),
        "pad_factor": int(CFG["PAD_FACTOR"]),
    }
    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    total_elapsed = time.perf_counter() - total_t0

    print()
    print("=" * 80)
    print("FIXED PINHOLE PROFILE ANALYSIS FINISHED")
    print("=" * 80)
    print(f"Output directory:              {output_dir}")
    print(f"Pinhole z:                    {pinhole_z:.6f} m")
    print(f"Pinhole diameter:             {pinhole_d * 1e6:.3f} um")
    print(f"Transmission:                 {transmission:.6f}")
    print(f"Number of profile z points:   {len(profile_z_values)}")
    print(f"Number of size-z points:      {len(z_size_values)}")
    print(f"Total runtime:                {total_elapsed:.2f} s")



def main() -> None:
    fixed_pinhole_profiles()


if __name__ == "__main__":
    main()
