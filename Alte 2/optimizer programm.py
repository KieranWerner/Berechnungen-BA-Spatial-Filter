from __future__ import annotations

from pathlib import Path
import json
import csv
import math
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import (
    gaussian_filter,
    uniform_filter,
    affine_transform,
)

# ============================================================
# USER SETTINGS
# ============================================================

# -----------------------------
# Theory: input files
# -----------------------------
PHA_FILE = Path(r"C:\Users\User\Desktop\combined PHA front csv\average_PHA no bg substract.csv")

INTENSITY_VARIANTS = [
    {
        "name": "int_raw",
        "path": Path(r"C:\Users\User\Desktop\combined INT front csv\average_INT no bg substract.csv"),
    },
    {
        "name": "int_corrected",
        "path": Path(r"C:\Users\User\Desktop\combined INT front csv\average_INT no bg substract_corrected.csv"),
    },
]

BASE_OUTPUT_DIR = Path(r"C:\Users\User\Desktop\extended_parameter_optimization")
BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_D4SIGMA_MM = 13.7
WAVELENGTH_VACUUM_M = 800e-9
Z_LIST = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 8.1, 10.0, 12.0, 13.5]

INTENSITY_THRESHOLD_REL = 0.02
THEORY_MODE = "with_phase"

# reproducibility
RANDOM_SEED = 12345

# -----------------------------
# Measurement: beam profiler
# -----------------------------
DATASET_NAME = "beamprofiler_front"
PIXEL_SIZE_MM = 0.011 * 2.1
UNIT_LABEL = "mm"

MEASUREMENTS = [
    (0.53, "messung 53cm referenz_bg_subtracted.tiff"),
    (2.00, "messung 200 cm_bg_subtracted.tiff"),
    (3.50, "messung 350cm_bg_subtracted.tiff"),
    (5.00, "messung 500cm_bg_subtracted.tiff"),
    (7.50, "messung 750cm_bg_subtracted.tiff"),
    (13.5, "messung ende_bg_subtracted.tiff"),
]

MEAS_BASE_DIR = Path(r"C:\Users\User\Desktop\2D Beamprofiler background substracted")

# -----------------------------
# Plot selection
# -----------------------------
PLOT_D4SIGMA_AVG = True
PLOT_D_EE50 = False
PLOT_D_EE86 = False
PLOT_D_AREA_50PCT = False
PLOT_D_AREA_13P5PCT = False
PLOT_FWHM = False

METRICS = [
    ("D4sigma_avg_mm", "D4σ"),
    ("D_EE50_mm", "EE50"),
    ("D_EE86_mm", "EE86"),
    ("D_area_50pct_mm", "Area50%"),
    ("D_area_13p5pct_mm", "Area13.5%"),
    ("FWHM_mm", "FWHM"),
]

# ============================================================
# COMPARISON CONFIGURATION
# ============================================================

COMPARISON_METRICS = [
    "D4sigma_avg_mm",
]

PLOT_ONLY_SELECTED_METRICS = False

# 50 simulations per effect
N_VALUES = 50
RANK_BY = "mae_mm"   # or "rmse_mm"

# ============================================================
# GLOBAL STYLE
# ============================================================

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
})

CFG = {}

# ============================================================
# DEFAULT SCENARIO
# ============================================================

DEFAULT_SCENARIO = {
    # existing effects
    "USE_SOFT_MASK": False,
    "USE_XML_MASK": False,
    "USE_PHASE": True,
    "PHASE_SIGN": -1.0,

    "USE_EXTRA_PHASE": False,
    "TILT_X": 0.0,
    "TILT_Y": 0.0,
    "DEFOCUS": 0.0,
    "ASTIG_X": 0.0,
    "ASTIG_Y": 0.0,

    "USE_Z_OFFSET": False,
    "Z_OFFSET_M": 0.0,

    "USE_DETECTOR": False,
    "BLUR_SIGMA_PX": 1.2,
    "BACKGROUND_LEVEL": 0.0,

    "USE_CIRCLE_MASK": True,
    "CIRCLE_RADIUS_PX": 180,
    "CIRCLE_CENTER_X": None,
    "CIRCLE_CENTER_Y": None,

    "USE_AIR_REFRACTION": False,
    "AIR_REF_INDEX": 1.00027,

    "USE_TURBULENCE": False,
    "TURB_SIGMA_RAD": 0.0,
    "TURB_CORR_SIGMA_PX": 12.0,

    "USE_SPHERICAL_ABERRATION": False,
    "SPHERICAL_COEFF": 0.0,

    "USE_COMA": False,
    "COMA_X": 0.0,
    "COMA_Y": 0.0,

    "USE_PIXEL_INTEGRATION": False,
    "PIXEL_BINNING_SIZE": 2,

    "USE_APERTURE_GAUSSIAN": False,
    "APERTURE_SIGMA_FACTOR": 0.85,

    # new: registration / alignment
    "USE_INPUT_WARP": False,
    "SHIFT_X_PX": 0.0,
    "SHIFT_Y_PX": 0.0,
    "ROTATION_DEG": 0.0,
    "SCALE_X": 1.0,
    "SCALE_Y": 1.0,

    # new: phase quality / mismatch
    "USE_PHASE_MODEL": False,
    "PHASE_SCALE": 1.0,
    "PHASE_OFFSET_RAD": 0.0,

    # new: higher aberrations
    "USE_HIGHER_ABERRATIONS": False,
    "TREFOIL_X": 0.0,
    "TREFOIL_Y": 0.0,
    "SECONDARY_ASTIG_X": 0.0,
    "SECONDARY_ASTIG_Y": 0.0,
    "QUADRAFOIL": 0.0,

    # new: amplitude / aperture / transmission
    "USE_ELLIPTICAL_APERTURE": False,
    "APERTURE_RADIUS_X_PX": 180.0,
    "APERTURE_RADIUS_Y_PX": 180.0,
    "APERTURE_SHIFT_X_PX": 0.0,
    "APERTURE_SHIFT_Y_PX": 0.0,

    "USE_TRANSMISSION_MODEL": False,
    "TRANSMISSION_GRAD_X": 0.0,
    "TRANSMISSION_GRAD_Y": 0.0,
    "TRANSMISSION_QUAD_X": 0.0,
    "TRANSMISSION_QUAD_Y": 0.0,

    # new: halo / stray light
    "USE_HALO": False,
    "HALO_STRENGTH": 0.0,
    "HALO_SIGMA_PX": 30.0,

    # new: practical detector effects
    "USE_CAMERA_MODEL": False,
    "SATURATION_LEVEL_REL": 1.0,   # relative to current peak
    "GAMMA": 1.0,
    "READ_NOISE_REL": 0.0,

    # new: jitter / partial coherence
    "USE_JITTER": False,
    "JITTER_REALIZATIONS": 5,
    "JITTER_SHIFT_SIGMA_PX": 0.0,
    "JITTER_TILT_SIGMA": 0.0,

    "USE_PARTIAL_COHERENCE": False,
    "PARTIAL_COHERENCE_MIX": 0.0,
    "PARTIAL_COHERENCE_SIGMA_PX": 3.0,
}

# ============================================================
# PARAMETER SCANS
# larger intervals, 50 values each
# ============================================================

PARAMETERS = [
    # original effects, larger ranges
    {
        "name": "DEFOCUS",
        "use_flags": {"USE_EXTRA_PHASE": True},
        "config_key": "DEFOCUS",
        "values": np.linspace(-4.0, +4.0, N_VALUES),
    },
    {
        "name": "TILT_X",
        "use_flags": {"USE_EXTRA_PHASE": True},
        "config_key": "TILT_X",
        "values": np.linspace(-0.009, +0.009, N_VALUES),
    },
    {
        "name": "TILT_Y",
        "use_flags": {"USE_EXTRA_PHASE": True},
        "config_key": "TILT_Y",
        "values": np.linspace(-0.009, +0.009, N_VALUES),
    },
    {
        "name": "ASTIG_X",
        "use_flags": {"USE_EXTRA_PHASE": True},
        "config_key": "ASTIG_X",
        "values": np.linspace(-1.5, +1.5, N_VALUES),
    },
    {
        "name": "ASTIG_Y",
        "use_flags": {"USE_EXTRA_PHASE": True},
        "config_key": "ASTIG_Y",
        "values": np.linspace(-1.5, +1.5, N_VALUES),
    },
    {
        "name": "BLUR_SIGMA_PX",
        "use_flags": {"USE_DETECTOR": True},
        "config_key": "BLUR_SIGMA_PX",
        "values": np.linspace(0.0, 7.5, N_VALUES),
    },
    {
        "name": "Z_OFFSET_M",
        "use_flags": {"USE_Z_OFFSET": True},
        "config_key": "Z_OFFSET_M",
        "values": np.linspace(-0.9, +0.9, N_VALUES),
    },
    {
        "name": "SPHERICAL_COEFF",
        "use_flags": {"USE_SPHERICAL_ABERRATION": True},
        "config_key": "SPHERICAL_COEFF",
        "values": np.linspace(-3.0, +3.0, N_VALUES),
    },
    {
        "name": "COMA_X",
        "use_flags": {"USE_COMA": True},
        "config_key": "COMA_X",
        "values": np.linspace(-3.0, +3.0, N_VALUES),
    },
    {
        "name": "COMA_Y",
        "use_flags": {"USE_COMA": True},
        "config_key": "COMA_Y",
        "values": np.linspace(-3.0, +3.0, N_VALUES),
    },

    # new alignment / registration
    {
        "name": "SHIFT_X_PX",
        "use_flags": {"USE_INPUT_WARP": True},
        "config_key": "SHIFT_X_PX",
        "values": np.linspace(-45.0, +45.0, N_VALUES),
    },
    {
        "name": "SHIFT_Y_PX",
        "use_flags": {"USE_INPUT_WARP": True},
        "config_key": "SHIFT_Y_PX",
        "values": np.linspace(-45.0, +45.0, N_VALUES),
    },
    {
        "name": "ROTATION_DEG",
        "use_flags": {"USE_INPUT_WARP": True},
        "config_key": "ROTATION_DEG",
        "values": np.linspace(-12.0, +12.0, N_VALUES),
    },
    {
        "name": "SCALE_X",
        "use_flags": {"USE_INPUT_WARP": True},
        "config_key": "SCALE_X",
        "values": np.linspace(0.85, 1.15, N_VALUES),
    },
    {
        "name": "SCALE_Y",
        "use_flags": {"USE_INPUT_WARP": True},
        "config_key": "SCALE_Y",
        "values": np.linspace(0.85, 1.15, N_VALUES),
    },

    # new phase mismatch
    {
        "name": "PHASE_SCALE",
        "use_flags": {"USE_PHASE_MODEL": True},
        "config_key": "PHASE_SCALE",
        "values": np.linspace(0.5, 1.5, N_VALUES),
    },
    {
        "name": "PHASE_OFFSET_RAD",
        "use_flags": {"USE_PHASE_MODEL": True},
        "config_key": "PHASE_OFFSET_RAD",
        "values": np.linspace(-np.pi, +np.pi, N_VALUES),
    },

    # new higher aberrations
    {
        "name": "TREFOIL_X",
        "use_flags": {"USE_HIGHER_ABERRATIONS": True},
        "config_key": "TREFOIL_X",
        "values": np.linspace(-3.0, +3.0, N_VALUES),
    },
    {
        "name": "TREFOIL_Y",
        "use_flags": {"USE_HIGHER_ABERRATIONS": True},
        "config_key": "TREFOIL_Y",
        "values": np.linspace(-3.0, +3.0, N_VALUES),
    },
    {
        "name": "SECONDARY_ASTIG_X",
        "use_flags": {"USE_HIGHER_ABERRATIONS": True},
        "config_key": "SECONDARY_ASTIG_X",
        "values": np.linspace(-3.0, +3.0, N_VALUES),
    },
    {
        "name": "SECONDARY_ASTIG_Y",
        "use_flags": {"USE_HIGHER_ABERRATIONS": True},
        "config_key": "SECONDARY_ASTIG_Y",
        "values": np.linspace(-3.0, +3.0, N_VALUES),
    },
    {
        "name": "QUADRAFOIL",
        "use_flags": {"USE_HIGHER_ABERRATIONS": True},
        "config_key": "QUADRAFOIL",
        "values": np.linspace(-3.0, +3.0, N_VALUES),
    },

    # new aperture / amplitude
    {
        "name": "APERTURE_RADIUS_X_PX",
        "use_flags": {"USE_ELLIPTICAL_APERTURE": True},
        "config_key": "APERTURE_RADIUS_X_PX",
        "values": np.linspace(90.0, 270.0, N_VALUES),
    },
    {
        "name": "APERTURE_RADIUS_Y_PX",
        "use_flags": {"USE_ELLIPTICAL_APERTURE": True},
        "config_key": "APERTURE_RADIUS_Y_PX",
        "values": np.linspace(90.0, 270.0, N_VALUES),
    },
    {
        "name": "APERTURE_SHIFT_X_PX",
        "use_flags": {"USE_ELLIPTICAL_APERTURE": True},
        "config_key": "APERTURE_SHIFT_X_PX",
        "values": np.linspace(-45.0, +45.0, N_VALUES),
    },
    {
        "name": "APERTURE_SHIFT_Y_PX",
        "use_flags": {"USE_ELLIPTICAL_APERTURE": True},
        "config_key": "APERTURE_SHIFT_Y_PX",
        "values": np.linspace(-45.0, +45.0, N_VALUES),
    },
    {
        "name": "TRANSMISSION_GRAD_X",
        "use_flags": {"USE_TRANSMISSION_MODEL": True},
        "config_key": "TRANSMISSION_GRAD_X",
        "values": np.linspace(-0.9, +0.9, N_VALUES),
    },
    {
        "name": "TRANSMISSION_GRAD_Y",
        "use_flags": {"USE_TRANSMISSION_MODEL": True},
        "config_key": "TRANSMISSION_GRAD_Y",
        "values": np.linspace(-0.9, +0.9, N_VALUES),
    },
    {
        "name": "TRANSMISSION_QUAD_X",
        "use_flags": {"USE_TRANSMISSION_MODEL": True},
        "config_key": "TRANSMISSION_QUAD_X",
        "values": np.linspace(-1.2, +1.2, N_VALUES),
    },
    {
        "name": "TRANSMISSION_QUAD_Y",
        "use_flags": {"USE_TRANSMISSION_MODEL": True},
        "config_key": "TRANSMISSION_QUAD_Y",
        "values": np.linspace(-1.2, +1.2, N_VALUES),
    },

    # halo / stray light
    {
        "name": "HALO_STRENGTH",
        "use_flags": {"USE_HALO": True},
        "config_key": "HALO_STRENGTH",
        "values": np.linspace(0.0, 0.4, N_VALUES),
    },
    {
        "name": "HALO_SIGMA_PX",
        "use_flags": {"USE_HALO": True},
        "config_key": "HALO_SIGMA_PX",
        "values": np.linspace(10.0, 120.0, N_VALUES),
    },

    # camera model
    {
        "name": "BACKGROUND_LEVEL",
        "use_flags": {"USE_DETECTOR": True},
        "config_key": "BACKGROUND_LEVEL",
        "values": np.linspace(0.0, 0.06, N_VALUES),
    },
    {
        "name": "SATURATION_LEVEL_REL",
        "use_flags": {"USE_CAMERA_MODEL": True},
        "config_key": "SATURATION_LEVEL_REL",
        "values": np.linspace(0.35, 1.0, N_VALUES),
    },
    {
        "name": "GAMMA",
        "use_flags": {"USE_CAMERA_MODEL": True},
        "config_key": "GAMMA",
        "values": np.linspace(0.7, 1.5, N_VALUES),
    },
    {
        "name": "READ_NOISE_REL",
        "use_flags": {"USE_CAMERA_MODEL": True},
        "config_key": "READ_NOISE_REL",
        "values": np.linspace(0.0, 0.04, N_VALUES),
    },

    # jitter / partial coherence
    {
        "name": "JITTER_SHIFT_SIGMA_PX",
        "use_flags": {"USE_JITTER": True},
        "config_key": "JITTER_SHIFT_SIGMA_PX",
        "values": np.linspace(0.0, 12.0, N_VALUES),
    },
    {
        "name": "JITTER_TILT_SIGMA",
        "use_flags": {"USE_JITTER": True},
        "config_key": "JITTER_TILT_SIGMA",
        "values": np.linspace(0.0, 0.006, N_VALUES),
    },
    {
        "name": "PARTIAL_COHERENCE_MIX",
        "use_flags": {"USE_PARTIAL_COHERENCE": True},
        "config_key": "PARTIAL_COHERENCE_MIX",
        "values": np.linspace(0.0, 0.5, N_VALUES),
    },
    {
        "name": "PARTIAL_COHERENCE_SIGMA_PX",
        "use_flags": {"USE_PARTIAL_COHERENCE": True},
        "config_key": "PARTIAL_COHERENCE_SIGMA_PX",
        "values": np.linspace(1.0, 15.0, N_VALUES),
    },
]

# ============================================================
# COMBINATION SCANS
# Each combination uses best single values and scans a global alpha
# ============================================================

COMBINATION_GROUPS = [
    {
        "name": "alignment_combo",
        "members": ["SHIFT_X_PX", "SHIFT_Y_PX", "ROTATION_DEG", "SCALE_X", "SCALE_Y"],
        "alpha_values": np.linspace(0.0, 1.8, N_VALUES),
    },
    {
        "name": "phase_low_order_combo",
        "members": ["TILT_X", "TILT_Y", "DEFOCUS", "ASTIG_X", "ASTIG_Y", "PHASE_SCALE", "PHASE_OFFSET_RAD", "Z_OFFSET_M"],
        "alpha_values": np.linspace(0.0, 1.8, N_VALUES),
    },
    {
        "name": "higher_aberration_combo",
        "members": ["SPHERICAL_COEFF", "COMA_X", "COMA_Y", "TREFOIL_X", "TREFOIL_Y", "SECONDARY_ASTIG_X", "SECONDARY_ASTIG_Y", "QUADRAFOIL"],
        "alpha_values": np.linspace(0.0, 1.8, N_VALUES),
    },
    {
        "name": "amplitude_aperture_combo",
        "members": [
            "APERTURE_RADIUS_X_PX",
            "APERTURE_RADIUS_Y_PX",
            "APERTURE_SHIFT_X_PX",
            "APERTURE_SHIFT_Y_PX",
            "TRANSMISSION_GRAD_X",
            "TRANSMISSION_GRAD_Y",
            "TRANSMISSION_QUAD_X",
            "TRANSMISSION_QUAD_Y",
        ],
        "alpha_values": np.linspace(0.0, 1.8, N_VALUES),
    },
    {
        "name": "detector_halo_combo",
        "members": [
            "BLUR_SIGMA_PX",
            "BACKGROUND_LEVEL",
            "HALO_STRENGTH",
            "HALO_SIGMA_PX",
            "SATURATION_LEVEL_REL",
            "GAMMA",
            "READ_NOISE_REL",
        ],
        "alpha_values": np.linspace(0.0, 1.8, N_VALUES),
    },
    {
        "name": "jitter_coherence_combo",
        "members": [
            "JITTER_SHIFT_SIGMA_PX",
            "JITTER_TILT_SIGMA",
            "PARTIAL_COHERENCE_MIX",
            "PARTIAL_COHERENCE_SIGMA_PX",
        ],
        "alpha_values": np.linspace(0.0, 1.8, N_VALUES),
    },
    {
        "name": "practical_global_combo",
        "members": [
            "SHIFT_X_PX", "SHIFT_Y_PX", "ROTATION_DEG", "SCALE_X", "SCALE_Y",
            "PHASE_SCALE", "DEFOCUS", "ASTIG_X", "ASTIG_Y", "Z_OFFSET_M",
            "SPHERICAL_COEFF", "COMA_X", "COMA_Y", "TREFOIL_X", "TREFOIL_Y",
            "APERTURE_RADIUS_X_PX", "APERTURE_RADIUS_Y_PX",
            "APERTURE_SHIFT_X_PX", "APERTURE_SHIFT_Y_PX",
            "TRANSMISSION_GRAD_X", "TRANSMISSION_GRAD_Y",
            "BLUR_SIGMA_PX", "BACKGROUND_LEVEL",
            "HALO_STRENGTH", "HALO_SIGMA_PX",
            "SATURATION_LEVEL_REL", "GAMMA",
            "JITTER_SHIFT_SIGMA_PX", "PARTIAL_COHERENCE_MIX",
        ],
        "alpha_values": np.linspace(0.0, 1.5, N_VALUES),
    },
]

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_csv_auto(path):
    for delim in [",", ";", "\t"]:
        try:
            arr = np.loadtxt(path, delimiter=delim)
            if arr.ndim == 2 and arr.size > 0:
                return arr
        except Exception:
            pass
    raise ValueError(f"Could not parse {path}")


def load_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path), dtype=np.float64)


def px_to_mm(value_px: float) -> float:
    return value_px * PIXEL_SIZE_MM


def apply_soft_mask(mask, sigma=3.0):
    soft = gaussian_filter(mask.astype(float), sigma)
    mx = np.max(soft)
    if mx > 0:
        soft /= mx
    return soft


def get_metric_label(metric_name: str) -> str:
    for m, label in METRICS:
        if m == metric_name:
            return label
    return metric_name


def get_active_metrics_for_plot():
    if PLOT_ONLY_SELECTED_METRICS:
        return [(m, label) for m, label in METRICS if m in COMPARISON_METRICS]
    return METRICS


def validate_comparison_metrics():
    valid = {m for m, _ in METRICS}
    invalid = [m for m in COMPARISON_METRICS if m not in valid]
    if invalid:
        raise ValueError(
            f"Invalid entries in COMPARISON_METRICS: {invalid}. "
            f"Valid options are: {sorted(valid)}"
        )


def sanitize_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("_", "-", ".") else "_" for c in name)


def get_wavelength_in_medium():
    if CFG.get("USE_AIR_REFRACTION", False):
        return WAVELENGTH_VACUUM_M / float(CFG["AIR_REF_INDEX"])
    return WAVELENGTH_VACUUM_M


def parameter_def_by_name(name: str) -> dict:
    for p in PARAMETERS:
        if p["name"] == name:
            return p
    raise KeyError(f"Unknown parameter: {name}")


def value_with_alpha(default_value, best_value, alpha):
    return default_value + alpha * (best_value - default_value)


# ============================================================
# BEAM METRICS
# ============================================================

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

    if i_left > 0:
        x1, x2 = axis_coords[i_left - 1], axis_coords[i_left]
        y1, y2 = profile[i_left - 1], profile[i_left]
        if y2 != y1:
            left = x1 + (half - y1) * (x2 - x1) / (y2 - y1)
        else:
            left = axis_coords[i_left]
    else:
        left = axis_coords[i_left]

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

# ============================================================
# THEORY FUNCTIONS
# ============================================================

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
        circle = (X - cx) ** 2 + (Y - cy) ** 2 <= circle_radius_px ** 2
        valid = valid & circle

    return valid, float(thr), float(cx), float(cy)


def warp_array(arr, shift_x_px=0.0, shift_y_px=0.0, rotation_deg=0.0, scale_x=1.0, scale_y=1.0, order=1):
    arr = np.asarray(arr, dtype=float)
    ny, nx = arr.shape
    cx = 0.5 * (nx - 1)
    cy = 0.5 * (ny - 1)

    theta = np.deg2rad(rotation_deg)
    c = np.cos(theta)
    s = np.sin(theta)

    # output -> input transform
    A = np.array([
        [c / scale_x, s / scale_x],
        [-s / scale_y, c / scale_y],
    ], dtype=float)

    center = np.array([cy, cx], dtype=float)
    shift_rc = np.array([shift_y_px, shift_x_px], dtype=float)

    offset = center - A @ (center + shift_rc)

    return affine_transform(
        arr,
        matrix=A,
        offset=offset,
        output_shape=arr.shape,
        order=order,
        mode="constant",
        cval=0.0,
        prefilter=(order > 1),
    )


def preprocess_input_maps(intensity, phase_waves):
    intensity = np.asarray(intensity, dtype=float)
    phase_waves = np.asarray(phase_waves, dtype=float)

    if CFG.get("USE_INPUT_WARP", False):
        intensity = warp_array(
            intensity,
            shift_x_px=CFG["SHIFT_X_PX"],
            shift_y_px=CFG["SHIFT_Y_PX"],
            rotation_deg=CFG["ROTATION_DEG"],
            scale_x=CFG["SCALE_X"],
            scale_y=CFG["SCALE_Y"],
            order=1,
        )
        phase_waves = warp_array(
            phase_waves,
            shift_x_px=CFG["SHIFT_X_PX"],
            shift_y_px=CFG["SHIFT_Y_PX"],
            rotation_deg=CFG["ROTATION_DEG"],
            scale_x=CFG["SCALE_X"],
            scale_y=CFG["SCALE_Y"],
            order=1,
        )

    return intensity, phase_waves


def build_aperture_and_transmission(A):
    ny, nx = A.shape
    y, x = np.indices((ny, nx))
    cx = 0.5 * (nx - 1)
    cy = 0.5 * (ny - 1)

    amp = np.ones_like(A, dtype=float)

    if CFG.get("USE_ELLIPTICAL_APERTURE", False):
        rx = max(1.0, float(CFG["APERTURE_RADIUS_X_PX"]))
        ry = max(1.0, float(CFG["APERTURE_RADIUS_Y_PX"]))
        sx = float(CFG["APERTURE_SHIFT_X_PX"])
        sy = float(CFG["APERTURE_SHIFT_Y_PX"])
        xn = (x - (cx + sx)) / rx
        yn = (y - (cy + sy)) / ry
        mask = (xn**2 + yn**2) <= 1.0
        amp *= mask.astype(float)

    if CFG.get("USE_APERTURE_GAUSSIAN", False):
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        r0 = max(1.0, float(CFG.get("CIRCLE_RADIUS_PX", 180)))
        sigma = float(CFG.get("APERTURE_SIGMA_FACTOR", 0.85)) * r0
        ap = np.exp(-(r ** 2) / (2.0 * sigma ** 2))
        amp *= ap

    if CFG.get("USE_TRANSMISSION_MODEL", False):
        scale = max(nx, ny)
        Xn = (x - cx) / scale
        Yn = (y - cy) / scale
        trans = (
            1.0
            + float(CFG["TRANSMISSION_GRAD_X"]) * Xn
            + float(CFG["TRANSMISSION_GRAD_Y"]) * Yn
            + float(CFG["TRANSMISSION_QUAD_X"]) * (Xn**2)
            + float(CFG["TRANSMISSION_QUAD_Y"]) * (Yn**2)
        )
        trans = np.clip(trans, 0.0, None)
        amp *= trans

    return amp


def build_field_from_measurement(intensity, valid_mask, phase_waves=None,
                                 use_phase=True, phase_sign=-1.0):
    intensity = np.asarray(intensity, dtype=float)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    if not np.any(valid_mask):
        raise ValueError("Valid mask is empty.")

    int_norm = np.zeros_like(intensity, dtype=float)
    int_norm[valid_mask] = intensity[valid_mask] / np.nanmax(intensity[valid_mask])

    A = np.sqrt(np.clip(int_norm, 0, None))

    if CFG["USE_SOFT_MASK"]:
        soft_mask = apply_soft_mask(valid_mask)
        A *= soft_mask
    else:
        A[~valid_mask] = 0.0

    A *= build_aperture_and_transmission(A)

    if CFG["USE_PHASE"] and use_phase:
        if phase_waves is None:
            raise ValueError("phase_waves required")
        phase = np.nan_to_num(phase_waves)

        if CFG.get("USE_PHASE_MODEL", False):
            phase = float(CFG["PHASE_SCALE"]) * phase + float(CFG["PHASE_OFFSET_RAD"]) / (2.0 * np.pi)

        phi = phase_sign * 2.0 * np.pi * phase
        U0 = A * np.exp(1j * phi)
    else:
        U0 = A.astype(np.complex128)

    return U0, int_norm


def apply_extra_phase(U0, dx, dy, wavelength):
    ny, nx = U0.shape
    x = (np.arange(nx) - nx // 2) * dx
    y = (np.arange(ny) - ny // 2) * dy
    X, Y = np.meshgrid(x, y)

    k = 2 * np.pi / wavelength
    phi = np.zeros_like(X, dtype=float)

    if CFG["USE_EXTRA_PHASE"]:
        phi += (
            k * (CFG["TILT_X"] * X + CFG["TILT_Y"] * Y)
            + k * CFG["DEFOCUS"] * (X ** 2 + Y ** 2)
            + k * (CFG["ASTIG_X"] * X ** 2 + CFG["ASTIG_Y"] * Y ** 2)
        )

    scale = max(np.max(np.abs(X)), np.max(np.abs(Y)), 1e-12)
    Xn = X / scale
    Yn = Y / scale
    rho2 = Xn ** 2 + Yn ** 2

    if CFG.get("USE_SPHERICAL_ABERRATION", False):
        phi += float(CFG["SPHERICAL_COEFF"]) * rho2 ** 2

    if CFG.get("USE_COMA", False):
        phi += float(CFG["COMA_X"]) * Xn * rho2 + float(CFG["COMA_Y"]) * Yn * rho2

    if CFG.get("USE_HIGHER_ABERRATIONS", False):
        phi += float(CFG["TREFOIL_X"]) * (Xn**3 - 3.0 * Xn * Yn**2)
        phi += float(CFG["TREFOIL_Y"]) * (3.0 * Xn**2 * Yn - Yn**3)
        phi += float(CFG["SECONDARY_ASTIG_X"]) * (Xn**2 - Yn**2) * rho2
        phi += float(CFG["SECONDARY_ASTIG_Y"]) * (2.0 * Xn * Yn) * rho2
        phi += float(CFG["QUADRAFOIL"]) * (Xn**4 - 6.0 * Xn**2 * Yn**2 + Yn**4)

    if CFG.get("USE_TURBULENCE", False):
        rng = np.random.default_rng(RANDOM_SEED)
        noise = rng.normal(0.0, 1.0, size=U0.shape)
        corr_sigma = max(0.1, float(CFG["TURB_CORR_SIGMA_PX"]))
        noise = gaussian_filter(noise, corr_sigma)
        std = np.std(noise)
        if std > 0:
            noise = noise / std
        phi += float(CFG["TURB_SIGMA_RAD"]) * noise

    if np.any(phi != 0):
        return U0 * np.exp(1j * phi)
    return U0


def apply_camera_and_detector_model(I):
    I = np.asarray(I, dtype=float)

    if CFG["USE_DETECTOR"]:
        I = gaussian_filter(I, CFG["BLUR_SIGMA_PX"])
        I = I + CFG["BACKGROUND_LEVEL"]

    if CFG.get("USE_HALO", False):
        halo_sigma = max(0.5, float(CFG["HALO_SIGMA_PX"]))
        halo_strength = np.clip(float(CFG["HALO_STRENGTH"]), 0.0, 0.95)
        I_halo = gaussian_filter(I, halo_sigma)
        I = (1.0 - halo_strength) * I + halo_strength * I_halo

    if CFG.get("USE_PARTIAL_COHERENCE", False):
        mix = np.clip(float(CFG["PARTIAL_COHERENCE_MIX"]), 0.0, 0.95)
        sigma_pc = max(0.5, float(CFG["PARTIAL_COHERENCE_SIGMA_PX"]))
        I_blur = gaussian_filter(I, sigma_pc)
        I = (1.0 - mix) * I + mix * I_blur

    if CFG.get("USE_PIXEL_INTEGRATION", False):
        size = max(1, int(CFG["PIXEL_BINNING_SIZE"]))
        I = uniform_filter(I, size=size, mode="nearest")

    if CFG.get("USE_CAMERA_MODEL", False):
        I = np.clip(I, 0.0, None)

        peak = np.max(I)
        if peak > 0:
            sat = max(1e-12, float(CFG["SATURATION_LEVEL_REL"]) * peak)
            I = np.clip(I, 0.0, sat)
            I = I / sat
        else:
            I = np.zeros_like(I)

        gamma = max(0.1, float(CFG["GAMMA"]))
        I = np.power(I, gamma)

        noise_rel = max(0.0, float(CFG["READ_NOISE_REL"]))
        if noise_rel > 0:
            rng = np.random.default_rng(RANDOM_SEED)
            noise = rng.normal(0.0, noise_rel, size=I.shape)
            I = I + noise

        I = np.clip(I, 0.0, None)

    return I


def simulate_intensity_with_jitter(U0, dx_m, dy_m, wavelength_m, z_eff):
    if not CFG.get("USE_JITTER", False):
        Uz = U0 if z_eff == 0 else angular_spectrum(U0, dx_m, dy_m, wavelength_m, z_eff, pad_factor=4)
        Iz = np.abs(Uz) ** 2
        return apply_camera_and_detector_model(Iz)

    n_real = max(1, int(CFG.get("JITTER_REALIZATIONS", 5)))
    shift_sigma = max(0.0, float(CFG["JITTER_SHIFT_SIGMA_PX"]))
    tilt_sigma = max(0.0, float(CFG["JITTER_TILT_SIGMA"]))

    rng = np.random.default_rng(RANDOM_SEED)
    I_acc = None

    for _ in range(n_real):
        Uj = U0

        if tilt_sigma > 0:
            old_tx = CFG["TILT_X"]
            old_ty = CFG["TILT_Y"]
            tx = old_tx + rng.normal(0.0, tilt_sigma)
            ty = old_ty + rng.normal(0.0, tilt_sigma)

            ny, nx = Uj.shape
            x = (np.arange(nx) - nx // 2) * dx_m
            y = (np.arange(ny) - ny // 2) * dy_m
            X, Y = np.meshgrid(x, y)
            k = 2 * np.pi / wavelength_m
            Uj = Uj * np.exp(1j * k * (tx * X + ty * Y))

        Uz = Uj if z_eff == 0 else angular_spectrum(Uj, dx_m, dy_m, wavelength_m, z_eff, pad_factor=4)
        Iz = np.abs(Uz) ** 2

        if shift_sigma > 0:
            sx = rng.normal(0.0, shift_sigma)
            sy = rng.normal(0.0, shift_sigma)
            Iz = warp_array(Iz, shift_x_px=sx, shift_y_px=sy, rotation_deg=0.0, scale_x=1.0, scale_y=1.0, order=1)

        Iz = apply_camera_and_detector_model(Iz)

        if I_acc is None:
            I_acc = Iz
        else:
            I_acc += Iz

    return I_acc / n_real


def propagate_and_analyze(label, U0, dx_m, dy_m, wavelength_m, z_list, output_dir):
    rows = []

    for z in z_list:
        z_eff = z + CFG["Z_OFFSET_M"] if CFG["USE_Z_OFFSET"] else z

        Iz = simulate_intensity_with_jitter(U0, dx_m, dy_m, wavelength_m, z_eff)
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


def run_theory_for_scenario(int_file: Path, output_dir: Path):
    print("\n" + "=" * 80)
    print(f"THEORY CALCULATION: {CFG['name']}")
    print("=" * 80)

    phase_waves = load_csv_auto(PHA_FILE)
    intensity = load_csv_auto(int_file)

    if phase_waves.shape != intensity.shape:
        raise ValueError(f"Shape mismatch: phase {phase_waves.shape} vs intensity {intensity.shape}")

    intensity, phase_waves = preprocess_input_maps(intensity, phase_waves)

    use_circle = CFG.get("USE_CIRCLE_MASK", True)
    if CFG.get("USE_XML_MASK", False):
        use_circle = True

    valid_mask, threshold_abs, cx, cy = make_valid_mask(
        intensity=intensity,
        threshold_rel=INTENSITY_THRESHOLD_REL,
        use_circle=use_circle,
        circle_radius_px=CFG.get("CIRCLE_RADIUS_PX", 180),
        cx=CFG.get("CIRCLE_CENTER_X", None),
        cy=CFG.get("CIRCLE_CENTER_Y", None),
    )

    int_norm_for_scale = np.zeros_like(intensity, dtype=float)
    int_norm_for_scale[valid_mask] = intensity[valid_mask] / np.nanmax(intensity[valid_mask])

    tmp_px = beam_metrics_units(int_norm_for_scale, 1.0, 1.0)
    input_d4_px = tmp_px["d4sigma_avg"]

    dx_m = (INPUT_D4SIGMA_MM / input_d4_px) * 1e-3
    dy_m = dx_m

    wavelength_medium = get_wavelength_in_medium()

    U0_with_phase, _ = build_field_from_measurement(
        intensity=intensity,
        valid_mask=valid_mask,
        phase_waves=phase_waves,
        use_phase=True,
        phase_sign=CFG["PHASE_SIGN"],
    )
    U0_with_phase = apply_extra_phase(U0_with_phase, dx_m, dy_m, wavelength_medium)

    rows_with_phase = propagate_and_analyze(
        label="with_phase",
        U0=U0_with_phase,
        dx_m=dx_m,
        dy_m=dy_m,
        wavelength_m=wavelength_medium,
        z_list=Z_LIST,
        output_dir=output_dir,
    )

    df_plot = pd.DataFrame([
        {
            "z_m": r["z_m"],
            "D4sigma_avg_mm_with_phase": r["D4sigma_avg_mm"],
            "D_EE50_mm_with_phase": r["D_EE50_mm"],
            "D_EE86_mm_with_phase": r["D_EE86_mm"],
            "D_area_50pct_mm_with_phase": r["D_area_50pct_mm"],
            "D_area_13p5pct_mm_with_phase": r["D_area_13p5pct_mm"],
            "FWHM_mm_with_phase": r["FWHM_mm"],
        }
        for r in rows_with_phase
    ])

    df_plot.to_csv(output_dir / "beam_size_vs_z_plot_data.csv", index=False)

    with open(output_dir / "beam_size_vs_z_plot_data.json", "w", encoding="utf-8") as f:
        json.dump(df_plot.to_dict(orient="records"), f, indent=2)

    meta = {
        "scenario": CFG,
        "intensity_file": str(int_file),
        "pixel_size_um": dx_m * 1e6,
        "input_d4sigma_mm": INPUT_D4SIGMA_MM,
        "wavelength_vacuum_nm": WAVELENGTH_VACUUM_M * 1e9,
        "wavelength_medium_nm": wavelength_medium * 1e9,
        "phase_sign": CFG["PHASE_SIGN"],
        "intensity_threshold_rel": INTENSITY_THRESHOLD_REL,
        "intensity_threshold_abs": threshold_abs,
        "valid_pixels": int(valid_mask.sum()),
        "circle_center_x_px": cx,
        "circle_center_y_px": cy,
    }
    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Theory finished.")
    print(f"Scenario: {CFG['name']}")
    print(f"Pixel size: {dx_m * 1e6:.3f} um")
    print(f"Absolute threshold: {threshold_abs:.6g}")
    print(f"Valid pixels: {int(valid_mask.sum())}")
    print(f"Wavelength in medium: {wavelength_medium * 1e9:.6f} nm")

    return df_plot

# ============================================================
# MEASUREMENT FUNCTIONS
# ============================================================

def centroid(img: np.ndarray) -> tuple[float, float]:
    total = img.sum()
    if total <= 0:
        raise ValueError("Image contains no positive total intensity.")
    y, x = np.indices(img.shape)
    cx = float((img * x).sum() / total)
    cy = float((img * y).sum() / total)
    return cx, cy


def d4sigma_avg_px(img: np.ndarray) -> float:
    total = img.sum()
    if total <= 0:
        return np.nan

    y, x = np.indices(img.shape)
    cx, cy = centroid(img)

    sigma_x = math.sqrt(float((img * (x - cx) ** 2).sum() / total))
    sigma_y = math.sqrt(float((img * (y - cy) ** 2).sum() / total))

    dx = 4.0 * sigma_x
    dy = 4.0 * sigma_y
    return 0.5 * (dx + dy)


def encircled_energy_diameter_px(img: np.ndarray, fraction: float) -> float:
    total = img.sum()
    if total <= 0:
        return np.nan

    cx, cy = centroid(img)
    y, x = np.indices(img.shape)

    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_int = np.floor(r).astype(np.int32)

    radial_sum = np.bincount(r_int.ravel(), weights=img.ravel())
    cum = np.cumsum(radial_sum)

    target = fraction * total
    idx = np.searchsorted(cum, target)

    if idx <= 0:
        radius = float(idx)
    elif idx >= len(cum):
        return np.nan
    else:
        c1 = cum[idx - 1]
        c2 = cum[idx]
        if c2 > c1:
            frac_local = (target - c1) / (c2 - c1)
        else:
            frac_local = 0.0
        radius = (idx - 1) + frac_local

    return 2.0 * float(radius)


def fwhm_profile_px(img: np.ndarray) -> float:
    total = img.sum()
    if total <= 0:
        return np.nan

    cx, cy = centroid(img)

    cx_i = int(round(cx))
    cy_i = int(round(cy))

    cx_i = max(0, min(cx_i, img.shape[1] - 1))
    cy_i = max(0, min(cy_i, img.shape[0] - 1))

    profile_x = img[cy_i, :]
    profile_y = img[:, cx_i]

    def single_fwhm(profile: np.ndarray) -> float:
        peak = float(profile.max())
        if peak <= 0:
            return np.nan

        half = peak / 2.0
        indices = np.where(profile >= half)[0]
        if len(indices) < 2:
            return np.nan

        return float(indices[-1] - indices[0])

    fwhm_x = single_fwhm(profile_x)
    fwhm_y = single_fwhm(profile_y)

    if np.isnan(fwhm_x) and np.isnan(fwhm_y):
        return np.nan
    if np.isnan(fwhm_x):
        return fwhm_y
    if np.isnan(fwhm_y):
        return fwhm_x

    return 0.5 * (fwhm_x + fwhm_y)


def area_equivalent_diameter_px(img: np.ndarray, rel_threshold: float) -> float:
    peak = img.max()
    if peak <= 0:
        return np.nan

    mask = img >= (rel_threshold * peak)
    area_px = float(mask.sum())

    if area_px <= 0:
        return np.nan

    return math.sqrt(4.0 * area_px / math.pi)


def run_measurement():
    print("\n" + "=" * 80)
    print("MEASUREMENT EVALUATION")
    print("=" * 80)

    positions_m = []
    curves = {
        "D4sigma_avg": [],
        "D_EE50": [],
        "D_EE86": [],
        "D_area_50pct": [],
        "D_area_13p5pct": [],
        "FWHM": [],
    }
    full_rows = []

    for position_m, filename in MEASUREMENTS:
        path = MEAS_BASE_DIR / filename

        if not path.exists():
            print(f"WARNING: File not found: {path}")
            continue

        img = load_image(path)
        img = np.clip(img, 0, None)
        positions_m.append(position_m)

        d4 = px_to_mm(d4sigma_avg_px(img))
        ee50 = px_to_mm(encircled_energy_diameter_px(img, 0.50))
        ee86 = px_to_mm(encircled_energy_diameter_px(img, 0.86))
        area50 = px_to_mm(area_equivalent_diameter_px(img, 0.50))
        area135 = px_to_mm(area_equivalent_diameter_px(img, 0.135))
        fwhm = px_to_mm(fwhm_profile_px(img))

        curves["D4sigma_avg"].append(d4)
        curves["D_EE50"].append(ee50)
        curves["D_EE86"].append(ee86)
        curves["D_area_50pct"].append(area50)
        curves["D_area_13p5pct"].append(area135)
        curves["FWHM"].append(fwhm)

        full_rows.append({
            "dataset": DATASET_NAME,
            "position_m": position_m,
            "filename": filename,
            "D4sigma_avg_mm": d4,
            "D_EE50_mm": ee50,
            "D_EE86_mm": ee86,
            "D_area_50pct_mm": area50,
            "D_area_13p5pct_mm": area135,
            "FWHM_mm": fwhm,
        })

        print(
            f"{filename:35s} | Position = {position_m:5.2f} m | "
            f"D4sigma_avg = {d4:7.3f} mm | "
            f"D_EE50 = {ee50:7.3f} mm | "
            f"D_EE86 = {ee86:7.3f} mm | "
            f"D_area_50pct = {area50:7.3f} mm | "
            f"D_area_13p5pct = {area135:7.3f} mm | "
            f"FWHM = {fwhm:7.3f} mm"
        )

    if not positions_m:
        raise RuntimeError("No valid image files found.")

    plot_rows = []
    for i, pos in enumerate(positions_m):
        plot_rows.append({
            "dataset": DATASET_NAME,
            "position_m": pos,
            "D4sigma_avg_mm": curves["D4sigma_avg"][i],
            "D_EE50_mm": curves["D_EE50"][i],
            "D_EE86_mm": curves["D_EE86"][i],
            "D_area_50pct_mm": curves["D_area_50pct"][i],
            "D_area_13p5pct_mm": curves["D_area_13p5pct"][i],
            "FWHM_mm": curves["FWHM"][i],
        })

    plot_csv_path = MEAS_BASE_DIR / "beam_size_vs_position_plot_data.csv"
    with open(plot_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(plot_rows[0].keys()))
        writer.writeheader()
        writer.writerows(plot_rows)

    plot_json_path = MEAS_BASE_DIR / "beam_size_vs_position_plot_data.json"
    with open(plot_json_path, "w", encoding="utf-8") as f:
        json.dump(plot_rows, f, indent=2)

    full_csv_path = MEAS_BASE_DIR / "beam_size_vs_position_full_results.csv"
    with open(full_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(full_rows[0].keys()))
        writer.writeheader()
        writer.writerows(full_rows)

    plt.figure(figsize=(10, 6))

    if PLOT_D4SIGMA_AVG:
        plt.plot(positions_m, curves["D4sigma_avg"], marker="o", label="D4sigma_avg")
    if PLOT_D_EE50:
        plt.plot(positions_m, curves["D_EE50"], marker="o", label="D_EE50")
    if PLOT_D_EE86:
        plt.plot(positions_m, curves["D_EE86"], marker="o", label="D_EE86")
    if PLOT_D_AREA_50PCT:
        plt.plot(positions_m, curves["D_area_50pct"], marker="o", label="D_area_50pct")
    if PLOT_D_AREA_13P5PCT:
        plt.plot(positions_m, curves["D_area_13p5pct"], marker="o", label="D_area_13p5pct")
    if PLOT_FWHM:
        plt.plot(positions_m, curves["FWHM"], marker="o", label="FWHM")

    plt.xlabel("Position [m]")
    plt.ylabel(f"Size [{UNIT_LABEL}]")
    plt.title("Measurement: Beam size versus position")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = MEAS_BASE_DIR / "beam_size_vs_position.png"
    plt.savefig(out_path, dpi=200)
    plt.close()

    print("Measurement finished.")
    print(f"Plot saved to: {out_path}")
    print(f"Plot CSV saved to: {plot_csv_path}")
    print(f"Plot JSON saved to: {plot_json_path}")
    print(f"Full CSV saved to: {full_csv_path}")

    return pd.DataFrame(plot_rows)

# ============================================================
# COMPARISON
# ============================================================

def calculate_metric_error_stats(diff_values_mm):
    diff_values_mm = np.asarray(diff_values_mm, dtype=float)
    diff_values_mm = diff_values_mm[np.isfinite(diff_values_mm)]

    if diff_values_mm.size == 0:
        return {
            "mean_signed_diff_mm": np.nan,
            "mae_mm": np.nan,
            "rmse_mm": np.nan,
            "max_abs_diff_mm": np.nan,
            "n_points": 0,
        }

    return {
        "mean_signed_diff_mm": float(np.mean(diff_values_mm)),
        "mae_mm": float(np.mean(np.abs(diff_values_mm))),
        "rmse_mm": float(np.sqrt(np.mean(diff_values_mm ** 2))),
        "max_abs_diff_mm": float(np.max(np.abs(diff_values_mm))),
        "n_points": int(diff_values_mm.size),
    }


def run_comparison(df_meas: pd.DataFrame, df_theory: pd.DataFrame, scenario_name: str, output_dir: Path):
    print("\n" + "=" * 80)
    print(f"COMPARISON: THEORY VS MEASUREMENT | {scenario_name}")
    print("=" * 80)

    df_meas = df_meas.sort_values("position_m").copy()
    df_theory = df_theory.sort_values("z_m").copy()

    active_plot_metrics = get_active_metrics_for_plot()

    plt.figure(figsize=(13, 9))
    diff_rows = []

    for metric, label in active_plot_metrics:
        theory_col = f"{metric}_{THEORY_MODE}"

        if metric not in df_meas.columns or theory_col not in df_theory.columns:
            print(f"Skipped: {metric}")
            continue

        z_meas = df_meas["position_m"].values
        meas_vals = df_meas[metric].values

        z_theory = df_theory["z_m"].values
        theory_vals = df_theory[theory_col].values

        theory_interp = np.interp(z_meas, z_theory, theory_vals)
        diff = meas_vals - theory_interp

        for i in range(len(z_meas)):
            diff_rows.append({
                "scenario": scenario_name,
                "metric": metric,
                "metric_label": label,
                "z_m": z_meas[i],
                "measurement_mm": meas_vals[i],
                "theory_mm": theory_interp[i],
                "difference_mm": diff[i],
                "selected_for_comparison": metric in COMPARISON_METRICS,
            })

        plt.plot(z_theory, theory_vals, linestyle="--", linewidth=2.0, label=f"{label} Theory")
        plt.plot(z_meas, meas_vals, linestyle="-", marker="o", linewidth=2.0, markersize=5, label=f"{label} Measurement")

    plt.xlabel("Position / z [m]")
    plt.ylabel("Beam size [mm]")
    plt.title(f"Comparison: Theory ({THEORY_MODE}) vs Measurement | {scenario_name}")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_all_curves.png", dpi=200)
    plt.close()

    plt.figure(figsize=(13, 7))
    for metric, label in active_plot_metrics:
        rows = [r for r in diff_rows if r["metric"] == metric]
        if not rows:
            continue
        z = [r["z_m"] for r in rows]
        diff = [r["difference_mm"] for r in rows]
        plt.plot(z, diff, marker="o", linestyle="-", linewidth=2.0, label=f"{label} (Measurement - Theory)")

    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Position / z [m]")
    plt.ylabel("Difference [mm]")
    plt.title(f"Difference: Measurement − Theory | {scenario_name}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "difference_plot.png", dpi=200)
    plt.close()

    plt.figure(figsize=(13, 7))
    rel_rows = []

    for metric, label in active_plot_metrics:
        rows = [r for r in diff_rows if r["metric"] == metric]
        if not rows:
            continue

        z = []
        ratio_vals = []

        for r in rows:
            if r["theory_mm"] == 0:
                continue

            ratio = r["measurement_mm"] / r["theory_mm"]
            rel_error = (r["measurement_mm"] - r["theory_mm"]) / r["theory_mm"]

            z.append(r["z_m"])
            ratio_vals.append(ratio)

            rel_rows.append({
                "scenario": scenario_name,
                "metric": metric,
                "metric_label": label,
                "z_m": r["z_m"],
                "measurement_mm": r["measurement_mm"],
                "theory_mm": r["theory_mm"],
                "difference_mm": r["difference_mm"],
                "ratio_meas_over_theory": ratio,
                "relative_error": rel_error,
                "selected_for_comparison": metric in COMPARISON_METRICS,
            })

        plt.plot(z, ratio_vals, marker="o", linestyle="-", linewidth=2.0, label=f"{label} (Measurement/Theory)")

    plt.axhline(1.0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Position / z [m]")
    plt.ylabel("Measurement / Theory")
    plt.title(f"Relative deviation (Measurement / Theory) | {scenario_name}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "relative_ratio_plot.png", dpi=200)
    plt.close()

    df_diff = pd.DataFrame(diff_rows)
    df_rel = pd.DataFrame(rel_rows)

    df_diff.to_csv(output_dir / "difference_data.csv", index=False)
    df_rel.to_csv(output_dir / "ratio_data.csv", index=False)

    with open(output_dir / "difference_data.json", "w", encoding="utf-8") as f:
        json.dump(diff_rows, f, indent=2)

    with open(output_dir / "ratio_data.json", "w", encoding="utf-8") as f:
        json.dump(rel_rows, f, indent=2)

    df_diff_selected = df_diff[df_diff["metric"].isin(COMPARISON_METRICS)].copy()
    df_rel_selected = df_rel[df_rel["metric"].isin(COMPARISON_METRICS)].copy()

    df_diff_selected.to_csv(output_dir / "difference_data_selected_metrics.csv", index=False)
    df_rel_selected.to_csv(output_dir / "ratio_data_selected_metrics.csv", index=False)

    metric_summary_rows = []
    for metric in COMPARISON_METRICS:
        label = get_metric_label(metric)
        vals = df_diff_selected.loc[df_diff_selected["metric"] == metric, "difference_mm"].values
        stats = calculate_metric_error_stats(vals)
        metric_summary_rows.append({
            "scenario": scenario_name,
            "metric": metric,
            "label": label,
            "used_for_ranking": True,
            **stats
        })

    df_metric_summary = pd.DataFrame(metric_summary_rows)
    df_metric_summary.to_csv(output_dir / "difference_summary_selected_metrics.csv", index=False)

    overall_stats = calculate_metric_error_stats(df_diff_selected["difference_mm"].values)
    overall_row = {
        "scenario": scenario_name,
        "comparison_metrics": " | ".join(COMPARISON_METRICS),
        "comparison_metric_labels": " | ".join(get_metric_label(m) for m in COMPARISON_METRICS),
        **overall_stats
    }
    df_overall = pd.DataFrame([overall_row])
    df_overall.to_csv(output_dir / "difference_summary_overall_selected_metrics.csv", index=False)

    print("Comparison finished.")
    print(f"Saved to: {output_dir}")
    print("Overall difference stats (selected metrics only):")
    for k, v in overall_row.items():
        if k in ("scenario", "comparison_metrics", "comparison_metric_labels"):
            continue
        if isinstance(v, (int, np.integer)):
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v:.6f}")

    return df_diff, df_metric_summary, df_overall

# ============================================================
# SINGLE-PARAMETER OPTIMIZATION
# ============================================================

def evaluate_single_value(df_meas: pd.DataFrame, int_file: Path, parameter_def: dict, value: float, index: int, output_root: Path):
    global CFG
    CFG = deepcopy(DEFAULT_SCENARIO)

    scenario_name = f"{parameter_def['name']}_{index:03d}"
    CFG.update({
        "name": scenario_name,
        parameter_def["config_key"]: float(value),
    })
    CFG.update(parameter_def["use_flags"])

    scenario_dir = output_root / "single_parameter_scans" / parameter_def["name"] / scenario_name
    scenario_dir.mkdir(parents=True, exist_ok=True)

    df_theory = run_theory_for_scenario(int_file=int_file, output_dir=scenario_dir)
    _, df_metric_summary, df_overall = run_comparison(
        df_meas=df_meas,
        df_theory=df_theory,
        scenario_name=scenario_name,
        output_dir=scenario_dir,
    )

    overall_row = df_overall.iloc[0].to_dict()
    overall_row["parameter"] = parameter_def["name"]
    overall_row["parameter_value"] = float(value)

    return overall_row, df_metric_summary


def optimize_one_parameter(df_meas: pd.DataFrame, int_file: Path, parameter_def: dict, output_root: Path):
    param_name = parameter_def["name"]
    values = parameter_def["values"]

    print("\n" + "=" * 100)
    print(f"OPTIMIZING PARAMETER: {param_name}")
    print("=" * 100)

    output_param_dir = output_root / "single_parameter_scans" / param_name
    output_param_dir.mkdir(parents=True, exist_ok=True)

    overall_rows = []
    metric_rows = []

    for idx, value in enumerate(values, start=1):
        print(f"[{param_name}] {idx}/{len(values)} -> {value:.10g}")
        overall_row, df_metric_summary = evaluate_single_value(
            df_meas=df_meas,
            int_file=int_file,
            parameter_def=parameter_def,
            value=float(value),
            index=idx,
            output_root=output_root,
        )
        overall_rows.append(overall_row)

        df_metric_summary = df_metric_summary.copy()
        df_metric_summary["parameter"] = param_name
        df_metric_summary["parameter_value"] = float(value)
        metric_rows.append(df_metric_summary)

    df_overall = pd.DataFrame(overall_rows)
    df_metric = pd.concat(metric_rows, ignore_index=True)

    df_overall_sorted = df_overall.sort_values(RANK_BY).reset_index(drop=True)

    df_overall.to_csv(output_param_dir / f"{param_name}_scan_overall.csv", index=False)
    df_overall_sorted.to_csv(output_param_dir / f"{param_name}_scan_overall_sorted.csv", index=False)
    df_metric.to_csv(output_param_dir / f"{param_name}_scan_metric_summary.csv", index=False)

    with open(output_param_dir / f"{param_name}_scan_overall.json", "w", encoding="utf-8") as f:
        json.dump(df_overall.to_dict(orient="records"), f, indent=2)

    best_row = df_overall_sorted.iloc[0]

    plt.figure(figsize=(10, 6))
    plt.plot(df_overall["parameter_value"], df_overall["mae_mm"], marker="o", linestyle="-", label="MAE")
    plt.plot(df_overall["parameter_value"], df_overall["rmse_mm"], marker="o", linestyle="-", label="RMSE")
    plt.axvline(best_row["parameter_value"], linestyle="--", linewidth=1.5, label=f"Best {RANK_BY}")
    plt.xlabel(param_name)
    plt.ylabel("Error [mm]")
    plt.title(f"{param_name}: single-parameter optimization")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_param_dir / f"{param_name}_optimization_plot.png", dpi=200)
    plt.close()

    print(f"\nBest for {param_name}:")
    print(f"  value      = {best_row['parameter_value']:.10g}")
    print(f"  mae_mm     = {best_row['mae_mm']:.6f}")
    print(f"  rmse_mm    = {best_row['rmse_mm']:.6f}")
    print(f"  scenario   = {best_row['scenario']}")

    return best_row, df_overall, df_metric

# ============================================================
# COMBINATION OPTIMIZATION
# ============================================================

def build_cfg_for_combination(best_single_map: dict[str, float], combo_def: dict, alpha: float):
    cfg = deepcopy(DEFAULT_SCENARIO)
    cfg["name"] = f"{combo_def['name']}_alpha_{alpha:.3f}"

    for member in combo_def["members"]:
        if member not in best_single_map:
            continue

        pdef = parameter_def_by_name(member)
        cfg.update(pdef["use_flags"])

        default_val = DEFAULT_SCENARIO[pdef["config_key"]]
        best_val = best_single_map[member]

        cfg[pdef["config_key"]] = value_with_alpha(default_val, best_val, alpha)

    return cfg


def evaluate_combination_alpha(df_meas: pd.DataFrame, int_file: Path, combo_def: dict, alpha: float, idx: int, best_single_map: dict[str, float], output_root: Path):
    global CFG
    CFG = build_cfg_for_combination(best_single_map, combo_def, alpha)

    scenario_name = f"{combo_def['name']}_{idx:03d}"
    CFG["name"] = scenario_name

    scenario_dir = output_root / "combination_scans" / combo_def["name"] / scenario_name
    scenario_dir.mkdir(parents=True, exist_ok=True)

    df_theory = run_theory_for_scenario(int_file=int_file, output_dir=scenario_dir)
    _, df_metric_summary, df_overall = run_comparison(
        df_meas=df_meas,
        df_theory=df_theory,
        scenario_name=scenario_name,
        output_dir=scenario_dir,
    )

    overall_row = df_overall.iloc[0].to_dict()
    overall_row["combination"] = combo_def["name"]
    overall_row["alpha"] = float(alpha)

    return overall_row, df_metric_summary


def optimize_combination(df_meas: pd.DataFrame, int_file: Path, combo_def: dict, best_single_map: dict[str, float], output_root: Path):
    combo_name = combo_def["name"]
    alpha_values = combo_def["alpha_values"]

    print("\n" + "=" * 100)
    print(f"OPTIMIZING COMBINATION: {combo_name}")
    print("=" * 100)

    combo_dir = output_root / "combination_scans" / combo_name
    combo_dir.mkdir(parents=True, exist_ok=True)

    overall_rows = []
    metric_rows = []

    for idx, alpha in enumerate(alpha_values, start=1):
        print(f"[{combo_name}] {idx}/{len(alpha_values)} -> alpha={alpha:.6f}")
        overall_row, df_metric_summary = evaluate_combination_alpha(
            df_meas=df_meas,
            int_file=int_file,
            combo_def=combo_def,
            alpha=float(alpha),
            idx=idx,
            best_single_map=best_single_map,
            output_root=output_root,
        )
        overall_rows.append(overall_row)

        df_metric_summary = df_metric_summary.copy()
        df_metric_summary["combination"] = combo_name
        df_metric_summary["alpha"] = float(alpha)
        metric_rows.append(df_metric_summary)

    df_overall = pd.DataFrame(overall_rows)
    df_metric = pd.concat(metric_rows, ignore_index=True)
    df_overall_sorted = df_overall.sort_values(RANK_BY).reset_index(drop=True)

    df_overall.to_csv(combo_dir / f"{combo_name}_scan_overall.csv", index=False)
    df_overall_sorted.to_csv(combo_dir / f"{combo_name}_scan_overall_sorted.csv", index=False)
    df_metric.to_csv(combo_dir / f"{combo_name}_scan_metric_summary.csv", index=False)

    best_row = df_overall_sorted.iloc[0]

    plt.figure(figsize=(10, 6))
    plt.plot(df_overall["alpha"], df_overall["mae_mm"], marker="o", linestyle="-", label="MAE")
    plt.plot(df_overall["alpha"], df_overall["rmse_mm"], marker="o", linestyle="-", label="RMSE")
    plt.axvline(best_row["alpha"], linestyle="--", linewidth=1.5, label=f"Best {RANK_BY}")
    plt.xlabel("Combination scale alpha")
    plt.ylabel("Error [mm]")
    plt.title(f"{combo_name}: combination optimization")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(combo_dir / f"{combo_name}_optimization_plot.png", dpi=200)
    plt.close()

    print(f"\nBest for {combo_name}:")
    print(f"  alpha      = {best_row['alpha']:.6f}")
    print(f"  mae_mm     = {best_row['mae_mm']:.6f}")
    print(f"  rmse_mm    = {best_row['rmse_mm']:.6f}")
    print(f"  scenario   = {best_row['scenario']}")

    return best_row, df_overall, df_metric


# ============================================================
# RUNNER PER INTENSITY VARIANT
# ============================================================

def run_one_intensity_variant(df_meas: pd.DataFrame, intensity_variant: dict):
    int_name = sanitize_name(intensity_variant["name"])
    int_file = intensity_variant["path"]

    output_root = BASE_OUTPUT_DIR / int_name
    output_root.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 100)
    print(f"RUN INTENSITY VARIANT: {int_name}")
    print(f"FILE: {int_file}")
    print("#" * 100)

    best_rows = []

    for parameter_def in PARAMETERS:
        best_row, _, _ = optimize_one_parameter(
            df_meas=df_meas,
            int_file=int_file,
            parameter_def=parameter_def,
            output_root=output_root,
        )
        best_rows.append(best_row)

    df_best = pd.DataFrame(best_rows)
    df_best_sorted = df_best.sort_values(RANK_BY).reset_index(drop=True)
    df_best.to_csv(output_root / "best_values_per_parameter.csv", index=False)
    df_best_sorted.to_csv(output_root / "best_values_per_parameter_sorted.csv", index=False)

    plt.figure(figsize=(14, 7))
    plt.bar(df_best_sorted["parameter"], df_best_sorted[RANK_BY])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(f"{RANK_BY} [mm]")
    plt.title(f"Best single value per parameter ranked by {RANK_BY} | {int_name}")
    plt.tight_layout()
    plt.savefig(output_root / "best_parameter_ranking.png", dpi=200)
    plt.close()

    best_single_map = {
        row["parameter"]: float(row["parameter_value"])
        for _, row in df_best_sorted.iterrows()
    }

    combo_best_rows = []
    for combo_def in COMBINATION_GROUPS:
        best_row, _, _ = optimize_combination(
            df_meas=df_meas,
            int_file=int_file,
            combo_def=combo_def,
            best_single_map=best_single_map,
            output_root=output_root,
        )
        combo_best_rows.append(best_row)

    df_combo_best = pd.DataFrame(combo_best_rows)
    df_combo_best_sorted = df_combo_best.sort_values(RANK_BY).reset_index(drop=True)
    df_combo_best.to_csv(output_root / "best_values_per_combination.csv", index=False)
    df_combo_best_sorted.to_csv(output_root / "best_values_per_combination_sorted.csv", index=False)

    plt.figure(figsize=(14, 7))
    plt.bar(df_combo_best_sorted["combination"], df_combo_best_sorted[RANK_BY])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(f"{RANK_BY} [mm]")
    plt.title(f"Best combination per group ranked by {RANK_BY} | {int_name}")
    plt.tight_layout()
    plt.savefig(output_root / "best_combination_ranking.png", dpi=200)
    plt.close()

    with open(output_root / "best_single_map.json", "w", encoding="utf-8") as f:
        json.dump(best_single_map, f, indent=2)

    print("\n" + "=" * 100)
    print(f"FINAL BEST SINGLE VALUES | {int_name}")
    print("=" * 100)
    print(df_best_sorted[["parameter", "parameter_value", "mae_mm", "rmse_mm", "scenario"]].to_string(index=False))

    print("\n" + "=" * 100)
    print(f"FINAL BEST COMBINATIONS | {int_name}")
    print("=" * 100)
    print(df_combo_best_sorted[["combination", "alpha", "mae_mm", "rmse_mm", "scenario"]].to_string(index=False))

    return {
        "intensity_name": int_name,
        "intensity_file": str(int_file),
        "best_single_df": df_best_sorted,
        "best_combo_df": df_combo_best_sorted,
    }

# ============================================================
# GLOBAL MAIN
# ============================================================

def main():
    validate_comparison_metrics()

    print("\n" + "=" * 100)
    print("RUN MEASUREMENT ONCE")
    print("=" * 100)
    df_meas = run_measurement()

    all_variant_single = []
    all_variant_combo = []

    results = []
    for intensity_variant in INTENSITY_VARIANTS:
        res = run_one_intensity_variant(df_meas=df_meas, intensity_variant=intensity_variant)
        results.append(res)

        tmp_single = res["best_single_df"].copy()
        tmp_single["intensity_variant"] = res["intensity_name"]
        all_variant_single.append(tmp_single)

        tmp_combo = res["best_combo_df"].copy()
        tmp_combo["intensity_variant"] = res["intensity_name"]
        all_variant_combo.append(tmp_combo)

    df_all_single = pd.concat(all_variant_single, ignore_index=True)
    df_all_combo = pd.concat(all_variant_combo, ignore_index=True)

    df_all_single.to_csv(BASE_OUTPUT_DIR / "all_intensity_variants_best_single_parameters.csv", index=False)
    df_all_combo.to_csv(BASE_OUTPUT_DIR / "all_intensity_variants_best_combinations.csv", index=False)

    plt.figure(figsize=(14, 8))
    for variant_name, grp in df_all_combo.groupby("intensity_variant"):
        plt.plot(grp["combination"], grp[RANK_BY], marker="o", linestyle="-", label=variant_name)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(f"{RANK_BY} [mm]")
    plt.title(f"Best combinations across both intensity variants ({RANK_BY})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(BASE_OUTPUT_DIR / "comparison_best_combinations_between_intensity_variants.png", dpi=200)
    plt.close()

    print("\n" + "=" * 100)
    print("ALL DONE")
    print("=" * 100)
    print(f"Base output:        {BASE_OUTPUT_DIR}")
    print(f"Measurement output: {MEAS_BASE_DIR}")
    print(f"Selected metrics:   {COMPARISON_METRICS}")
    print(f"Intensity variants: {[v['name'] for v in INTENSITY_VARIANTS]}")


if __name__ == "__main__":
    main()