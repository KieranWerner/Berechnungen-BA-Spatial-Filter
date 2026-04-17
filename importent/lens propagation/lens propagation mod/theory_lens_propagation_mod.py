from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# SZENARIEN
# ============================================================

SCENARIOS = {
    "baseline": {},

    "opt_tilt_x_only": {
        "USE_EXTRA_PHASE": True,
        "TILT_X": 0.000455,
    },
    "opt_tilt_y_only": {
        "USE_EXTRA_PHASE": True,
        "TILT_Y": -0.000333,
    },
    "opt_astig_x_only": {
        "USE_EXTRA_PHASE": True,
        "ASTIG_X": -0.005051,
    },
    "opt_astig_y_only": {
        "USE_EXTRA_PHASE": True,
        "ASTIG_Y": -0.005051,
    },
    "opt_defocus_only": {
        "USE_EXTRA_PHASE": True,
        "DEFOCUS": 0.013461,
    },
    "opt_blur_only": {
        "USE_DETECTOR": True,
        "BLUR_SIGMA_PX": 0.0,
    },
    "opt_z_offset_only": {
        "USE_Z_OFFSET": True,
        "Z_OFFSET_M": -0.300000,
    },
    "opt_spherical_only": {
        "USE_SPHERICAL_ABERRATION": True,
        "SPHERICAL_COEFF": -1.0,
    },
    "opt_coma_x_only": {
        "USE_COMA": True,
        "COMA_X": 1.0,
        "COMA_Y": 0.0,
    },
    "opt_coma_y_only": {
        "USE_COMA": True,
        "COMA_X": 0.0,
        "COMA_Y": 1.0,
    },

    "opt_all_except_defocus": {
        "USE_EXTRA_PHASE": True,
        "TILT_X": 0.000455,
        "TILT_Y": -0.000333,
        "ASTIG_X": -0.005051,
        "ASTIG_Y": -0.005051,

        "USE_DETECTOR": True,
        "BLUR_SIGMA_PX": 0.0,

        "USE_Z_OFFSET": True,
        "Z_OFFSET_M": -0.300000,

        "USE_SPHERICAL_ABERRATION": True,
        "SPHERICAL_COEFF": -1.0,

        "USE_COMA": True,
        "COMA_X": 1.0,
        "COMA_Y": 1.0,
    },
}

ACTIVE_SCENARIO = "opt_all_except_defocus"


# ============================================================
# BASIS-KONFIGURATION
# ============================================================

CFG = {
    "PHA_FILE": Path(r"C:\Users\User\Desktop\combined PHA front csv\average_PHA no bg substract.csv"),
    "INT_FILE": Path(r"C:\Users\User\Desktop\combined average INT front bg substracted\average_INT front bg substracted.csv"),

    "OUTPUT_DIR": Path.home() / "Desktop" / "lens propagation",

    "WAVELENGTH_VACUUM_M": 800e-9,

    # Skalierung des Startfelds
    "USE_INPUT_D4SIGMA_FOR_SCALING": True,
    "INPUT_D4SIGMA_MM": 13.7,
    "THEORY_PIXEL_SIZE_UM": None,

    # Maske
    "INTENSITY_THRESHOLD_REL": 0.02,
    "USE_CIRCLE_MASK": True,
    "CIRCLE_RADIUS_PX": 180,
    "CIRCLE_CENTER_X": None,
    "CIRCLE_CENTER_Y": None,
    "USE_SOFT_MASK": False,
    "SOFT_MASK_SIGMA_PX": 3.0,

    # Phase aus Messung
    "USE_PHASE_MAP": True,
    "PHASE_SIGN": -1.0,

    # Linse
    "APPLY_LENS_PHASE": True,
    "LENS_FOCAL_LENGTH_M": 5.0,
    "LENS_X_OFFSET_M": 0.0,
    "LENS_Y_OFFSET_M": 0.0,

    # Zusatzterme
    "USE_EXTRA_PHASE": False,
    "TILT_X": 0.0,
    "TILT_Y": 0.0,
    "ASTIG_X": 0.0,
    "ASTIG_Y": 0.0,
    "DEFOCUS": 0.0,

    "USE_Z_OFFSET": False,
    "Z_OFFSET_M": 0.0,

    "USE_SPHERICAL_ABERRATION": False,
    "SPHERICAL_COEFF": 0.0,

    "USE_COMA": False,
    "COMA_X": 0.0,
    "COMA_Y": 0.0,

    "USE_DETECTOR": False,
    "BLUR_SIGMA_PX": 0.0,

    # Propagation
    "Z_LIST_M": [
        0.0, 0.5, 1.0, 1.5, 2.0, 2.5,
        3.0, 3.5, 4.0, 4.5, 5.0, 5.5,
        6.0, 7.0, 8.0, 9.0, 10.0
    ],
    "PAD_FACTOR": 4,

    # FWHM
    "FWHM_DEFINITION": "cross_section",
}


# ============================================================
# HILFSFUNKTIONEN
# ============================================================

def build_cfg() -> dict:
    cfg = dict(CFG)
    scenario = SCENARIOS[ACTIVE_SCENARIO]
    cfg.update(scenario)
    return cfg


def load_csv_auto(path: Path) -> np.ndarray:
    for delim in [",", ";", "\t"]:
        try:
            arr = np.loadtxt(path, delimiter=delim)
            if arr.ndim == 2 and arr.size > 0:
                return arr.astype(np.float64)
        except Exception:
            pass
    raise ValueError(f"Could not parse file: {path}")


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


def gaussian_blur_fallback(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img.copy()

    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(img, sigma=sigma)
    except Exception:
        out = img.astype(float).copy()
        n = max(1, int(round(sigma)))
        for _ in range(n):
            out = (
                np.roll(out, 1, 0) + np.roll(out, -1, 0) +
                np.roll(out, 1, 1) + np.roll(out, -1, 1) +
                4.0 * out
            ) / 8.0
        return out


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


def encircled_energy_radius(I: np.ndarray, R: np.ndarray, frac: float, P: float) -> float:
    order = np.argsort(R.ravel())
    r_sorted = R.ravel()[order]
    i_sorted = I.ravel()[order]
    cdf = np.cumsum(i_sorted) / P
    idx = min(np.searchsorted(cdf, frac), len(r_sorted) - 1)
    return float(r_sorted[idx])


def area_equivalent_diameter(I: np.ndarray, dx: float, dy: float, rel_threshold: float) -> float:
    peak = float(np.max(I))
    if peak <= 0:
        return np.nan
    mask = I >= rel_threshold * peak
    area = float(mask.sum()) * dx * dy
    if area <= 0:
        return np.nan
    return float(np.sqrt(4.0 * area / np.pi))


def fwhm_1d(axis_coords: np.ndarray, profile: np.ndarray) -> float:
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
        left = x1 + (half - y1) * (x2 - x1) / (y2 - y1) if y2 != y1 else axis_coords[i_left]
    else:
        left = axis_coords[i_left]

    if i_right < profile.size - 1:
        x1, x2 = axis_coords[i_right], axis_coords[i_right + 1]
        y1, y2 = profile[i_right], profile[i_right + 1]
        right = x1 + (half - y1) * (x2 - x1) / (y2 - y1) if y2 != y1 else axis_coords[i_right]
    else:
        right = axis_coords[i_right]

    return float(right - left)


def beam_metrics_units(I: np.ndarray, dx: float, dy: float, fwhm_definition: str = "cross_section") -> dict[str, float]:
    I = np.clip(np.asarray(I, float), 0, None)
    x0_px, y0_px, P, X, Y = centroid_pixels(I)

    Xc = (X - x0_px) * dx
    Yc = (Y - y0_px) * dy
    R = np.sqrt(Xc**2 + Yc**2)

    sigma_x = float(np.sqrt((I * Xc**2).sum() / P))
    sigma_y = float(np.sqrt((I * Yc**2).sum() / P))
    d4sigma_x = 4.0 * sigma_x
    d4sigma_y = 4.0 * sigma_y
    d4sigma_avg = 0.5 * (d4sigma_x + d4sigma_y)

    r50 = encircled_energy_radius(I, R, 0.50, P)
    r86 = encircled_energy_radius(I, R, 0.86, P)
    d_ee50 = 2.0 * r50
    d_ee86 = 2.0 * r86

    d_area_50pct = area_equivalent_diameter(I, dx, dy, 0.50)
    d_area_13p5pct = area_equivalent_diameter(I, dx, dy, np.exp(-2.0))

    x_axis = (np.arange(I.shape[1]) - x0_px) * dx
    y_axis = (np.arange(I.shape[0]) - y0_px) * dy

    cx_i = int(np.clip(round(x0_px), 0, I.shape[1] - 1))
    cy_i = int(np.clip(round(y0_px), 0, I.shape[0] - 1))

    if fwhm_definition == "cross_section":
        profile_x = I[cy_i, :]
        profile_y = I[:, cx_i]
    elif fwhm_definition == "integrated":
        profile_x = I.sum(axis=0)
        profile_y = I.sum(axis=1)
    else:
        raise ValueError("FWHM_DEFINITION must be 'cross_section' or 'integrated'.")

    fwhm_x = fwhm_1d(x_axis, profile_x)
    fwhm_y = fwhm_1d(y_axis, profile_y)
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
# PHYSIK
# ============================================================

def angular_spectrum(U0: np.ndarray, dx: float, dy: float, wavelength: float, z: float, pad_factor: int = 4) -> np.ndarray:
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


def build_initial_field(intensity: np.ndarray, phase_waves: np.ndarray | None, valid_mask: np.ndarray, cfg: dict) -> np.ndarray:
    intensity = np.asarray(intensity, dtype=float)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    if not np.any(valid_mask):
        raise ValueError("Valid mask is empty.")

    int_norm = np.zeros_like(intensity, dtype=float)
    int_norm[valid_mask] = intensity[valid_mask] / np.nanmax(intensity[valid_mask])

    A = np.sqrt(np.clip(int_norm, 0, None))

    if cfg["USE_SOFT_MASK"]:
        A *= apply_soft_mask(valid_mask, sigma=float(cfg["SOFT_MASK_SIGMA_PX"]))
    else:
        A[~valid_mask] = 0.0

    if cfg["USE_PHASE_MAP"]:
        if phase_waves is None:
            raise ValueError("USE_PHASE_MAP=True, aber keine PHA-Datei vorhanden.")
        phi = float(cfg["PHASE_SIGN"]) * 2.0 * np.pi * np.nan_to_num(phase_waves)
        U0 = A * np.exp(1j * phi)
    else:
        U0 = A.astype(np.complex128)

    return U0


def apply_all_phases(U0: np.ndarray, dx: float, dy: float, wavelength: float, cfg: dict) -> np.ndarray:
    ny, nx = U0.shape
    x = (np.arange(nx) - nx // 2) * dx
    y = (np.arange(ny) - ny // 2) * dy
    X, Y = np.meshgrid(x, y)

    k = 2.0 * np.pi / wavelength
    phi = np.zeros_like(X, dtype=float)

    # Linse
    if cfg["APPLY_LENS_PHASE"]:
        f = float(cfg["LENS_FOCAL_LENGTH_M"])
        x0 = float(cfg["LENS_X_OFFSET_M"])
        y0 = float(cfg["LENS_Y_OFFSET_M"])
        phi += -k * (((X - x0) ** 2 + (Y - y0) ** 2) / (2.0 * f))

    # Zusätzliche einfache Phasen
    if cfg["USE_EXTRA_PHASE"]:
        phi += k * float(cfg["DEFOCUS"]) * (X**2 + Y**2)
        phi += k * float(cfg["TILT_X"]) * X
        phi += k * float(cfg["TILT_Y"]) * Y
        phi += k * float(cfg["ASTIG_X"]) * X**2
        phi += k * float(cfg["ASTIG_Y"]) * Y**2

    # Spherical Aberration
    if cfg["USE_SPHERICAL_ABERRATION"]:
        r2 = X**2 + Y**2
        r2_norm = r2 / max(float(np.max(r2)), 1e-30)
        phi += float(cfg["SPHERICAL_COEFF"]) * (r2_norm ** 2)

    # Coma
    if cfg["USE_COMA"]:
        r2 = X**2 + Y**2
        r_norm = np.sqrt(r2 / max(float(np.max(r2)), 1e-30))
        phi += float(cfg["COMA_X"]) * (X / max(np.max(np.abs(X)), 1e-30)) * (r_norm ** 2)
        phi += float(cfg["COMA_Y"]) * (Y / max(np.max(np.abs(Y)), 1e-30)) * (r_norm ** 2)

    return U0 * np.exp(1j * phi)


def determine_dx_dy_m(intensity: np.ndarray, valid_mask: np.ndarray, cfg: dict) -> tuple[float, float]:
    if cfg["USE_INPUT_D4SIGMA_FOR_SCALING"]:
        int_norm = np.zeros_like(intensity, dtype=float)
        int_norm[valid_mask] = intensity[valid_mask] / np.nanmax(intensity[valid_mask])

        tmp = beam_metrics_units(int_norm, 1.0, 1.0, fwhm_definition=cfg["FWHM_DEFINITION"])
        d4_px = tmp["d4sigma_avg"]
        if not np.isfinite(d4_px) or d4_px <= 0:
            raise RuntimeError("Could not determine D4sigma in pixels for scaling.")

        dx_m = (float(cfg["INPUT_D4SIGMA_MM"]) / d4_px) * 1e-3
        dy_m = dx_m
        return dx_m, dy_m

    pitch_um = cfg["THEORY_PIXEL_SIZE_UM"]
    if pitch_um is None:
        raise RuntimeError(
            "Bitte entweder USE_INPUT_D4SIGMA_FOR_SCALING=True setzen oder THEORY_PIXEL_SIZE_UM angeben."
        )

    dx_m = float(pitch_um) * 1e-6
    dy_m = dx_m
    return dx_m, dy_m


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    cfg = build_cfg()

    output_dir = Path(cfg["OUTPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)

    phase_waves = load_csv_auto(Path(cfg["PHA_FILE"])) if cfg["USE_PHASE_MAP"] else None
    intensity = load_csv_auto(Path(cfg["INT_FILE"]))

    if phase_waves is not None and phase_waves.shape != intensity.shape:
        raise ValueError(f"Shape mismatch: phase {phase_waves.shape} vs intensity {intensity.shape}")

    valid_mask, threshold_abs, cx, cy = make_valid_mask(
        intensity=intensity,
        threshold_rel=float(cfg["INTENSITY_THRESHOLD_REL"]),
        use_circle=bool(cfg["USE_CIRCLE_MASK"]),
        circle_radius_px=float(cfg["CIRCLE_RADIUS_PX"]),
        cx=cfg["CIRCLE_CENTER_X"],
        cy=cfg["CIRCLE_CENTER_Y"],
    )

    dx_m, dy_m = determine_dx_dy_m(intensity, valid_mask, cfg)
    wavelength = float(cfg["WAVELENGTH_VACUUM_M"])

    U0 = build_initial_field(
        intensity=intensity,
        phase_waves=phase_waves,
        valid_mask=valid_mask,
        cfg=cfg,
    )

    U0 = apply_all_phases(U0, dx_m, dy_m, wavelength, cfg)

    rows = []

    for z in cfg["Z_LIST_M"]:
        z = float(z)
        z_eff = z + float(cfg["Z_OFFSET_M"]) if cfg["USE_Z_OFFSET"] else z

        Uz = U0 if abs(z_eff) < 1e-15 else angular_spectrum(
            U0, dx_m, dy_m, wavelength, z_eff, pad_factor=int(cfg["PAD_FACTOR"])
        )

        Iz = np.abs(Uz) ** 2

        if cfg["USE_DETECTOR"] and float(cfg["BLUR_SIGMA_PX"]) > 0:
            Iz = gaussian_blur_fallback(Iz, float(cfg["BLUR_SIGMA_PX"]))

        m = beam_metrics_units(Iz, dx_m, dy_m, fwhm_definition=cfg["FWHM_DEFINITION"])

        rows.append({
            "z_m": z,
            "z_cm": z * 100.0,
            "z_effective_m": z_eff,
            "centroid_x_mm": m["centroid_x"] * 1e3,
            "centroid_y_mm": m["centroid_y"] * 1e3,
            "D4sigma_x_mm": m["d4sigma_x"] * 1e3,
            "D4sigma_y_mm": m["d4sigma_y"] * 1e3,
            "D4sigma_avg_mm": m["d4sigma_avg"] * 1e3,
            "D_EE50_mm": m["d_ee50"] * 1e3,
            "D_EE86_mm": m["d_ee86"] * 1e3,
            "D_area_50pct_mm": m["d_area_50pct"] * 1e3,
            "D_area_13p5pct_mm": m["d_area_13p5pct"] * 1e3,
            "FWHM_x_mm": m["fwhm_x"] * 1e3,
            "FWHM_y_mm": m["fwhm_y"] * 1e3,
            "FWHM_mm": m["fwhm_avg"] * 1e3,
        })

    stem = f"theory_lens_propagation_{ACTIVE_SCENARIO}"

    csv_path = output_dir / f"{stem}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    meta = {
        "active_scenario": ACTIVE_SCENARIO,
        "scenario_values": SCENARIOS[ACTIVE_SCENARIO],
        "wavelength_nm": wavelength * 1e9,
        "dx_um": dx_m * 1e6,
        "dy_um": dy_m * 1e6,
        "threshold_abs": threshold_abs,
        "valid_pixels": int(valid_mask.sum()),
        "mask_center_x_px": cx,
        "mask_center_y_px": cy,
        "lens_focal_length_m": cfg["LENS_FOCAL_LENGTH_M"],
        "fwhm_definition": cfg["FWHM_DEFINITION"],
    }
    with open(output_dir / f"{stem}_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    z = np.array([r["z_m"] for r in rows], dtype=float)

    plt.figure(figsize=(11, 7))
    plt.plot(z, [r["D4sigma_avg_mm"] for r in rows], marker="o", label="D4σ")
    plt.plot(z, [r["D_EE50_mm"] for r in rows], marker="o", label="EE50")
    plt.plot(z, [r["D_EE86_mm"] for r in rows], marker="o", label="EE86")
    plt.plot(z, [r["D_area_50pct_mm"] for r in rows], marker="o", label="Area50%")
    plt.plot(z, [r["D_area_13p5pct_mm"] for r in rows], marker="o", label="Area13.5%")
    plt.plot(z, [r["FWHM_mm"] for r in rows], marker="o", label=f"FWHM ({cfg['FWHM_DEFINITION']})")

    plt.xlabel("z [m]")
    plt.ylabel("Beam size [mm]")
    plt.title(f"Theory lens propagation ({ACTIVE_SCENARIO})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    png_path = output_dir / f"{stem}.png"
    plt.savefig(png_path, dpi=220)
    plt.show()

    print("=" * 80)
    print("MODIFIED THEORY PROPAGATION FINISHED")
    print("=" * 80)
    print(f"Scenario: {ACTIVE_SCENARIO}")
    print(f"CSV:      {csv_path}")
    print(f"PNG:      {png_path}")


if __name__ == "__main__":
    main()