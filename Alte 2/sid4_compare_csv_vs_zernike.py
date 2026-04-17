from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import center_of_mass, distance_transform_edt, shift


# =========================
# Default configuration
# =========================
DEFAULT_INPUT_DIR = Path.home() / "Desktop" / "Eingabe"
DEFAULT_OUTPUT_DIRNAME = "propagation_results"
DEFAULT_DISTANCE_M = 10.0
DEFAULT_PAD_FACTOR = 4
DEFAULT_APOD_EDGE_PX = 6.0
DEFAULT_PHASE_SIGN = -1.0  # For the uploaded SID4 exports this matched the real 10 m output much better.
DEFAULT_PHASE_SCALE = 1.0

# Small, robust grid search against INT output.csv.
FIT_SIGN_OPTIONS = (-1.0, 1.0)
FIT_PHASE_SCALES = tuple(np.linspace(0.75, 1.35, 13))

# Optional extra search dimensions. Keep these small so the script stays practical.
FIT_PIXEL_SCALES = (1.0,)   # Example for later: (0.95, 1.0, 1.05)
FIT_Z_VALUES_M = (DEFAULT_DISTANCE_M,)  # Example for later: (9.5, 10.0, 10.5)


@dataclass
class Meta:
    wavelength_m: float
    pupil_size_m: float
    phase_ptv_rad: Optional[float]
    phase_rms_rad: Optional[float]
    max_intensity: Optional[float]
    tilt_x_mrad: Optional[float]
    tilt_y_mrad: Optional[float]
    strehl_ratio: Optional[float]
    zernike_coeffs: Dict[int, float]
    zernike_labels: Dict[int, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "SID4 propagation tool: builds the complex field from intensity + phase CSV, "
            "reconstructs a Zernike phase from ACC XML, propagates both with the Angular Spectrum "
            "method, and compares the result to a measured output intensity."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--acc", type=Path, default=None, help="Path to ACC ... .xml")
    parser.add_argument("--pha", type=Path, default=None, help="Path to PHA ... .csv")
    parser.add_argument("--int-in", type=Path, default=None, help="Path to input intensity CSV")
    parser.add_argument("--int-out", type=Path, default=None, help="Path to measured output intensity CSV")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for results")
    parser.add_argument("--distance-m", type=float, default=DEFAULT_DISTANCE_M)
    parser.add_argument("--pad-factor", type=int, default=DEFAULT_PAD_FACTOR)
    parser.add_argument("--apod-edge-px", type=float, default=DEFAULT_APOD_EDGE_PX)
    parser.add_argument("--phase-sign", type=float, default=DEFAULT_PHASE_SIGN)
    parser.add_argument("--phase-scale", type=float, default=DEFAULT_PHASE_SCALE)
    parser.add_argument("--fit-to-output", action="store_true", help="Coarse fit of phase sign/scale against INT output")
    return parser.parse_args()


def smart_load_csv(path: Path) -> np.ndarray:
    last_error: Optional[Exception] = None
    for delimiter in [",", ";", "\t", None]:
        try:
            arr = np.loadtxt(path, delimiter=delimiter)
            if arr.ndim != 2:
                raise ValueError(f"Expected a 2D array in {path}, got shape {arr.shape}")
            if not np.isfinite(arr).all():
                raise ValueError(f"Non-finite values found in {path}")
            return arr.astype(np.float64)
        except Exception as exc:  # noqa: PERF203
            last_error = exc
    raise RuntimeError(f"Could not read CSV file {path}. Last error: {last_error}")


def pick_single(candidates: Iterable[Path], label: str) -> Path:
    items = sorted(set(candidates))
    if len(items) == 1:
        return items[0]
    if len(items) == 0:
        raise FileNotFoundError(f"No file found for {label}.")
    joined = "\n  ".join(str(x) for x in items)
    raise RuntimeError(f"Multiple candidates found for {label}; please specify it explicitly:\n  {joined}")


def auto_detect_files(input_dir: Path) -> Tuple[Path, Path, Path, Optional[Path]]:
    acc = pick_single(input_dir.glob("ACC*.xml"), "ACC XML")
    pha = pick_single(input_dir.glob("PHA*.csv"), "phase CSV")

    intensity_candidates = [
        p for p in input_dir.glob("INT*.csv")
        if "output" not in p.name.lower()
    ]
    int_in = pick_single(intensity_candidates, "input intensity CSV")

    output_candidates = [p for p in input_dir.glob("INT*output*.csv")]
    int_out = output_candidates[0] if len(output_candidates) == 1 else None
    return acc, pha, int_in, int_out


def parse_acc_xml(path: Path) -> Meta:
    root = ET.parse(path).getroot()

    wavelength_nm = float(root.findtext("./UserProfile/WaveLength"))

    phase_ptv_rad = None
    phase_rms_rad = None
    max_intensity = None
    tilt_x_mrad = None
    tilt_y_mrad = None
    strehl_ratio = None
    pupil_size_m = None
    zernike_coeffs: Dict[int, float] = {}
    zernike_labels: Dict[int, str] = {}

    for data in root.findall("Data"):
        data_type = data.findtext("DataType")
        label = (data.findtext("Label") or "").strip()
        value_text = (data.findtext("Value") or "").strip()
        index_text = (data.findtext("Index") or "").strip()

        if not value_text:
            continue

        if data_type == "Phase" and label == "PtV Phase":
            phase_ptv_rad = float(value_text)
        elif data_type == "Phase" and label == "RMS Phase":
            phase_rms_rad = float(value_text)
        elif data_type == "Intensity" and label == "Max Intensity":
            max_intensity = float(value_text)
        elif data_type == "Tilt" and label == "Tilt X (mrad)":
            tilt_x_mrad = float(value_text)
        elif data_type == "Tilt" and label == "Tilt Y (mrad)":
            tilt_y_mrad = float(value_text)
        elif data_type == "FarField" and label == "Strehl ratio":
            strehl_ratio = float(value_text)
        elif data_type == "Projection" and label == "PupilSize (mm)":
            pupil_size_m = float(value_text) * 1e-3

        if data_type == "Projection" and index_text.isdigit():
            idx = int(index_text)
            zernike_coeffs[idx] = float(value_text)
            zernike_labels[idx] = label or f"Projection_{idx}"

    if pupil_size_m is None:
        raise ValueError("Could not find 'PupilSize (mm)' in ACC XML.")

    return Meta(
        wavelength_m=wavelength_nm * 1e-9,
        pupil_size_m=pupil_size_m,
        phase_ptv_rad=phase_ptv_rad,
        phase_rms_rad=phase_rms_rad,
        max_intensity=max_intensity,
        tilt_x_mrad=tilt_x_mrad,
        tilt_y_mrad=tilt_y_mrad,
        strehl_ratio=strehl_ratio,
        zernike_coeffs=zernike_coeffs,
        zernike_labels=zernike_labels,
    )


# Mapping that matches the SID4 XML labels seen in the uploaded files.
# It is NOT the classic Noll numbering. It is the order used by SID4 here:
# piston, tilt x, tilt y, defocus, astig 0, astig 45, coma x, coma y, ...
SID4_ZERNIKE_MAP: Dict[int, Tuple[int, int]] = {
    1: (0, 0),
    2: (1, 1),
    3: (1, -1),
    4: (2, 0),
    5: (2, 2),
    6: (2, -2),
    7: (3, 1),
    8: (3, -1),
    9: (3, 3),
    10: (3, -3),
    11: (4, 0),
    12: (4, 2),
    13: (4, -2),
    14: (4, 4),
    15: (4, -4),
    16: (5, 1),
    17: (5, -1),
    18: (5, 3),
    19: (5, -3),
    20: (5, 5),
    21: (5, -5),
    22: (6, 0),
    23: (6, 2),
    24: (6, -2),
    25: (6, 4),
    26: (6, -4),
    27: (6, 6),
    28: (6, -6),
}


def radial_poly(n: int, m_abs: int, rho: np.ndarray) -> np.ndarray:
    if (n - m_abs) % 2 != 0:
        return np.zeros_like(rho)
    out = np.zeros_like(rho)
    for s in range((n - m_abs) // 2 + 1):
        coeff = ((-1) ** s) * math.factorial(n - s)
        coeff /= math.factorial(s)
        coeff /= math.factorial((n + m_abs) // 2 - s)
        coeff /= math.factorial((n - m_abs) // 2 - s)
        out += coeff * rho ** (n - 2 * s)
    return out


def zernike_nm(n: int, m: int, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
    m_abs = abs(m)
    R = radial_poly(n, m_abs, rho)
    if m == 0:
        return math.sqrt(n + 1) * R
    if m > 0:
        return math.sqrt(2 * (n + 1)) * R * np.cos(m_abs * theta)
    return math.sqrt(2 * (n + 1)) * R * np.sin(m_abs * theta)


def build_mask(intensity: np.ndarray, phase: np.ndarray) -> np.ndarray:
    mask = (intensity > 0) & np.isfinite(phase)
    if not np.any(mask):
        raise ValueError("No valid overlap between intensity and phase.")
    return mask


def equivalent_pupil_geometry(mask: np.ndarray) -> Tuple[float, float, float]:
    ys, xs = np.nonzero(mask)
    cy = float(ys.mean())
    cx = float(xs.mean())
    radius_px = math.sqrt(mask.sum() / math.pi)
    return cy, cx, radius_px


def reconstruct_phase_from_zernikes(shape: Tuple[int, int], mask: np.ndarray, coeffs: Dict[int, float]) -> np.ndarray:
    cy, cx, radius_px = equivalent_pupil_geometry(mask)
    y, x = np.indices(shape)
    xn = (x - cx) / radius_px
    yn = (y - cy) / radius_px
    rho = np.sqrt(xn**2 + yn**2)
    theta = np.arctan2(yn, xn)

    phase = np.zeros(shape, dtype=np.float64)
    for idx, coeff in coeffs.items():
        if idx not in SID4_ZERNIKE_MAP:
            continue
        n, m = SID4_ZERNIKE_MAP[idx]
        phase += coeff * zernike_nm(n, m, rho, theta)

    phase[~mask] = 0.0
    if np.any(mask):
        phase[mask] -= phase[mask].mean()
    return phase


def soft_mask_from_binary(mask: np.ndarray, edge_px: float) -> np.ndarray:
    dist_in = distance_transform_edt(mask)
    out = np.zeros(mask.shape, dtype=np.float64)
    out[mask] = np.minimum(dist_in[mask] / edge_px, 1.0)
    return out


def angular_spectrum(
    u0: np.ndarray,
    dx: float,
    dy: float,
    wavelength: float,
    z: float,
    pad_factor: int = 4,
    remove_evanescent: bool = True,
) -> np.ndarray:
    ny, nx = u0.shape
    py = int((pad_factor - 1) * ny / 2)
    px = int((pad_factor - 1) * nx / 2)
    u = np.pad(u0, ((py, py), (px, px)), mode="constant")

    Ny, Nx = u.shape
    fx = np.fft.fftfreq(Nx, d=dx)
    fy = np.fft.fftfreq(Ny, d=dy)
    FX, FY = np.meshgrid(fx, fy)

    k = 2 * np.pi / wavelength
    kz_sq = k**2 - (2 * np.pi * FX) ** 2 - (2 * np.pi * FY) ** 2
    kz = np.sqrt(np.maximum(kz_sq, 0.0))
    H = np.exp(1j * kz * z)
    if remove_evanescent:
        H[kz_sq < 0] = 0.0

    uz = np.fft.ifft2(np.fft.fft2(u) * H)
    return uz[py:py + ny, px:px + nx]


def align_to_target(pred: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)
    pred = pred / pred.max() if pred.max() > 0 else pred
    target = target / target.max() if target.max() > 0 else target

    cp = center_of_mass(pred)
    ct = center_of_mass(target)
    shift_yx = (float(ct[0] - cp[0]), float(ct[1] - cp[1]))
    aligned = shift(pred, shift_yx, order=1, mode="constant", cval=0.0)
    aligned = aligned / aligned.max() if aligned.max() > 0 else aligned
    return aligned, shift_yx


def compare_images(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    aligned, shift_yx = align_to_target(pred, target)
    target_n = target / target.max() if target.max() > 0 else target
    corr = float(np.corrcoef(aligned.ravel(), target_n.ravel())[0, 1])
    rmse = float(np.sqrt(np.mean((aligned - target_n) ** 2)))
    return {
        "corr": corr,
        "rmse": rmse,
        "shift_y_px": shift_yx[0],
        "shift_x_px": shift_yx[1],
    }


def propagate_intensity(
    intensity_in: np.ndarray,
    phase: np.ndarray,
    mask: np.ndarray,
    meta: Meta,
    distance_m: float,
    apod_edge_px: float,
    phase_sign: float,
    phase_scale: float,
    pixel_scale: float,
    pad_factor: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    _, _, radius_px = equivalent_pupil_geometry(mask)
    effective_diameter_px = 2.0 * radius_px
    dx = meta.pupil_size_m / effective_diameter_px
    dx *= pixel_scale
    dy = dx

    amplitude = np.sqrt(np.clip(intensity_in, 0.0, None))
    max_amp = float(amplitude[mask].max())
    if max_amp <= 0:
        raise ValueError("Input intensity has no positive values inside the mask.")
    amplitude /= max_amp
    amplitude *= soft_mask_from_binary(mask, apod_edge_px)

    phase_used = np.zeros_like(phase, dtype=np.float64)
    phase_used[mask] = phase_sign * phase_scale * phase[mask]

    u0 = amplitude * np.exp(1j * phase_used)
    uz = angular_spectrum(u0, dx, dy, meta.wavelength_m, distance_m, pad_factor=pad_factor)
    intensity_out = np.abs(uz) ** 2
    if intensity_out.max() > 0:
        intensity_out /= intensity_out.max()

    info = {
        "dx_um": dx * 1e6,
        "dy_um": dy * 1e6,
        "effective_pupil_diameter_px": effective_diameter_px,
        "phase_sign": phase_sign,
        "phase_scale": phase_scale,
        "pixel_scale": pixel_scale,
        "distance_m": distance_m,
        "pad_factor": pad_factor,
    }
    return intensity_out, info


def fit_to_output(
    intensity_in: np.ndarray,
    phase: np.ndarray,
    mask: np.ndarray,
    meta: Meta,
    measured_output: np.ndarray,
    apod_edge_px: float,
    pad_factor: int,
) -> Dict[str, float]:
    best: Optional[Dict[str, float]] = None

    for sign in FIT_SIGN_OPTIONS:
        for scale in FIT_PHASE_SCALES:
            for pixel_scale in FIT_PIXEL_SCALES:
                for distance_m in FIT_Z_VALUES_M:
                    pred, info = propagate_intensity(
                        intensity_in=intensity_in,
                        phase=phase,
                        mask=mask,
                        meta=meta,
                        distance_m=distance_m,
                        apod_edge_px=apod_edge_px,
                        phase_sign=sign,
                        phase_scale=scale,
                        pixel_scale=pixel_scale,
                        pad_factor=max(2, pad_factor // 2),
                    )
                    metrics = compare_images(pred, measured_output)
                    candidate = {**info, **metrics}
                    candidate["score"] = candidate["corr"] - candidate["rmse"]

                    if best is None or candidate["score"] > best["score"]:
                        best = candidate

    assert best is not None
    return best


def save_array_csv(path: Path, arr: np.ndarray) -> None:
    np.savetxt(path, arr, delimiter=",", fmt="%.8f")


def plot_phase_summary(
    path: Path,
    intensity_in: np.ndarray,
    phase_csv: np.ndarray,
    phase_zernike: np.ndarray,
    phase_residual: np.ndarray,
    measured_output: Optional[np.ndarray],
    meta: Meta,
    mask: np.ndarray,
    dx_um: float,
) -> None:
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    axs[0, 0].imshow(intensity_in / intensity_in.max(), cmap="inferno")
    axs[0, 0].set_title("Eingangsintensität")
    axs[0, 1].imshow(np.where(mask, phase_csv, np.nan), cmap="coolwarm")
    axs[0, 1].set_title("Phase aus CSV [rad]")
    axs[0, 2].imshow(np.where(mask, phase_zernike, np.nan), cmap="coolwarm")
    axs[0, 2].set_title("Phase aus Zernikes [rad]")
    axs[1, 0].imshow(np.where(mask, phase_residual, np.nan), cmap="coolwarm")
    axs[1, 0].set_title("CSV - Zernike [rad]")

    if measured_output is not None:
        axs[1, 1].imshow(measured_output / measured_output.max(), cmap="inferno")
        axs[1, 1].set_title("INT output (real)")
    else:
        axs[1, 1].axis("off")

    z4 = meta.zernike_coeffs.get(4, float("nan"))
    z5 = meta.zernike_coeffs.get(5, float("nan"))
    z6 = meta.zernike_coeffs.get(6, float("nan"))
    z11 = meta.zernike_coeffs.get(11, float("nan"))
    text = "\n".join([
        f"lambda = {meta.wavelength_m*1e9:.0f} nm",
        f"Pupil = {meta.pupil_size_m*1e3:.3f} mm",
        f"dx ≈ {dx_um:.2f} um",
        f"Phase PtV = {meta.phase_ptv_rad:.3f} rad" if meta.phase_ptv_rad is not None else "Phase PtV = n/a",
        f"Phase RMS = {meta.phase_rms_rad:.3f} rad" if meta.phase_rms_rad is not None else "Phase RMS = n/a",
        f"Tilt X = {meta.tilt_x_mrad:.3f} mrad" if meta.tilt_x_mrad is not None else "Tilt X = n/a",
        f"Tilt Y = {meta.tilt_y_mrad:.3f} mrad" if meta.tilt_y_mrad is not None else "Tilt Y = n/a",
        f"Strehl = {meta.strehl_ratio:.3f}" if meta.strehl_ratio is not None else "Strehl = n/a",
        f"Defocus = {z4:.4f}",
        f"Astig 0 = {z5:.4f}",
        f"Astig 45 = {z6:.4f}",
        f"Spherical = {z11:.4f}",
    ])
    axs[1, 2].axis("off")
    axs[1, 2].text(0.02, 0.98, text, va="top", family="monospace")

    for ax in axs.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_propagation_summary(
    path: Path,
    measured_output: Optional[np.ndarray],
    pred_csv: np.ndarray,
    pred_zernike: np.ndarray,
    pred_csv_fit: Optional[np.ndarray],
    pred_zernike_fit: Optional[np.ndarray],
    csv_metrics: Dict[str, float],
    z_metrics: Dict[str, float],
    csv_fit_metrics: Optional[Dict[str, float]],
    z_fit_metrics: Optional[Dict[str, float]],
) -> None:
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    if measured_output is not None:
        real = measured_output / measured_output.max()
        axs[0, 0].imshow(real, cmap="inferno")
        axs[0, 0].set_title("Real: INT output")
        axs[1, 0].imshow(np.abs(pred_csv - real), cmap="viridis")
        axs[1, 0].set_title("Abweichung CSV vs real")
    else:
        axs[0, 0].axis("off")
        axs[1, 0].axis("off")

    axs[0, 1].imshow(pred_csv, cmap="inferno")
    axs[0, 1].set_title(f"CSV-Phase\ncorr={csv_metrics['corr']:.3f}, rmse={csv_metrics['rmse']:.3f}")
    axs[0, 2].imshow(pred_zernike, cmap="inferno")
    axs[0, 2].set_title(f"Zernike-Phase\ncorr={z_metrics['corr']:.3f}, rmse={z_metrics['rmse']:.3f}")

    if pred_csv_fit is not None and csv_fit_metrics is not None:
        axs[1, 1].imshow(pred_csv_fit, cmap="inferno")
        axs[1, 1].set_title(
            f"CSV fit\n"
            f"corr={csv_fit_metrics['corr']:.3f}, rmse={csv_fit_metrics['rmse']:.3f}\n"
            f"sign={csv_fit_metrics['phase_sign']:.0f}, scale={csv_fit_metrics['phase_scale']:.2f}"
        )
    else:
        axs[1, 1].axis("off")

    if pred_zernike_fit is not None and z_fit_metrics is not None:
        axs[1, 2].imshow(pred_zernike_fit, cmap="inferno")
        axs[1, 2].set_title(
            f"Zernike fit\n"
            f"corr={z_fit_metrics['corr']:.3f}, rmse={z_fit_metrics['rmse']:.3f}\n"
            f"sign={z_fit_metrics['phase_sign']:.0f}, scale={z_fit_metrics['phase_scale']:.2f}"
        )
    else:
        axs[1, 2].axis("off")

    for ax in axs.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir

    acc = args.acc
    pha = args.pha
    int_in = args.int_in
    int_out = args.int_out

    if any(x is None for x in (acc, pha, int_in)):
        detected_acc, detected_pha, detected_int_in, detected_int_out = auto_detect_files(input_dir)
        acc = acc or detected_acc
        pha = pha or detected_pha
        int_in = int_in or detected_int_in
        int_out = int_out or detected_int_out

    if acc is None or pha is None or int_in is None:
        raise RuntimeError("Could not determine the required input files.")

    if args.output_dir is None:
        output_dir = input_dir / DEFAULT_OUTPUT_DIRNAME
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = parse_acc_xml(acc)
    intensity_in = smart_load_csv(int_in)
    phase_csv = smart_load_csv(pha)
    measured_output = smart_load_csv(int_out) if int_out is not None and int_out.exists() else None

    if intensity_in.shape != phase_csv.shape:
        raise ValueError(f"Shape mismatch: intensity {intensity_in.shape} vs phase {phase_csv.shape}")
    if measured_output is not None and measured_output.shape != intensity_in.shape:
        raise ValueError(
            f"Measured output shape {measured_output.shape} differs from input shape {intensity_in.shape}. "
            "This script expects matching grids."
        )

    mask = build_mask(intensity_in, phase_csv)
    phase_csv = phase_csv.astype(np.float64)
    phase_csv[~mask] = 0.0
    phase_csv[mask] -= phase_csv[mask].mean()

    phase_zernike = reconstruct_phase_from_zernikes(phase_csv.shape, mask, meta.zernike_coeffs)
    phase_residual = np.zeros_like(phase_csv)
    phase_residual[mask] = phase_csv[mask] - phase_zernike[mask]

    pred_csv, info_csv = propagate_intensity(
        intensity_in=intensity_in,
        phase=phase_csv,
        mask=mask,
        meta=meta,
        distance_m=args.distance_m,
        apod_edge_px=args.apod_edge_px,
        phase_sign=args.phase_sign,
        phase_scale=args.phase_scale,
        pixel_scale=1.0,
        pad_factor=args.pad_factor,
    )
    pred_zernike, info_z = propagate_intensity(
        intensity_in=intensity_in,
        phase=phase_zernike,
        mask=mask,
        meta=meta,
        distance_m=args.distance_m,
        apod_edge_px=args.apod_edge_px,
        phase_sign=args.phase_sign,
        phase_scale=args.phase_scale,
        pixel_scale=1.0,
        pad_factor=args.pad_factor,
    )

    csv_metrics = compare_images(pred_csv, measured_output) if measured_output is not None else {}
    z_metrics = compare_images(pred_zernike, measured_output) if measured_output is not None else {}

    pred_csv_aligned, _ = align_to_target(pred_csv, measured_output) if measured_output is not None else (pred_csv, (0.0, 0.0))
    pred_zernike_aligned, _ = align_to_target(pred_zernike, measured_output) if measured_output is not None else (pred_zernike, (0.0, 0.0))

    baseline_metrics = compare_images(intensity_in, measured_output) if measured_output is not None else {}

    csv_fit_metrics = None
    z_fit_metrics = None
    pred_csv_fit_aligned = None
    pred_zernike_fit_aligned = None

    if measured_output is not None and args.fit_to_output:
        csv_fit_metrics = fit_to_output(
            intensity_in=intensity_in,
            phase=phase_csv,
            mask=mask,
            meta=meta,
            measured_output=measured_output,
            apod_edge_px=args.apod_edge_px,
            pad_factor=args.pad_factor,
        )
        z_fit_metrics = fit_to_output(
            intensity_in=intensity_in,
            phase=phase_zernike,
            mask=mask,
            meta=meta,
            measured_output=measured_output,
            apod_edge_px=args.apod_edge_px,
            pad_factor=args.pad_factor,
        )

        pred_csv_fit, _ = propagate_intensity(
            intensity_in=intensity_in,
            phase=phase_csv,
            mask=mask,
            meta=meta,
            distance_m=csv_fit_metrics["distance_m"],
            apod_edge_px=args.apod_edge_px,
            phase_sign=csv_fit_metrics["phase_sign"],
            phase_scale=csv_fit_metrics["phase_scale"],
            pixel_scale=csv_fit_metrics["pixel_scale"],
            pad_factor=args.pad_factor,
        )
        pred_zernike_fit, _ = propagate_intensity(
            intensity_in=intensity_in,
            phase=phase_zernike,
            mask=mask,
            meta=meta,
            distance_m=z_fit_metrics["distance_m"],
            apod_edge_px=args.apod_edge_px,
            phase_sign=z_fit_metrics["phase_sign"],
            phase_scale=z_fit_metrics["phase_scale"],
            pixel_scale=z_fit_metrics["pixel_scale"],
            pad_factor=args.pad_factor,
        )
        pred_csv_fit_aligned, _ = align_to_target(pred_csv_fit, measured_output)
        pred_zernike_fit_aligned, _ = align_to_target(pred_zernike_fit, measured_output)

    # Save arrays
    save_array_csv(output_dir / "phase_from_csv.csv", phase_csv)
    save_array_csv(output_dir / "phase_from_zernike.csv", phase_zernike)
    save_array_csv(output_dir / "phase_residual_csv_minus_zernike.csv", phase_residual)
    save_array_csv(output_dir / "propagated_from_csv_phase.csv", pred_csv_aligned)
    save_array_csv(output_dir / "propagated_from_zernike_phase.csv", pred_zernike_aligned)
    if pred_csv_fit_aligned is not None:
        save_array_csv(output_dir / "propagated_from_csv_phase_fit.csv", pred_csv_fit_aligned)
    if pred_zernike_fit_aligned is not None:
        save_array_csv(output_dir / "propagated_from_zernike_phase_fit.csv", pred_zernike_fit_aligned)

    np.savez_compressed(
        output_dir / "all_results.npz",
        intensity_in=intensity_in,
        mask=mask.astype(np.uint8),
        phase_csv=phase_csv,
        phase_zernike=phase_zernike,
        phase_residual=phase_residual,
        pred_csv=pred_csv_aligned,
        pred_zernike=pred_zernike_aligned,
        measured_output=measured_output if measured_output is not None else np.array([]),
        pred_csv_fit=pred_csv_fit_aligned if pred_csv_fit_aligned is not None else np.array([]),
        pred_zernike_fit=pred_zernike_fit_aligned if pred_zernike_fit_aligned is not None else np.array([]),
    )

    plot_phase_summary(
        path=output_dir / "phase_analysis.png",
        intensity_in=intensity_in,
        phase_csv=phase_csv,
        phase_zernike=phase_zernike,
        phase_residual=phase_residual,
        measured_output=measured_output,
        meta=meta,
        mask=mask,
        dx_um=info_csv["dx_um"],
    )
    plot_propagation_summary(
        path=output_dir / "propagation_comparison.png",
        measured_output=measured_output,
        pred_csv=pred_csv_aligned,
        pred_zernike=pred_zernike_aligned,
        pred_csv_fit=pred_csv_fit_aligned,
        pred_zernike_fit=pred_zernike_fit_aligned,
        csv_metrics=csv_metrics,
        z_metrics=z_metrics,
        csv_fit_metrics=csv_fit_metrics,
        z_fit_metrics=z_fit_metrics,
    )

    metrics = {
        "files": {
            "acc": str(acc),
            "pha": str(pha),
            "int_in": str(int_in),
            "int_out": str(int_out) if int_out is not None else None,
        },
        "meta": {
            "wavelength_nm": meta.wavelength_m * 1e9,
            "pupil_size_mm": meta.pupil_size_m * 1e3,
            "phase_ptv_rad_xml": meta.phase_ptv_rad,
            "phase_rms_rad_xml": meta.phase_rms_rad,
            "max_intensity_xml": meta.max_intensity,
            "tilt_x_mrad": meta.tilt_x_mrad,
            "tilt_y_mrad": meta.tilt_y_mrad,
            "strehl_ratio": meta.strehl_ratio,
        },
        "geometry": info_csv,
        "phase_compare": {
            "csv_ptv_rad": float(phase_csv[mask].max() - phase_csv[mask].min()),
            "csv_rms_rad": float(phase_csv[mask].std()),
            "zernike_vs_csv_corr": float(np.corrcoef(phase_csv[mask], phase_zernike[mask])[0, 1]),
            "zernike_vs_csv_rmse_rad": float(np.sqrt(np.mean((phase_csv[mask] - phase_zernike[mask]) ** 2))),
        },
        "baseline_input_vs_output": baseline_metrics,
        "csv_propagation_vs_output": csv_metrics,
        "zernike_propagation_vs_output": z_metrics,
        "csv_fit": csv_fit_metrics,
        "zernike_fit": z_fit_metrics,
    }

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"\nSaved results to: {output_dir}")


if __name__ == "__main__":
    main()
