#!/usr/bin/env python3
"""
Vergleicht 5 Propagationsfälle für ein SID4-Datenset bei freier Propagation
(Angular Spectrum) und berechnet robuste Vergleichsgrößen gegen INT output.

Fälle:
1) Defocus
2) Defocus + Astigmatismus (0°, 45°)
3) Defocus + Astigmatismus + sphärische Aberration
4) volle Zernike-Rekonstruktion aus XML
5) volle CSV-Phase

Ausgabe:
- PNG mit Propagationsbildern
- CSV/JSON mit Metriken
- NPY/NPZ mit Intensitäten

Anpassen:
- INPUT_DIR
- PROP_DISTANCE_M
- CSV_DELIMITER ggf. ',' oder ';'
"""
from __future__ import annotations

import csv
import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt

# =========================
# Benutzer-Einstellungen
# =========================
INPUT_DIR = Path.home() / "Desktop" / "Theory free propagation"
ACC_FILE = INPUT_DIR / "ACC SID4 22h33m46s105ms.xml"
PHA_FILE = INPUT_DIR / "PHA SID4 22h33m46s105ms.csv"
INT_FILE = INPUT_DIR / "INT SID4 22h33m46s105ms.csv"
OUTPUT_FILE = INPUT_DIR / "INT output.csv"

CSV_DELIMITER = ","  # falls nötig auf ';' ändern
PROP_DISTANCE_M = 10.0
PAD_FACTOR = 4
REMOVE_EVANESCENT = True
APOD_EDGE_PX = 6.0
PHASE_SIGN = -1.0  # Für SID4 bei deinen Daten war -1 deutlich plausibler
PHASE_SCALE = 1.0
OUTPUT_DIR = INPUT_DIR / "ablation_results"

# Genau 5 Modellfälle wie gewünscht
CASE_DEFINITIONS = [
    ("defocus", [4]),
    ("defocus_astig", [4, 5, 6]),
    ("defocus_astig_spherical", [4, 5, 6, 11]),
    ("full_zernike", list(range(1, 29))),
    ("csv_phase", None),
]


# =========================
# Hilfsdatenstrukturen
# =========================
@dataclass
class Sid4Meta:
    wavelength_m: float
    pupil_diameter_m: float
    zernike_coeffs: Dict[int, float]
    tilt_x_mrad: float | None = None
    tilt_y_mrad: float | None = None
    strehl: float | None = None
    phase_rms: float | None = None
    phase_ptv: float | None = None


# =========================
# Laden / Parsen
# =========================
def smart_loadtxt(path: Path, preferred_delimiter: str = ",") -> np.ndarray:
    errors = []
    for delim in [preferred_delimiter, ";", ",", "\t", None]:
        try:
            if delim is None:
                arr = np.loadtxt(path)
            else:
                arr = np.loadtxt(path, delimiter=delim)
            if arr.ndim == 2 and arr.size > 0:
                return arr.astype(np.float64)
            errors.append(f"delimiter={delim!r} -> shape={arr.shape}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"delimiter={delim!r} -> {exc}")
    raise RuntimeError(f"Konnte {path} nicht laden. Versuche: {errors}")


def parse_float(text: str | None) -> float | None:
    if text is None:
        return None
    text = text.strip().replace(",", ".")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_acc_xml(path: Path) -> Sid4Meta:
    tree = ET.parse(path)
    root = tree.getroot()

    wavelength_nm = None
    pupil_mm = None
    tilt_x = None
    tilt_y = None
    strehl = None
    phase_rms = None
    phase_ptv = None
    zernikes: Dict[int, float] = {}

    # WaveLength direkt suchen
    for elem in root.iter():
        tag = elem.tag.split("}")[-1]
        txt = (elem.text or "").strip()
        if tag == "WaveLength":
            wavelength_nm = parse_float(txt)

    data_type = None
    label = None
    index = None
    for data in root.iter():
        tag = data.tag.split("}")[-1]
        if tag == "DataType":
            data_type = (data.text or "").strip()
        elif tag == "Label":
            label = (data.text or "").strip()
        elif tag == "Index":
            index = (data.text or "").strip()
        elif tag == "Value":
            value = parse_float(data.text)
            if data_type == "Projection" and index and index.isdigit() and value is not None:
                zernikes[int(index)] = value
                if label == "PupilSize (mm)":
                    pupil_mm = value
            elif data_type == "Projection" and label == "PupilSize (mm)" and value is not None:
                pupil_mm = value
            elif data_type == "Tilt" and label == "Tilt X (mrad)":
                tilt_x = value
            elif data_type == "Tilt" and label == "Tilt Y (mrad)":
                tilt_y = value
            elif data_type == "FarField" and label == "Strehl ratio":
                strehl = value
            elif data_type == "Phase" and label == "RMS Phase":
                phase_rms = value
            elif data_type == "Phase" and label == "PtV Phase":
                phase_ptv = value
            elif data_type == "Projection" and label == "PupilSize (mm)":
                pupil_mm = value

    # Fallback: PupilSize (mm) als beliebiges Label suchen
    if pupil_mm is None:
        projection_mode = False
        current_label = None
        for elem in root.iter():
            tag = elem.tag.split("}")[-1]
            txt = (elem.text or "").strip()
            if tag == "DataType" and txt == "Projection":
                projection_mode = True
            elif projection_mode and tag == "Label":
                current_label = txt
            elif projection_mode and tag == "Value" and current_label == "PupilSize (mm)":
                pupil_mm = parse_float(txt)
                break

    if wavelength_nm is None:
        raise ValueError("WaveLength nicht in XML gefunden.")
    if pupil_mm is None:
        raise ValueError("PupilSize (mm) nicht in XML gefunden.")

    return Sid4Meta(
        wavelength_m=wavelength_nm * 1e-9,
        pupil_diameter_m=pupil_mm * 1e-3,
        zernike_coeffs=zernikes,
        tilt_x_mrad=tilt_x,
        tilt_y_mrad=tilt_y,
        strehl=strehl,
        phase_rms=phase_rms,
        phase_ptv=phase_ptv,
    )


# =========================
# Optische Hilfsfunktionen
# =========================
def soft_mask_from_binary(mask: np.ndarray, edge_px: float = 6.0) -> np.ndarray:
    if edge_px <= 0:
        return mask.astype(np.float64)
    dist_in = distance_transform_edt(mask)
    soft = np.zeros_like(mask, dtype=np.float64)
    inside = dist_in > 0
    soft[inside] = np.minimum(dist_in[inside] / edge_px, 1.0)
    return soft


def effective_diameter_from_mask(mask: np.ndarray) -> float:
    area = float(np.count_nonzero(mask))
    if area <= 0:
        raise ValueError("Leere Maske")
    return math.sqrt(4.0 * area / math.pi)


def zernike_noll(j: int, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
    # Noll-Index -> (n, m)
    noll_map = {
        1: (0, 0),
        2: (1, -1), 3: (1, 1),
        4: (2, 0), 5: (2, -2), 6: (2, 2),
        7: (3, -1), 8: (3, 1), 9: (3, -3), 10: (3, 3),
        11: (4, 0), 12: (4, -2), 13: (4, 2), 14: (4, -4), 15: (4, 4),
        16: (5, -1), 17: (5, 1), 18: (5, -3), 19: (5, 3), 20: (5, -5), 21: (5, 5),
        22: (6, 0), 23: (6, -2), 24: (6, 2), 25: (6, -4), 26: (6, 4), 27: (6, -6), 28: (6, 6),
    }
    if j not in noll_map:
        raise ValueError(f"Noll-Index {j} in diesem Skript nicht implementiert.")
    n, m = noll_map[j]
    m_abs = abs(m)

    R = np.zeros_like(rho, dtype=np.float64)
    for s in range((n - m_abs) // 2 + 1):
        c = ((-1) ** s * math.factorial(n - s) /
             (math.factorial(s) * math.factorial((n + m_abs) // 2 - s) * math.factorial((n - m_abs) // 2 - s)))
        R += c * rho ** (n - 2 * s)

    Z = np.zeros_like(rho, dtype=np.float64)
    inside = rho <= 1.0
    if m == 0:
        Z[inside] = R[inside]
    elif m < 0:
        Z[inside] = R[inside] * np.sin(m_abs * theta[inside])
    else:
        Z[inside] = R[inside] * np.cos(m_abs * theta[inside])
    return Z


def build_coordinate_system(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.indices(mask.shape)
    yx = np.argwhere(mask)
    cy, cx = yx.mean(axis=0)
    eff_d = effective_diameter_from_mask(mask)
    radius = eff_d / 2.0
    x_norm = (xx - cx) / radius
    y_norm = (yy - cy) / radius
    rho = np.sqrt(x_norm ** 2 + y_norm ** 2)
    theta = np.arctan2(y_norm, x_norm)
    return x_norm, y_norm, rho, theta


def build_zernike_phase(mask: np.ndarray, coeffs: Dict[int, float], indices: List[int]) -> np.ndarray:
    _, _, rho, theta = build_coordinate_system(mask)
    phase = np.zeros(mask.shape, dtype=np.float64)
    for j in indices:
        if j in coeffs:
            phase += coeffs[j] * zernike_noll(j, rho, theta)
    phase[~mask] = 0.0
    return phase


def angular_spectrum(U0: np.ndarray, dx: float, dy: float, wavelength: float, z: float,
                     pad_factor: int = 4, remove_evanescent: bool = True) -> np.ndarray:
    ny, nx = U0.shape
    py = int((pad_factor - 1) * ny / 2)
    px = int((pad_factor - 1) * nx / 2)
    U = np.pad(U0, ((py, py), (px, px)), mode="constant")

    Ny, Nx = U.shape
    fx = np.fft.fftfreq(Nx, d=dx)
    fy = np.fft.fftfreq(Ny, d=dy)
    FX, FY = np.meshgrid(fx, fy)

    k = 2.0 * np.pi / wavelength
    kz_sq = k ** 2 - (2.0 * np.pi * FX) ** 2 - (2.0 * np.pi * FY) ** 2
    kz = np.sqrt(np.maximum(kz_sq, 0.0))
    H = np.exp(1j * kz * z)
    if remove_evanescent:
        H[kz_sq < 0] = 0.0

    Uz = np.fft.ifft2(np.fft.fft2(U) * H)
    return Uz[py:py + ny, px:px + nx]


# =========================
# Metriken
# =========================
def normalize_image(img: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    arr = np.array(img, dtype=np.float64, copy=True)
    if mask is not None:
        arr[~mask] = 0.0
        maxv = arr[mask].max() if np.any(mask) else arr.max()
    else:
        maxv = arr.max()
    if maxv > 0:
        arr /= maxv
    return arr


def compute_beam_metrics(I: np.ndarray, dx: float, dy: float, mask: np.ndarray | None = None) -> Dict[str, float]:
    I = np.clip(np.asarray(I, dtype=np.float64), 0.0, None)
    if mask is not None:
        I = I.copy()
        I[~mask] = 0.0

    yy, xx = np.indices(I.shape)
    x = (xx - (I.shape[1] - 1) / 2.0) * dx
    y = (yy - (I.shape[0] - 1) / 2.0) * dy

    P = I.sum()
    if P <= 0:
        return {k: float("nan") for k in [
            "power", "centroid_x_mm", "centroid_y_mm", "sigma_x_mm", "sigma_y_mm",
            "d4sigma_x_mm", "d4sigma_y_mm", "mean_radius_mm", "r50_mm", "r86_mm", "peak_radius_mm"
        ]}

    xc = (I * x).sum() / P
    yc = (I * y).sum() / P
    x2 = (I * (x - xc) ** 2).sum() / P
    y2 = (I * (y - yc) ** 2).sum() / P
    sigma_x = math.sqrt(max(x2, 0.0))
    sigma_y = math.sqrt(max(y2, 0.0))

    r = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
    mean_r = (I * r).sum() / P

    # Encircled energy radii
    rr = r.ravel()
    ww = I.ravel()
    order = np.argsort(rr)
    rr_s = rr[order]
    ww_s = ww[order]
    cum = np.cumsum(ww_s)
    cum /= cum[-1]
    r50 = rr_s[np.searchsorted(cum, 0.50)]
    r86 = rr_s[np.searchsorted(cum, 0.865)]  # 86.5 % ~ Gaussian 1/e^2 Bezug grob nützlich

    peak_idx = np.unravel_index(np.argmax(I), I.shape)
    peak_r = math.sqrt((x[peak_idx] - xc) ** 2 + (y[peak_idx] - yc) ** 2)

    return {
        "power": float(P),
        "centroid_x_mm": float(xc * 1e3),
        "centroid_y_mm": float(yc * 1e3),
        "sigma_x_mm": float(sigma_x * 1e3),
        "sigma_y_mm": float(sigma_y * 1e3),
        "d4sigma_x_mm": float(4.0 * sigma_x * 1e3),
        "d4sigma_y_mm": float(4.0 * sigma_y * 1e3),
        "mean_radius_mm": float(mean_r * 1e3),
        "r50_mm": float(r50 * 1e3),
        "r86_mm": float(r86 * 1e3),
        "peak_radius_mm": float(peak_r * 1e3),
    }


def image_similarity(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> Dict[str, float]:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if mask is not None:
        aa = aa[mask]
        bb = bb[mask]
    aa = aa.ravel()
    bb = bb.ravel()
    if aa.size == 0:
        return {"corr": float("nan"), "rmse": float("nan")}
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    denom = np.sqrt(np.sum(aa ** 2) * np.sum(bb ** 2))
    corr = float(np.sum(aa * bb) / denom) if denom > 0 else float("nan")
    rmse = float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
    return {"corr": corr, "rmse": rmse}


# =========================
# Hauptlogik
# =========================
def prepare_input_field(meta: Sid4Meta, intensity_csv: np.ndarray, phase_csv: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    if intensity_csv.shape != phase_csv.shape:
        raise ValueError(f"Shape mismatch: intensity={intensity_csv.shape}, phase={phase_csv.shape}")

    mask = (intensity_csv > 0) & np.isfinite(phase_csv)
    if not np.any(mask):
        raise ValueError("Keine gültigen Pixel in der Überlappung von Intensität und Phase")

    eff_d_px = effective_diameter_from_mask(mask)
    dx = meta.pupil_diameter_m / eff_d_px
    dy = dx

    I0 = np.zeros_like(intensity_csv, dtype=np.float64)
    I0[mask] = intensity_csv[mask]
    I0 = normalize_image(I0, mask)
    amplitude = np.sqrt(I0) * soft_mask_from_binary(mask, APOD_EDGE_PX)

    phase = np.zeros_like(phase_csv, dtype=np.float64)
    phase[mask] = phase_csv[mask]
    phase[mask] -= np.mean(phase[mask])  # Piston entfernen, physikalisch egal

    return amplitude, phase, mask, dx, dy


def build_case_phases(csv_phase: np.ndarray, zernike_coeffs: Dict[int, float], mask: np.ndarray) -> Dict[str, np.ndarray]:
    phases: Dict[str, np.ndarray] = {}
    for name, indices in CASE_DEFINITIONS:
        if indices is None:
            ph = csv_phase.copy()
        else:
            ph = build_zernike_phase(mask, zernike_coeffs, indices)
            if np.any(mask):
                ph[mask] -= np.mean(ph[mask])
        ph[~mask] = 0.0
        phases[name] = ph
    return phases


def propagate_cases(amplitude: np.ndarray, phases: Dict[str, np.ndarray], dx: float, dy: float,
                    wavelength: float, z_m: float) -> Dict[str, np.ndarray]:
    results: Dict[str, np.ndarray] = {}
    for name, phase in phases.items():
        U0 = amplitude * np.exp(1j * PHASE_SIGN * PHASE_SCALE * phase)
        Uz = angular_spectrum(U0, dx, dy, wavelength, z_m, PAD_FACTOR, REMOVE_EVANESCENT)
        Iz = np.abs(Uz) ** 2
        results[name] = normalize_image(Iz)
    return results


def make_overview_plot(real_out: np.ndarray, propagated: Dict[str, np.ndarray], mask: np.ndarray, save_path: Path) -> None:
    n = len(propagated) + 1
    cols = 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.4 * cols, 4.8 * rows), constrained_layout=True)
    axes = np.array(axes).reshape(rows, cols)

    ax0 = axes.flat[0]
    ax0.imshow(real_out, cmap="inferno")
    ax0.set_title("Real: INT output")
    ax0.axis("off")

    for ax, (name, img) in zip(axes.flat[1:], propagated.items()):
        diff = image_similarity(img, real_out, mask)
        ax.imshow(img, cmap="inferno")
        ax.set_title(f"{name}\ncorr={diff['corr']:.3f}, rmse={diff['rmse']:.3f}")
        ax.axis("off")

    for ax in axes.flat[n:]:
        ax.axis("off")

    fig.suptitle("Vergleich der 5 Propagationsfälle bei 10 m", fontsize=16)
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def save_metrics_table(metrics_rows: List[Dict[str, float | str]], csv_path: Path, json_path: Path) -> None:
    fieldnames = list(metrics_rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_rows)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_rows, f, indent=2, ensure_ascii=False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    meta = parse_acc_xml(ACC_FILE)
    intensity_csv = smart_loadtxt(INT_FILE, CSV_DELIMITER)
    phase_csv = smart_loadtxt(PHA_FILE, CSV_DELIMITER)
    output_csv = smart_loadtxt(OUTPUT_FILE, CSV_DELIMITER)

    amplitude, csv_phase, mask, dx, dy = prepare_input_field(meta, intensity_csv, phase_csv)

    if output_csv.shape != intensity_csv.shape:
        raise ValueError(
            f"INT output shape {output_csv.shape} passt nicht zu Input {intensity_csv.shape}. "
            "Für dieses Skript wird derselbe Pixelraster vorausgesetzt."
        )

    # Real-Output nur normieren, nicht auf andere Größe zoomen
    real_out = normalize_image(output_csv)

    phases = build_case_phases(csv_phase, meta.zernike_coeffs, mask)
    propagated = propagate_cases(amplitude, phases, dx, dy, meta.wavelength_m, PROP_DISTANCE_M)

    metrics_rows: List[Dict[str, float | str]] = []
    real_metrics = compute_beam_metrics(real_out, dx, dy, mask)
    metrics_rows.append({"case": "real_output", **real_metrics, "corr_to_real": 1.0, "rmse_to_real": 0.0})

    for name, img in propagated.items():
        beam = compute_beam_metrics(img, dx, dy, mask)
        sim = image_similarity(img, real_out, mask)
        metrics_rows.append({"case": name, **beam, "corr_to_real": sim["corr"], "rmse_to_real": sim["rmse"]})

    save_metrics_table(metrics_rows, OUTPUT_DIR / "beam_metrics.csv", OUTPUT_DIR / "beam_metrics.json")
    make_overview_plot(real_out, propagated, mask, OUTPUT_DIR / "overview_5cases.png")

    np.savez(
        OUTPUT_DIR / "all_images.npz",
        real_output=real_out,
        mask=mask.astype(np.uint8),
        amplitude=amplitude,
        csv_phase=csv_phase,
        dx_m=dx,
        dy_m=dy,
        wavelength_m=meta.wavelength_m,
        **{f"phase_{k}": v for k, v in phases.items()},
        **{f"prop_{k}": v for k, v in propagated.items()},
    )

    summary = {
        "input_dir": str(INPUT_DIR),
        "acc_file": str(ACC_FILE),
        "pha_file": str(PHA_FILE),
        "int_file": str(INT_FILE),
        "output_file": str(OUTPUT_FILE),
        "shape": list(intensity_csv.shape),
        "wavelength_nm": meta.wavelength_m * 1e9,
        "effective_pixel_size_um": dx * 1e6,
        "propagation_distance_m": PROP_DISTANCE_M,
        "phase_sign": PHASE_SIGN,
        "phase_scale": PHASE_SCALE,
        "meta": {
            "strehl": meta.strehl,
            "phase_rms": meta.phase_rms,
            "phase_ptv": meta.phase_ptv,
            "tilt_x_mrad": meta.tilt_x_mrad,
            "tilt_y_mrad": meta.tilt_y_mrad,
            "dominant_zernikes": {
                "Z4_defocus": meta.zernike_coeffs.get(4),
                "Z5_astig_0": meta.zernike_coeffs.get(5),
                "Z6_astig_45": meta.zernike_coeffs.get(6),
                "Z11_spherical": meta.zernike_coeffs.get(11),
            },
        },
    }
    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Fertig.")
    print(f"Ergebnisse in: {OUTPUT_DIR}")
    print(f"effektive Pixelgröße: {dx * 1e6:.3f} µm")
    print("Top-Vergleichsgrößen siehe: beam_metrics.csv")


if __name__ == "__main__":
    main()
