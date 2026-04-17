#!/usr/bin/env python3
"""
PCA / SVD analysis for Phasics SID4 beam data.

This script performs two analyses:
1) PCA/SVD on intensity images only
2) PCA/SVD on the complex optical field U = sqrt(I) * exp(i * phi)

Designed for folders containing many CSV files:
- intensity folder: CSV matrices of intensity
- phase folder: CSV matrices of phase

Default example folders (edit if needed):
    INT_DIR = r"C:\\Users\\User\\Desktop\\all INT front"
    PHA_DIR = r"C:\\Users\\User\\Desktop\\all PHA front"

Outputs:
- explained variance plots
- mean image / mean phase
- first principal component images
- scree plots
- summary text file with key numbers
- optional score scatter plots

Notes:
- Complex-field PCA is performed with SVD directly on complex-valued data.
- Intensity PCA uses centered real-valued data.
- Optional center-of-mass alignment and ROI cropping are included.
"""

from __future__ import annotations

import os
import re
import sys
import math
import json
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


# =========================
# Defaults
# =========================
INT_DIR = r"C:\Users\User\Desktop\all INT front"
PHA_DIR = r"C:\Users\User\Desktop\all PHA front"
OUT_DIR = r"C:\Users\User\Desktop\PCA_results"

ALLOWED_EXTENSIONS = {".csv", ".txt"}
DEFAULT_MAX_FILES = None  # set to an int for testing, e.g. 20
DEFAULT_COMPONENTS_TO_SAVE = 6
DEFAULT_ALIGNMENT = True
DEFAULT_NORMALIZE_INTENSITY = True
DEFAULT_CROP = True
DEFAULT_CROP_THRESHOLD = 0.005  # ROI from mean intensity mask
DEFAULT_PHASE_UNITS = "radians"  # "radians" or "waves"
DEFAULT_PAIR_BY_STEM = False  # if True, match files by timestamp/stem similarity
DEFAULT_DPI = 160


# =========================
# Helpers
# =========================
def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def numeric_sort_key(path: Path):
    s = path.stem
    parts = re.split(r"(\d+)", s)
    key = []
    for p in parts:
        if p.isdigit():
            key.append(int(p))
        else:
            key.append(p.lower())
    return key


def list_matrix_files(folder: Path, max_files: Optional[int] = None) -> List[Path]:
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS]
    files = sorted(files, key=numeric_sort_key)
    if max_files is not None:
        files = files[:max_files]
    return files


def read_matrix_csv(path: Path) -> np.ndarray:
    try:
        arr = np.loadtxt(path, delimiter=",")
    except Exception:
        arr = np.genfromtxt(path, delimiter=",")
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D matrix in {path}, got shape {arr.shape}")
    arr = np.nan_to_num(arr.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def parse_phase_units(phase: np.ndarray, phase_units: str) -> np.ndarray:
    phase_units = phase_units.lower().strip()
    if phase_units in {"rad", "radian", "radians"}:
        return phase
    if phase_units in {"wave", "waves", "lambda"}:
        return 2.0 * np.pi * phase
    raise ValueError("phase_units must be 'radians' or 'waves'")


def center_of_mass(img: np.ndarray) -> Tuple[float, float]:
    total = img.sum()
    if total <= 0:
        return (img.shape[0] / 2.0, img.shape[1] / 2.0)
    yy, xx = np.indices(img.shape)
    cy = float((yy * img).sum() / total)
    cx = float((xx * img).sum() / total)
    return cy, cx


def shift_image_fft(img: np.ndarray, shift_y: float, shift_x: float) -> np.ndarray:
    """Subpixel shift using Fourier theorem. Works for real/complex arrays."""
    ny, nx = img.shape
    fy = np.fft.fftfreq(ny)
    fx = np.fft.fftfreq(nx)
    phase_ramp = np.exp(-2j * np.pi * (fy[:, None] * shift_y + fx[None, :] * shift_x))
    shifted = np.fft.ifft2(np.fft.fft2(img) * phase_ramp)
    if np.isrealobj(img):
        shifted = shifted.real
    return shifted


def align_stack_to_first(stack: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Align using center of mass of intensity-like weights."""
    ref = weights[0] if weights is not None else np.abs(stack[0])
    ref_cy, ref_cx = center_of_mass(ref)
    out = np.empty_like(stack)
    for i in range(stack.shape[0]):
        w = weights[i] if weights is not None else np.abs(stack[i])
        cy, cx = center_of_mass(w)
        dy = ref_cy - cy
        dx = ref_cx - cx
        out[i] = shift_image_fft(stack[i], dy, dx)
    return out


def normalize_intensity_stack(stack: np.ndarray) -> np.ndarray:
    sums = stack.reshape(stack.shape[0], -1).sum(axis=1)
    sums[sums == 0] = 1.0
    return stack / sums[:, None, None]


def build_roi_mask(mean_intensity: np.ndarray, threshold: float = DEFAULT_CROP_THRESHOLD) -> np.ndarray:
    maxv = float(np.max(mean_intensity))
    if maxv <= 0:
        return np.ones_like(mean_intensity, dtype=bool)
    return mean_intensity >= threshold * maxv


def bbox_from_mask(mask: np.ndarray, padding: int = 8) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return 0, mask.shape[0], 0, mask.shape[1]
    y0 = max(int(ys.min()) - padding, 0)
    y1 = min(int(ys.max()) + 1 + padding, mask.shape[0])
    x0 = max(int(xs.min()) - padding, 0)
    x1 = min(int(xs.max()) + 1 + padding, mask.shape[1])
    return y0, y1, x0, x1


def crop_stack(stack: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    y0, y1, x0, x1 = bbox
    return stack[:, y0:y1, x0:x1]


def flatten_stack(stack: np.ndarray) -> np.ndarray:
    return stack.reshape(stack.shape[0], -1)


def save_image(img: np.ndarray, path: Path, title: str, cmap: str = "viridis", vmin=None, vmax=None) -> None:
    plt.figure(figsize=(6, 5))
    plt.imshow(img, cmap=cmap, origin="upper", vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=DEFAULT_DPI)
    plt.close()


def save_log_image(img: np.ndarray, path: Path, title: str, cmap: str = "viridis") -> None:
    disp = np.log10(np.maximum(img, np.max(img) * 1e-6 + 1e-12))
    plt.figure(figsize=(6, 5))
    plt.imshow(disp, cmap=cmap, origin="upper")
    plt.colorbar(label="log10(scale)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=DEFAULT_DPI)
    plt.close()


def save_scree_plot(explained: np.ndarray, path: Path, title: str) -> None:
    n = len(explained)
    plt.figure(figsize=(7, 4.5))
    plt.plot(np.arange(1, n + 1), explained * 100, marker="o")
    plt.xlabel("Component")
    plt.ylabel("Explained variance [%]")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=DEFAULT_DPI)
    plt.close()


def save_cumulative_plot(explained: np.ndarray, path: Path, title: str) -> None:
    cum = np.cumsum(explained)
    plt.figure(figsize=(7, 4.5))
    plt.plot(np.arange(1, len(cum) + 1), cum * 100, marker="o")
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance [%]")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=DEFAULT_DPI)
    plt.close()


def save_scores_plot(scores: np.ndarray, path: Path, title: str) -> None:
    if scores.shape[1] < 2:
        return
    plt.figure(figsize=(6, 5))
    plt.scatter(scores[:, 0], scores[:, 1], s=20, alpha=0.8)
    plt.xlabel("PC1 score")
    plt.ylabel("PC2 score")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=DEFAULT_DPI)
    plt.close()


# =========================
# PCA / SVD core
# =========================
@dataclass
class PCAResult:
    mean_vector: np.ndarray
    components: np.ndarray
    scores: np.ndarray
    singular_values: np.ndarray
    explained_variance_ratio: np.ndarray



def pca_real(X: np.ndarray) -> PCAResult:
    """X shape: (n_samples, n_features), real-valued."""
    mean_vec = X.mean(axis=0, keepdims=True)
    Xc = X - mean_vec
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    denom = max(X.shape[0] - 1, 1)
    eigenvalues = (s ** 2) / denom
    total = eigenvalues.sum()
    explained = eigenvalues / total if total > 0 else np.zeros_like(eigenvalues)
    scores = U * s[None, :]
    return PCAResult(mean_vec.ravel(), Vt, scores, s, explained)



def pca_complex(X: np.ndarray) -> PCAResult:
    """Complex PCA via centered complex SVD."""
    mean_vec = X.mean(axis=0, keepdims=True)
    Xc = X - mean_vec
    U, s, Vh = np.linalg.svd(Xc, full_matrices=False)
    denom = max(X.shape[0] - 1, 1)
    eigenvalues = (s ** 2) / denom
    total = eigenvalues.sum()
    explained = eigenvalues / total if total > 0 else np.zeros_like(eigenvalues)
    scores = U * s[None, :]
    return PCAResult(mean_vec.ravel(), Vh, scores, s, explained)


# =========================
# Analysis pipelines
# =========================

def load_intensity_stack(int_dir: Path, max_files: Optional[int]) -> Tuple[np.ndarray, List[Path]]:
    int_files = list_matrix_files(int_dir, max_files)
    if not int_files:
        raise FileNotFoundError(f"No intensity CSV/TXT files found in {int_dir}")
    stack = np.stack([read_matrix_csv(p) for p in int_files], axis=0)
    return stack, int_files



def load_phase_stack(pha_dir: Path, max_files: Optional[int], phase_units: str) -> Tuple[np.ndarray, List[Path]]:
    pha_files = list_matrix_files(pha_dir, max_files)
    if not pha_files:
        raise FileNotFoundError(f"No phase CSV/TXT files found in {pha_dir}")
    stack = np.stack([parse_phase_units(read_matrix_csv(p), phase_units) for p in pha_files], axis=0)
    return stack, pha_files



def maybe_match_by_stem(int_files: List[Path], pha_files: List[Path]) -> Tuple[List[Path], List[Path]]:
    def normalize_stem(stem: str) -> str:
        s = stem.lower()
        s = re.sub(r"^(int|pha)\s+", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    pha_map = {normalize_stem(p.stem): p for p in pha_files}
    int_matched, pha_matched = [], []
    for ip in int_files:
        key = normalize_stem(ip.stem)
        if key in pha_map:
            int_matched.append(ip)
            pha_matched.append(pha_map[key])
    if not int_matched:
        raise RuntimeError("Could not match intensity and phase files by stem.")
    return int_matched, pha_matched



def preprocess_stacks(
    I_stack: np.ndarray,
    P_stack: Optional[np.ndarray],
    align: bool,
    normalize_intensity: bool,
    crop: bool,
    crop_threshold: float,
) -> Tuple[np.ndarray, Optional[np.ndarray], Tuple[int, int, int, int]]:
    if normalize_intensity:
        I_stack = normalize_intensity_stack(I_stack)

    if align:
        I_stack = align_stack_to_first(I_stack, weights=I_stack)
        if P_stack is not None:
            phase_complex = np.exp(1j * P_stack)
            phase_complex = align_stack_to_first(phase_complex, weights=I_stack)
            P_stack = np.angle(phase_complex)

    bbox = (0, I_stack.shape[1], 0, I_stack.shape[2])
    if crop:
        roi_mask = build_roi_mask(I_stack.mean(axis=0), threshold=crop_threshold)
        bbox = bbox_from_mask(roi_mask)
        I_stack = crop_stack(I_stack, bbox)
        if P_stack is not None:
            P_stack = crop_stack(P_stack, bbox)

    return I_stack, P_stack, bbox



def run_intensity_pca(I_stack: np.ndarray, out_dir: Path, n_components_to_save: int) -> PCAResult:
    ensure_dir(out_dir)
    mean_I = I_stack.mean(axis=0)
    std_I = I_stack.std(axis=0)

    save_image(mean_I, out_dir / "intensity_mean.png", "Mean intensity")
    save_log_image(mean_I, out_dir / "intensity_mean_log.png", "Mean intensity (log)")
    save_image(std_I, out_dir / "intensity_std.png", "Intensity standard deviation")

    X = flatten_stack(I_stack)
    res = pca_real(X)

    save_scree_plot(res.explained_variance_ratio, out_dir / "intensity_scree.png", "Intensity PCA scree plot")
    save_cumulative_plot(res.explained_variance_ratio, out_dir / "intensity_cumulative.png", "Intensity PCA cumulative variance")
    save_scores_plot(res.scores, out_dir / "intensity_scores_pc1_pc2.png", "Intensity PCA scores")

    h, w = I_stack.shape[1:]
    nsave = min(n_components_to_save, res.components.shape[0])
    for k in range(nsave):
        comp = res.components[k].reshape(h, w)
        save_image(comp, out_dir / f"intensity_pc{k+1}.png", f"Intensity PC{k+1}", cmap="seismic")

    return res



def run_complex_pca(I_stack: np.ndarray, P_stack: np.ndarray, out_dir: Path, n_components_to_save: int) -> PCAResult:
    ensure_dir(out_dir)
    U_stack = np.sqrt(np.maximum(I_stack, 0.0)) * np.exp(1j * P_stack)

    mean_U = U_stack.mean(axis=0)
    mean_amp = np.abs(mean_U)
    mean_phase = np.angle(mean_U)
    amp_std = np.abs(U_stack - mean_U[None, :, :]).std(axis=0)

    save_image(mean_amp, out_dir / "complex_mean_amplitude.png", "Mean complex-field amplitude")
    save_log_image(mean_amp, out_dir / "complex_mean_amplitude_log.png", "Mean complex-field amplitude (log)")
    save_image(mean_phase, out_dir / "complex_mean_phase.png", "Mean complex-field phase", cmap="twilight")
    save_image(amp_std, out_dir / "complex_amplitude_std.png", "Complex-field amplitude std")

    X = flatten_stack(U_stack)
    res = pca_complex(X)

    save_scree_plot(res.explained_variance_ratio, out_dir / "complex_scree.png", "Complex-field PCA scree plot")
    save_cumulative_plot(res.explained_variance_ratio, out_dir / "complex_cumulative.png", "Complex-field PCA cumulative variance")
    # scores are complex; use real/imag of PC1 score plot
    if res.scores.shape[1] >= 1:
        plt.figure(figsize=(6, 5))
        plt.scatter(res.scores[:, 0].real, res.scores[:, 0].imag, s=20, alpha=0.8)
        plt.xlabel("Re(score PC1)")
        plt.ylabel("Im(score PC1)")
        plt.title("Complex-field PC1 scores")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "complex_scores_pc1_real_imag.png", dpi=DEFAULT_DPI)
        plt.close()

    h, w = I_stack.shape[1:]
    nsave = min(n_components_to_save, res.components.shape[0])
    for k in range(nsave):
        comp = res.components[k].reshape(h, w)
        save_image(np.abs(comp), out_dir / f"complex_pc{k+1}_amplitude.png", f"Complex PC{k+1} amplitude")
        save_image(np.angle(comp), out_dir / f"complex_pc{k+1}_phase.png", f"Complex PC{k+1} phase", cmap="twilight")

    return res



def write_summary(
    path: Path,
    intensity_result: PCAResult,
    complex_result: PCAResult,
    n_files_int: int,
    n_files_pha: int,
    bbox: Tuple[int, int, int, int],
) -> None:
    def lines_for(name: str, res: PCAResult) -> List[str]:
        exp = res.explained_variance_ratio
        cum = np.cumsum(exp)
        return [
            f"[{name}]",
            f"PC1 explained variance: {exp[0]*100:.3f}%" if len(exp) > 0 else "PC1 explained variance: n/a",
            f"PC2 explained variance: {exp[1]*100:.3f}%" if len(exp) > 1 else "PC2 explained variance: n/a",
            f"PC3 explained variance: {exp[2]*100:.3f}%" if len(exp) > 2 else "PC3 explained variance: n/a",
            f"First 3 cumulative: {cum[min(2, len(cum)-1)]*100:.3f}%" if len(cum) > 0 else "First 3 cumulative: n/a",
            f"Components for 90% variance: {int(np.searchsorted(cum, 0.90) + 1) if len(cum) > 0 else 'n/a'}",
            f"Components for 95% variance: {int(np.searchsorted(cum, 0.95) + 1) if len(cum) > 0 else 'n/a'}",
            f"Components for 99% variance: {int(np.searchsorted(cum, 0.99) + 1) if len(cum) > 0 else 'n/a'}",
            "",
        ]

    text = []
    text.append("PCA/SVD beam analysis summary")
    text.append("=" * 40)
    text.append(f"Number of intensity files: {n_files_int}")
    text.append(f"Number of phase files: {n_files_pha}")
    text.append(f"ROI bbox (y0, y1, x0, x1): {bbox}")
    text.append("")
    text.extend(lines_for("Intensity PCA", intensity_result))
    text.extend(lines_for("Complex-field PCA", complex_result))
    path.write_text("\n".join(text), encoding="utf-8")



def main() -> None:
    parser = argparse.ArgumentParser(description="PCA/SVD analysis for beam intensity and complex field")
    parser.add_argument("--int-dir", type=str, default=INT_DIR, help="Folder with intensity CSV/TXT files")
    parser.add_argument("--pha-dir", type=str, default=PHA_DIR, help="Folder with phase CSV/TXT files")
    parser.add_argument("--out-dir", type=str, default=OUT_DIR, help="Output folder")
    parser.add_argument("--phase-units", type=str, default=DEFAULT_PHASE_UNITS, choices=["radians", "waves"], help="Units of phase CSV values")
    parser.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES, help="Limit number of files (for testing)")
    parser.add_argument("--components", type=int, default=DEFAULT_COMPONENTS_TO_SAVE, help="Number of PCs to save as images")
    parser.add_argument("--no-align", action="store_true", help="Disable center-of-mass alignment")
    parser.add_argument("--no-normalize-intensity", action="store_true", help="Disable framewise intensity normalization")
    parser.add_argument("--no-crop", action="store_true", help="Disable ROI crop")
    parser.add_argument("--crop-threshold", type=float, default=DEFAULT_CROP_THRESHOLD, help="Threshold for ROI crop based on mean intensity max fraction")
    parser.add_argument("--pair-by-stem", action="store_true", default=DEFAULT_PAIR_BY_STEM, help="Match intensity/phase files by similar stem instead of simple sort order")
    args = parser.parse_args()

    int_dir = Path(args.int_dir)
    pha_dir = Path(args.pha_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    log("Loading intensity files...")
    int_files = list_matrix_files(int_dir, args.max_files)
    pha_files = list_matrix_files(pha_dir, args.max_files)
    if args.pair_by_stem:
        int_files, pha_files = maybe_match_by_stem(int_files, pha_files)
    else:
        n = min(len(int_files), len(pha_files))
        int_files = int_files[:n]
        pha_files = pha_files[:n]

    if len(int_files) == 0 or len(pha_files) == 0:
        raise RuntimeError("No matching intensity/phase files found.")

    log(f"Using {len(int_files)} intensity files and {len(pha_files)} phase files.")

    I_stack = np.stack([read_matrix_csv(p) for p in int_files], axis=0)
    P_stack = np.stack([parse_phase_units(read_matrix_csv(p), args.phase_units) for p in pha_files], axis=0)

    if I_stack.shape != P_stack.shape:
        raise RuntimeError(f"Intensity and phase stacks have different shapes: {I_stack.shape} vs {P_stack.shape}")

    log(f"Original stack shape: {I_stack.shape}")
    I_stack, P_stack, bbox = preprocess_stacks(
        I_stack,
        P_stack,
        align=not args.no_align,
        normalize_intensity=not args.no_normalize_intensity,
        crop=not args.no_crop,
        crop_threshold=args.crop_threshold,
    )
    log(f"Processed stack shape: {I_stack.shape}")
    log(f"ROI bbox: {bbox}")

    intensity_out = out_dir / "intensity_pca"
    complex_out = out_dir / "complex_pca"

    log("Running intensity PCA...")
    intensity_res = run_intensity_pca(I_stack, intensity_out, args.components)

    log("Running complex-field PCA...")
    complex_res = run_complex_pca(I_stack, P_stack, complex_out, args.components)

    write_summary(out_dir / "summary.txt", intensity_res, complex_res, len(int_files), len(pha_files), bbox)

    config = {
        "int_dir": str(int_dir),
        "pha_dir": str(pha_dir),
        "out_dir": str(out_dir),
        "phase_units": args.phase_units,
        "max_files": args.max_files,
        "components_saved": args.components,
        "alignment": not args.no_align,
        "normalize_intensity": not args.no_normalize_intensity,
        "crop": not args.no_crop,
        "crop_threshold": args.crop_threshold,
        "pair_by_stem": args.pair_by_stem,
        "input_shape": list(map(int, I_stack.shape)),
        "roi_bbox": list(map(int, bbox)),
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    log("Done.")
    log(f"Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
