from pathlib import Path
import argparse
import json
import csv
import numpy as np
import matplotlib.pyplot as plt

# =========================
# User-editable defaults
# =========================
DEFAULT_INPUT_DIR = Path(r"C:\Users\User\Desktop\Theory lens propagation")
DEFAULT_PHASE_FILE = "average_PHA.csv"
DEFAULT_INTENSITY_FILE = "average_INT.csv"

DEFAULT_INPUT_D4SIGMA_MM = 11.0
DEFAULT_WAVELENGTH_NM = 800.0
DEFAULT_PHASE_SIGN = -1.0

# Thin lens settings
DEFAULT_USE_LENS = True
DEFAULT_LENS_FOCAL_LENGTH_M = 5.0
DEFAULT_LENS_POSITION_M = 0.0   # lens position relative to the input plane

DEFAULT_Z_MIN_M = 0.0
DEFAULT_Z_MAX_M = 15.0
DEFAULT_Z_STEP_M = 0.25

DEFAULT_OUTPUT_DIR_NAME = "beam_size_vs_z_with_lens"

# =========================
# Helpers
# =========================
def load_csv_auto(path: Path) -> np.ndarray:
    for delim in [",", ";", "\t"]:
        try:
            arr = np.loadtxt(path, delimiter=delim)
            if arr.ndim == 2 and arr.size > 0:
                return arr
        except Exception:
            pass
    raise ValueError(f"Could not parse file: {path}")

def angular_spectrum(U0: np.ndarray, dx: float, dy: float, wavelength: float, z: float, pad_factor: int = 4) -> np.ndarray:
    ny, nx = U0.shape
    py = int((pad_factor - 1) * ny / 2)
    px = int((pad_factor - 1) * nx / 2)
    U = np.pad(U0, ((py, py), (px, px)), mode="constant")

    Ny, Nx = U.shape
    fx = np.fft.fftfreq(Nx, d=dx)
    fy = np.fft.fftfreq(Ny, d=dy)
    FX, FY = np.meshgrid(fx, fy)

    k = 2 * np.pi / wavelength
    kz_sq = k**2 - (2 * np.pi * FX) ** 2 - (2 * np.pi * FY) ** 2
    kz = np.sqrt(np.maximum(kz_sq, 0.0))
    H = np.exp(1j * kz * z)
    H[kz_sq < 0] = 0.0

    Uz = np.fft.ifft2(np.fft.fft2(U) * H)
    return Uz[py:py + ny, px:px + nx]

def beam_metrics_units(I: np.ndarray, dx: float, dy: float) -> dict:
    I = np.clip(np.asarray(I, dtype=float), 0, None)
    P = I.sum()
    if P <= 0:
        raise ValueError("Beam intensity contains no positive power.")

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

    def qrad(frac: float) -> float:
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

def apply_thin_lens(U: np.ndarray, dx: float, dy: float, wavelength: float, focal_length_m: float) -> np.ndarray:
    ny, nx = U.shape
    y = (np.arange(ny) - (ny - 1) / 2.0) * dy
    x = (np.arange(nx) - (nx - 1) / 2.0) * dx
    X, Y = np.meshgrid(x, y)
    k = 2 * np.pi / wavelength
    lens_phase = np.exp(-1j * k * (X**2 + Y**2) / (2.0 * focal_length_m))
    return U * lens_phase

def make_z_list(z_min: float, z_max: float, z_step: float):
    n = int(round((z_max - z_min) / z_step))
    arr = [z_min + i * z_step for i in range(n + 1)]
    if abs(arr[-1] - z_max) > 1e-12:
        arr.append(z_max)
    return arr

def propagate_with_optional_lens(U0, dx, dy, wavelength, z, use_lens, lens_position, focal_length):
    if not use_lens:
        return angular_spectrum(U0, dx, dy, wavelength, z) if z != 0 else U0

    if z <= lens_position:
        return angular_spectrum(U0, dx, dy, wavelength, z) if z != 0 else U0

    # propagate to lens plane, apply lens, then propagate further
    U_at_lens = angular_spectrum(U0, dx, dy, wavelength, lens_position) if lens_position != 0 else U0
    U_after_lens = apply_thin_lens(U_at_lens, dx, dy, wavelength, focal_length)
    dz = z - lens_position
    return angular_spectrum(U_after_lens, dx, dy, wavelength, dz) if dz != 0 else U_after_lens

def main():
    parser = argparse.ArgumentParser(description="Beam size versus z using measured intensity + measured phase, optionally including a thin lens.")
    parser.add_argument("--input-dir", type=str, default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--phase-file", type=str, default=DEFAULT_PHASE_FILE)
    parser.add_argument("--intensity-file", type=str, default=DEFAULT_INTENSITY_FILE)
    parser.add_argument("--input-d4sigma-mm", type=float, default=DEFAULT_INPUT_D4SIGMA_MM)
    parser.add_argument("--wavelength-nm", type=float, default=DEFAULT_WAVELENGTH_NM)
    parser.add_argument("--phase-sign", type=float, default=DEFAULT_PHASE_SIGN)
    parser.add_argument("--use-lens", action="store_true", default=DEFAULT_USE_LENS)
    parser.add_argument("--no-lens", action="store_true", help="Disable the thin lens.")
    parser.add_argument("--focal-length-m", type=float, default=DEFAULT_LENS_FOCAL_LENGTH_M)
    parser.add_argument("--lens-position-m", type=float, default=DEFAULT_LENS_POSITION_M)
    parser.add_argument("--z-min-m", type=float, default=DEFAULT_Z_MIN_M)
    parser.add_argument("--z-max-m", type=float, default=DEFAULT_Z_MAX_M)
    parser.add_argument("--z-step-m", type=float, default=DEFAULT_Z_STEP_M)
    parser.add_argument("--output-dir-name", type=str, default=DEFAULT_OUTPUT_DIR_NAME)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    phase_path = input_dir / args.phase_file
    intensity_path = input_dir / args.intensity_file
    output_dir = input_dir / args.output_dir_name
    output_dir.mkdir(exist_ok=True)

    use_lens = args.use_lens and not args.no_lens
    wavelength_m = args.wavelength_nm * 1e-9

    phase = load_csv_auto(phase_path)
    intensity = load_csv_auto(intensity_path)

    if phase.shape != intensity.shape:
        raise ValueError(f"Shape mismatch: phase {phase.shape} vs intensity {intensity.shape}")

    mask = (intensity > 0) & np.isfinite(phase)
    int_norm = np.zeros_like(intensity, dtype=float)
    int_norm[mask] = intensity[mask] / intensity[mask].max()
    amplitude = np.sqrt(int_norm)

    # Calibrate pixel size so that input beam has the known D4σ diameter
    tmp_px = beam_metrics_units(int_norm, 1.0, 1.0)
    input_d4_px = tmp_px["d4sigma_mean"]
    dx_m = (args.input_d4sigma_mm / input_d4_px) * 1e-3
    dy_m = dx_m

    U0 = amplitude * np.exp(1j * args.phase_sign * phase)

    z_list = make_z_list(args.z_min_m, args.z_max_m, args.z_step_m)
    rows = []
    selected_images = {}
    selected_planes = [args.z_min_m, 5.0, 10.0, args.z_max_m]
    for z in z_list:
        Uz = propagate_with_optional_lens(
            U0=U0,
            dx=dx_m,
            dy=dy_m,
            wavelength=wavelength_m,
            z=z,
            use_lens=use_lens,
            lens_position=args.lens_position_m,
            focal_length=args.focal_length_m,
        )
        Iz = np.abs(Uz) ** 2
        m = beam_metrics_units(Iz, dx_m, dy_m)
        rows.append({
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
        for sp in selected_planes:
            if abs(z - sp) < 1e-12:
                norm = Iz - Iz.min()
                if norm.max() > 0:
                    norm = norm / norm.max()
                selected_images[z] = norm

    # Save table
    csv_path = output_dir / "beam_size_vs_z_with_lens.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_dir / "beam_size_vs_z_with_lens.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "phase_file": str(phase_path),
            "intensity_file": str(intensity_path),
            "input_d4sigma_mm": args.input_d4sigma_mm,
            "pixel_size_um": dx_m * 1e6,
            "wavelength_nm": args.wavelength_nm,
            "phase_sign": args.phase_sign,
            "use_lens": use_lens,
            "focal_length_m": args.focal_length_m if use_lens else None,
            "lens_position_m": args.lens_position_m if use_lens else None,
            "rows": rows,
        }, f, indent=2)

    # Plot beam size vs z
    z = [r["z_m"] for r in rows]
    plt.figure(figsize=(8, 5))
    plt.plot(z, [r["d4sigma_mean_mm"] for r in rows], marker="o", markersize=3, label="Mean D4σ diameter")
    plt.plot(z, [r["d4sigma_x_mm"] for r in rows], label="D4σ x")
    plt.plot(z, [r["d4sigma_y_mm"] for r in rows], label="D4σ y")
    title = "Beam size versus propagation distance"
    if use_lens:
        title += f" (thin lens f = {args.focal_length_m:g} m at z = {args.lens_position_m:g} m)"
    plt.xlabel("Propagation distance z [m]")
    plt.ylabel("Beam size [mm]")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "beam_size_vs_z_with_lens.png", dpi=180)
    plt.close()

    # Plot radii vs z
    plt.figure(figsize=(8, 5))
    plt.plot(z, [r["mean_radius_mm"] for r in rows], marker="o", markersize=3, label="Mean radius")
    plt.plot(z, [r["r50_mm"] for r in rows], label="r50")
    plt.plot(z, [r["r86_mm"] for r in rows], label="r86")
    title = "Beam radius versus propagation distance"
    if use_lens:
        title += f" (thin lens f = {args.focal_length_m:g} m)"
    plt.xlabel("Propagation distance z [m]")
    plt.ylabel("Radius [mm]")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "beam_radius_vs_z_with_lens.png", dpi=180)
    plt.close()

    # Plot selected intensity planes
    available_planes = [sp for sp in selected_planes if sp in selected_images]
    fig, axes = plt.subplots(1, len(available_planes), figsize=(3.5 * len(available_planes), 3.4), constrained_layout=True)
    if len(available_planes) == 1:
        axes = [axes]
    for ax, zz in zip(axes, available_planes):
        ax.imshow(selected_images[zz], cmap="inferno", origin="lower")
        row = next(r for r in rows if abs(r["z_m"] - zz) < 1e-12)
        ax.set_title(f"z = {zz:g} m\nD4σ = {row['d4sigma_mean_mm']:.2f} mm")
        ax.set_xticks([])
        ax.set_yticks([])
    suptitle = "Propagated intensity at selected distances"
    if use_lens:
        suptitle += f"\nThin lens: f = {args.focal_length_m:g} m, position z = {args.lens_position_m:g} m"
    fig.suptitle(suptitle)
    fig.savefig(output_dir / "selected_planes_with_lens.png", dpi=180)
    plt.close(fig)

    print(json.dumps({
        "output_dir": str(output_dir),
        "pixel_size_um": dx_m * 1e6,
        "input_d4sigma_mm": args.input_d4sigma_mm,
        "use_lens": use_lens,
        "focal_length_m": args.focal_length_m if use_lens else None,
        "lens_position_m": args.lens_position_m if use_lens else None,
        "beam_at_10m_mm": next((r["d4sigma_mean_mm"] for r in rows if abs(r["z_m"] - 10.0) < 1e-12), None),
    }, indent=2))

if __name__ == "__main__":
    main()
