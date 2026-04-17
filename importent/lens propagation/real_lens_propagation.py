from __future__ import annotations

import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# KONFIGURATION
# ============================================================

CFG = {
    # Ordner mit den realen WCF-Dateien
    "REAL_WCF_FOLDER": Path(r"C:\Users\User\Desktop\Real lens propagation"),

    # Gemeinsamer Ausgabeordner
    "OUTPUT_DIR": Path.home() / "Desktop" / "lens propagation",

    # Reale Skalierung:
    # 11 px/mm bedeutet: 1 px = 1/11 mm
    "REAL_PX_PER_MM": 110,

    # Kandidaten für Start des Pixelblocks im WCF-Frame
    "CANDIDATE_PIXEL_OFFSETS": [944, 960, 992, 1024, 1152, 1200],

    # Area-Schwellwerte relativ zum Peak
    "REL_THRESHOLDS": {
        "D_area_50pct": 0.50,
        "D_area_13p5pct": math.exp(-2.0),  # ~ 1/e^2
    },

    # Encircled Energy
    "EE_TARGETS": {
        "D_EE50": 0.50,
        "D_EE86": 0.865,
    },

    # FWHM-Definition:
    # "cross_section" = horizontale + vertikale Linie durch Schwerpunkt, dann Mittelwert
    # "integrated"    = integrierte 1D-Profile entlang x und y, dann Mittelwert
    "FWHM_DEFINITION": "cross_section",
}


# ============================================================
# HILFSFUNKTIONEN WCF
# ============================================================

def u32(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset:offset + 4], "little", signed=False)


def parse_z_from_filename(path: Path) -> float:
    """
    Extrahiert z aus Dateinamen wie:
    350.wcf, 350cm.wcf, z_350.wcf, scan_350_cm.wcf

    Rückgabe:
        z_cm
    """
    stem = path.stem.replace(",", ".")
    m = re.search(r"(\d+(?:\.\d+)?)", stem)
    if not m:
        raise ValueError(f"Keine z-Position im Dateinamen gefunden: {path.name}")
    return float(m.group(1))


def border_values(img: np.ndarray, border: int = 40) -> np.ndarray:
    top = img[:border, :]
    bottom = img[-border:, :]
    left = img[:, :border]
    right = img[:, -border:]
    return np.concatenate([top.ravel(), bottom.ravel(), left.ravel(), right.ravel()])


def choose_pixel_offset(frame_bytes: bytes, width: int, height: int, candidates: list[int]) -> int:
    best_offset = None
    best_score = np.inf

    npx = width * height
    need = npx * 2

    for off in candidates:
        if off + need > len(frame_bytes):
            continue

        img = np.frombuffer(
            frame_bytes, dtype="<u2", count=npx, offset=off
        ).reshape(height, width).astype(np.float64)

        b = border_values(img, border=min(40, width // 8, height // 8))
        center = img[height // 4: 3 * height // 4, width // 4: 3 * width // 4]

        border_mean = float(np.mean(b))
        center_mean = float(np.mean(center))
        max_y, max_x = np.unravel_index(np.argmax(img), img.shape)

        on_edge = (max_x < 5 or max_x >= width - 5 or max_y < 5 or max_y >= height - 5)

        score = border_mean / max(center_mean, 1e-12)
        if on_edge:
            score += 0.5

        if score < best_score:
            best_score = score
            best_offset = off

    if best_offset is None:
        raise RuntimeError("Kein plausibler Pixel-Offset im WCF-Frame gefunden.")

    return best_offset


def read_wcf_mean_image(path: Path) -> np.ndarray:
    """
    Liest eine DataRay-artige WCF-Datei und mittelt alle nutzbaren Frames.
    """
    data = path.read_bytes()

    if not data.startswith(b".IRD"):
        raise RuntimeError(f"{path.name}: Unbekannter WCF-Header (erwartet '.IRD').")

    global_header = u32(data, 8)
    nominal_frames = u32(data, 12)
    frame_size = u32(data, 16)

    if global_header <= 0 or frame_size <= 0:
        raise RuntimeError(f"{path.name}: Ungültiger WCF-Header.")

    if len(data) < global_header + 256:
        raise RuntimeError(f"{path.name}: Datei zu kurz.")

    first_frame_offset = global_header
    width = u32(data, first_frame_offset + 20)
    height = u32(data, first_frame_offset + 24)

    if width <= 0 or height <= 0 or width * height > 10000 * 10000:
        raise RuntimeError(f"{path.name}: Unplausible Bildgröße: {width} x {height}")

    usable_frames = max(1, min(nominal_frames, (len(data) - global_header) // frame_size))

    first_frame_end = min(len(data), first_frame_offset + frame_size)
    first_frame = data[first_frame_offset:first_frame_end]
    pixel_offset = choose_pixel_offset(first_frame, width, height, CFG["CANDIDATE_PIXEL_OFFSETS"])

    acc = np.zeros((height, width), dtype=np.float64)
    used = 0
    bytes_needed = pixel_offset + width * height * 2

    for i in range(usable_frames):
        f0 = global_header + i * frame_size
        f1 = min(len(data), f0 + frame_size)
        frame = data[f0:f1]

        if len(frame) < bytes_needed:
            continue

        img = np.frombuffer(
            frame, dtype="<u2", count=width * height, offset=pixel_offset
        ).reshape(height, width)

        acc += img
        used += 1

    if used == 0:
        raise RuntimeError(f"{path.name}: Kein vollständiger Frame lesbar.")

    return acc / used


# ============================================================
# METRIKEN
# ============================================================

def subtract_background(img: np.ndarray) -> np.ndarray:
    bg = np.percentile(img, 8)
    out = img.astype(np.float64) - bg
    out[out < 0] = 0.0
    return out


def weighted_centroid(img: np.ndarray) -> tuple[float, float]:
    total = float(img.sum())
    if total <= 0:
        raise RuntimeError("Bild hat nach Hintergrundabzug keine positive Intensität.")

    y, x = np.indices(img.shape)
    cx = float((img * x).sum() / total)
    cy = float((img * y).sum() / total)
    return cx, cy


def second_moment_diameter_avg(img: np.ndarray) -> dict[str, float]:
    total = float(img.sum())
    y, x = np.indices(img.shape)
    cx, cy = weighted_centroid(img)

    dx = x - cx
    dy = y - cy

    sigma_x = math.sqrt(float((img * dx**2).sum() / total))
    sigma_y = math.sqrt(float((img * dy**2).sum() / total))

    d4sigma_x = 4.0 * sigma_x
    d4sigma_y = 4.0 * sigma_y
    d4sigma_avg = 0.5 * (d4sigma_x + d4sigma_y)

    return {
        "D4sigma_x": d4sigma_x,
        "D4sigma_y": d4sigma_y,
        "D4sigma_avg": d4sigma_avg,
    }


def encircled_energy_diameter(img: np.ndarray, fraction: float) -> float:
    cx, cy = weighted_centroid(img)
    y, x = np.indices(img.shape)
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    order = np.argsort(r.ravel())
    r_sorted = r.ravel()[order]
    i_sorted = img.ravel()[order]

    cum = np.cumsum(i_sorted)
    total = cum[-1]
    target = fraction * total

    idx = int(np.searchsorted(cum, target))
    idx = min(max(idx, 0), len(r_sorted) - 1)
    return 2.0 * float(r_sorted[idx])


def area_equivalent_diameter(img: np.ndarray, rel_threshold: float) -> float:
    peak = float(np.max(img))
    if peak <= 0:
        return np.nan
    mask = img >= rel_threshold * peak
    area = float(np.count_nonzero(mask))
    if area <= 0:
        return np.nan
    return 2.0 * math.sqrt(area / math.pi)


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


def fwhm_avg_px(img: np.ndarray, definition: str = "cross_section") -> dict[str, float]:
    cx, cy = weighted_centroid(img)

    x_axis = np.arange(img.shape[1]) - cx
    y_axis = np.arange(img.shape[0]) - cy

    cx_i = int(np.clip(round(cx), 0, img.shape[1] - 1))
    cy_i = int(np.clip(round(cy), 0, img.shape[0] - 1))

    if definition == "cross_section":
        profile_x = img[cy_i, :]
        profile_y = img[:, cx_i]
    elif definition == "integrated":
        profile_x = img.sum(axis=0)
        profile_y = img.sum(axis=1)
    else:
        raise ValueError("FWHM_DEFINITION must be 'cross_section' or 'integrated'.")

    fwhm_x = fwhm_1d(x_axis, profile_x)
    fwhm_y = fwhm_1d(y_axis, profile_y)
    fwhm_avg = 0.5 * (fwhm_x + fwhm_y)

    return {
        "FWHM_x": fwhm_x,
        "FWHM_y": fwhm_y,
        "FWHM_avg": fwhm_avg,
    }


def compute_metrics_px(img: np.ndarray) -> dict[str, float]:
    img2 = subtract_background(img)

    metrics = {}
    metrics.update(second_moment_diameter_avg(img2))

    for name, frac in CFG["EE_TARGETS"].items():
        metrics[name] = encircled_energy_diameter(img2, frac)

    for name, thr in CFG["REL_THRESHOLDS"].items():
        metrics[name] = area_equivalent_diameter(img2, thr)

    metrics.update(fwhm_avg_px(img2, CFG["FWHM_DEFINITION"]))
    return metrics


def px_to_mm(values: dict[str, float]) -> dict[str, float]:
    mm_per_px = 1.0 / float(CFG["REAL_PX_PER_MM"])
    return {k: v * mm_per_px for k, v in values.items()}


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    real_folder = Path(CFG["REAL_WCF_FOLDER"]).expanduser()
    output_dir = Path(CFG["OUTPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if not real_folder.exists():
        raise SystemExit(f"Ordner nicht gefunden: {real_folder}")

    files = sorted(real_folder.glob("*.wcf"))
    if not files:
        raise SystemExit(f"Keine .wcf-Dateien gefunden in: {real_folder}")

    rows = []
    for path in files:
        z_cm = parse_z_from_filename(path)
        img = read_wcf_mean_image(path)
        metrics_px = compute_metrics_px(img)
        metrics_mm = px_to_mm(metrics_px)

        row = {
            "file": path.name,
            "z_cm": z_cm,
            "z_m": z_cm / 100.0,
        }
        row.update({
            "D4sigma_x_mm": metrics_mm["D4sigma_x"],
            "D4sigma_y_mm": metrics_mm["D4sigma_y"],
            "D4sigma_avg_mm": metrics_mm["D4sigma_avg"],
            "D_EE50_mm": metrics_mm["D_EE50"],
            "D_EE86_mm": metrics_mm["D_EE86"],
            "D_area_50pct_mm": metrics_mm["D_area_50pct"],
            "D_area_13p5pct_mm": metrics_mm["D_area_13p5pct"],
            "FWHM_x_mm": metrics_mm["FWHM_x"],
            "FWHM_y_mm": metrics_mm["FWHM_y"],
            "FWHM_mm": metrics_mm["FWHM_avg"],
        })
        rows.append(row)
        print(f"OK: {path.name}")

    rows.sort(key=lambda r: r["z_cm"])

    csv_path = output_dir / "real_lens_propagation.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    z = np.array([r["z_cm"] for r in rows], dtype=float)

    plt.figure(figsize=(11, 7))
    plt.plot(z, [r["D4sigma_avg_mm"] for r in rows], marker="o", linewidth=2, label="D4σ")
    plt.plot(z, [r["D_EE50_mm"] for r in rows], marker="o", linewidth=2, label="EE50")
    plt.plot(z, [r["D_EE86_mm"] for r in rows], marker="o", linewidth=2, label="EE86")
    plt.plot(z, [r["D_area_50pct_mm"] for r in rows], marker="o", linewidth=2, label="Area50%")
    plt.plot(z, [r["D_area_13p5pct_mm"] for r in rows], marker="o", linewidth=2, label="Area13.5%")
    plt.plot(z, [r["FWHM_mm"] for r in rows], marker="o", linewidth=2, label=f"FWHM ({CFG['FWHM_DEFINITION']})")

    plt.xlabel("z [cm]")
    plt.ylabel("Beam size [mm]")
    plt.title("Real lens propagation")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plot_path = output_dir / "real_lens_propagation.png"
    plt.savefig(plot_path, dpi=220)
    plt.show()

    print()
    print("Fertig.")
    print(f"CSV:  {csv_path}")
    print(f"PNG:  {plot_path}")


if __name__ == "__main__":
    main()