from __future__ import annotations

import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# =========================
# Einstellungen
# =========================

DEFAULT_FOLDER = Path.home() / "Desktop" / "focus scan 5 new"

# Wenn None, wird in Pixeln geplottet
PIXEL_SIZE_UM: float | None = 5.5

# Kandidaten für den Start des Pixelblocks innerhalb eines WCF-Frames
CANDIDATE_PIXEL_OFFSETS = [944, 960, 992, 1024, 1152, 1200]

# Schwellen für area-equivalent Diameter relativ zum Peak
REL_THRESHOLDS = {
    "D_area_50pct": 0.50,
    "D_area_13p5pct": math.exp(-2.0),  # ~1/e^2
}

# Encircled energy targets
EE_TARGETS = {
    "D_EE50": 0.50,
    "D_EE86": 0.865,
    "D_EE95": 0.95,
}

# Einheitliche Vorverarbeitung
SMOOTHING_PASSES = 1
ROI_MARGIN_FACTOR = 3.0
ROI_MIN_HALF_SIZE = 120
ROI_MAX_HALF_SIZE = 420
SAVE_PREPARED_IMAGES = True


def u32(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset:offset + 4], "little", signed=False)


def parse_z_from_filename(path: Path) -> float:
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

        img = np.frombuffer(frame_bytes, dtype="<u2", count=npx, offset=off).reshape(height, width).astype(np.float64)
        b = border_values(img, border=min(40, width // 8, height // 8))
        center = img[height // 4: 3 * height // 4, width // 4: 3 * width // 4]

        border_mean = float(np.mean(b))
        center_mean = float(np.mean(center))
        max_y, max_x = np.unravel_index(np.argmax(img), img.shape)

        on_edge = (max_x < 5 or max_x >= width - 5 or max_y < 5 or max_y >= height - 5)
        score = border_mean / max(center_mean, 1e-9)
        if on_edge:
            score += 0.5

        if score < best_score:
            best_score = score
            best_offset = off

    if best_offset is None:
        raise RuntimeError("Kein plausibler Pixel-Offset im WCF-Frame gefunden.")

    return best_offset


def read_wcf_mean_image(path: Path) -> np.ndarray:
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
    pixel_offset = choose_pixel_offset(first_frame, width, height, CANDIDATE_PIXEL_OFFSETS)

    acc = np.zeros((height, width), dtype=np.float64)
    used = 0
    bytes_needed = pixel_offset + width * height * 2

    for i in range(usable_frames):
        f0 = global_header + i * frame_size
        f1 = min(len(data), f0 + frame_size)
        frame = data[f0:f1]
        if len(frame) < bytes_needed:
            continue

        img = np.frombuffer(frame, dtype="<u2", count=width * height, offset=pixel_offset).reshape(height, width)
        acc += img
        used += 1

    if used == 0:
        raise RuntimeError(f"{path.name}: Kein vollständiger Frame lesbar.")

    return acc / used


def subtract_background(img: np.ndarray) -> tuple[np.ndarray, float]:
    border = min(40, img.shape[0] // 8, img.shape[1] // 8)
    bg = float(np.median(border_values(img, border=border)))
    out = img.astype(np.float64) - bg
    out[out < 0] = 0.0
    return out, bg


def smooth3x3(img: np.ndarray, passes: int = 1) -> np.ndarray:
    out = img.astype(np.float64).copy()
    for _ in range(max(0, passes)):
        p = np.pad(out, 1, mode="edge")
        out = (
            p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
            p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:] +
            p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]
        ) / 9.0
    return out


def weighted_centroid(img: np.ndarray) -> tuple[float, float]:
    total = float(img.sum())
    if total <= 0:
        raise RuntimeError("Bild hat nach Vorverarbeitung keine positive Intensität.")

    y, x = np.indices(img.shape)
    cx = float((img * x).sum() / total)
    cy = float((img * y).sum() / total)
    return cx, cy


def second_moment_diameters(img: np.ndarray) -> dict[str, float]:
    total = float(img.sum())
    y, x = np.indices(img.shape)
    cx, cy = weighted_centroid(img)

    dx = x - cx
    dy = y - cy

    sigma_x = math.sqrt(float((img * dx**2).sum() / total))
    sigma_y = math.sqrt(float((img * dy**2).sum() / total))

    d4sx = 4.0 * sigma_x
    d4sy = 4.0 * sigma_y
    d4s_eq = math.sqrt(d4sx * d4sy)

    return {
        "D4sigma_x": d4sx,
        "D4sigma_y": d4sy,
        "D4sigma_eq": d4s_eq,
    }


def encircled_energy_diameter(img: np.ndarray, fraction: float) -> float:
    cx, cy = weighted_centroid(img)
    y, x = np.indices(img.shape)
    r = np.sqrt((x - cx)**2 + (y - cy)**2)

    order = np.argsort(r.ravel())
    r_sorted = r.ravel()[order]
    i_sorted = img.ravel()[order]

    cum = np.cumsum(i_sorted)
    total = cum[-1]
    if total <= 0:
        return np.nan

    target = fraction * total
    idx = int(np.searchsorted(cum, target))
    idx = min(max(idx, 0), len(r_sorted) - 1)
    return 2.0 * float(r_sorted[idx])


def area_equivalent_diameter(img: np.ndarray, rel_threshold: float) -> float:
    peak = float(np.max(img))
    if peak <= 0:
        return np.nan
    mask = img >= (rel_threshold * peak)
    area = float(np.count_nonzero(mask))
    if area <= 0:
        return np.nan
    return 2.0 * math.sqrt(area / math.pi)


def crop_around_center(img: np.ndarray, cx: float, cy: float, half_size: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = img.shape
    x0 = max(0, int(round(cx)) - half_size)
    x1 = min(w, int(round(cx)) + half_size + 1)
    y0 = max(0, int(round(cy)) - half_size)
    y1 = min(h, int(round(cy)) + half_size + 1)
    return img[y0:y1, x0:x1].copy(), (x0, x1, y0, y1)


def percentile_clip_for_preview(img: np.ndarray) -> np.ndarray:
    if np.max(img) <= 0:
        return img.copy()
    vmax = np.percentile(img, 99.8)
    if vmax <= 0:
        vmax = np.max(img)
    return np.clip(img, 0, vmax)


def prepare_image(raw_img: np.ndarray, roi_half_size: int) -> dict:
    bg_sub, bg = subtract_background(raw_img)
    smoothed = smooth3x3(bg_sub, passes=SMOOTHING_PASSES)

    cx0, cy0 = weighted_centroid(smoothed)
    roi, bounds = crop_around_center(smoothed, cx0, cy0, roi_half_size)
    cx_roi, cy_roi = weighted_centroid(roi)

    return {
        "prepared": roi,
        "background_level": bg,
        "cx_full": cx0,
        "cy_full": cy0,
        "cx_roi": cx_roi,
        "cy_roi": cy_roi,
        "roi_bounds": bounds,
    }


def compute_metrics(prepared_img: np.ndarray) -> dict[str, float]:
    metrics = {}
    metrics.update(second_moment_diameters(prepared_img))

    for name, frac in EE_TARGETS.items():
        metrics[name] = encircled_energy_diameter(prepared_img, frac)

    for name, thr in REL_THRESHOLDS.items():
        metrics[name] = area_equivalent_diameter(prepared_img, thr)

    return metrics


def maybe_scale(values: dict[str, float]) -> tuple[dict[str, float], str]:
    if PIXEL_SIZE_UM is None:
        return values, "px"

    scaled = {}
    for k, v in values.items():
        scaled[k] = v * PIXEL_SIZE_UM / 1000.0
    return scaled, "mm"


def estimate_global_roi_half_size(raw_images: list[np.ndarray]) -> int:
    diameters = []

    for img in raw_images:
        bg_sub, _ = subtract_background(img)
        smoothed = smooth3x3(bg_sub, passes=SMOOTHING_PASSES)
        try:
            mets = second_moment_diameters(smoothed)
            diameters.append(max(mets["D4sigma_x"], mets["D4sigma_y"]))
        except Exception:
            continue

    if not diameters:
        return ROI_MIN_HALF_SIZE

    half_size = int(math.ceil(ROI_MARGIN_FACTOR * max(diameters)))
    half_size = max(ROI_MIN_HALF_SIZE, min(ROI_MAX_HALF_SIZE, half_size))
    return half_size


def save_prepared_preview(prepared_img: np.ndarray, outpath: Path, title: str) -> None:
    plt.figure(figsize=(6, 5))
    show = percentile_clip_for_preview(prepared_img)
    plt.imshow(show, cmap="inferno", origin="lower")
    plt.colorbar(label="a.u.")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def main(folder: Path = DEFAULT_FOLDER) -> None:
    folder = Path(folder).expanduser()

    if not folder.exists():
        raise SystemExit(f"Ordner nicht gefunden: {folder}")

    files = sorted(folder.glob("*.wcf"))
    if not files:
        raise SystemExit(f"Keine .wcf-Dateien gefunden in: {folder}")

    raw_entries: list[dict] = []
    for path in files:
        z_cm = parse_z_from_filename(path)
        raw_img = read_wcf_mean_image(path)
        raw_entries.append({
            "file": path.name,
            "path": path,
            "z_cm": z_cm,
            "raw_img": raw_img,
        })
        print(f"Rohbild gelesen: {path.name}")

    raw_entries.sort(key=lambda r: r["z_cm"])

    roi_half_size = estimate_global_roi_half_size([e["raw_img"] for e in raw_entries])
    print(f"Einheitliche ROI-Halbgröße: {roi_half_size} px")

    prepared_dir = folder / "prepared_images"
    if SAVE_PREPARED_IMAGES:
        prepared_dir.mkdir(exist_ok=True)

    prepared_entries: list[dict] = []
    for e in raw_entries:
        prep = prepare_image(e["raw_img"], roi_half_size=roi_half_size)
        prepared_img = prep["prepared"]

        metrics = compute_metrics(prepared_img)
        metrics_scaled, unit = maybe_scale(metrics)

        row = {
            "file": e["file"],
            "z_cm": e["z_cm"],
            "background_level": prep["background_level"],
            "roi_half_size_px": roi_half_size,
            "cx_full_px": prep["cx_full"],
            "cy_full_px": prep["cy_full"],
            "cx_roi_px": prep["cx_roi"],
            "cy_roi_px": prep["cy_roi"],
        }
        row.update(metrics_scaled)
        prepared_entries.append(row)

        if SAVE_PREPARED_IMAGES:
            outpng = prepared_dir / f"{e['path'].stem}_prepared.png"
            save_prepared_preview(prepared_img, outpng, f"{e['path'].name} (prepared)")

        print(f"Vorbereitet und ausgewertet: {e['file']}")

    prepared_entries.sort(key=lambda r: r["z_cm"])

    csv_path = folder / "beam_size_vs_z_prepared.csv"
    fieldnames = list(prepared_entries[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(prepared_entries)

    z = np.array([r["z_cm"] for r in prepared_entries], dtype=float)

    plt.figure(figsize=(11, 7))
    plot_keys = [
        "D4sigma_eq",
        "D_EE50",
        "D_EE86",
        "D_EE95",
        "D_area_50pct",
        "D_area_13p5pct",
    ]

    for key in plot_keys:
        y = np.array([r[key] for r in prepared_entries], dtype=float)
        plt.plot(z, y, marker="o", linewidth=2, label=key)

    plt.xlabel("z [cm]")
    plt.ylabel(f"Beam size [{unit}]")
    plt.title("Laser beam size in propagation direction (prepared images)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plot_path = folder / "beam_size_vs_z_prepared.png"
    plt.savefig(plot_path, dpi=180)
    plt.show()

    plt.figure(figsize=(11, 6))
    plt.plot(z, [r["D4sigma_x"] for r in prepared_entries], marker="o", linewidth=2, label="D4sigma_x")
    plt.plot(z, [r["D4sigma_y"] for r in prepared_entries], marker="o", linewidth=2, label="D4sigma_y")
    plt.plot(z, [r["D4sigma_eq"] for r in prepared_entries], marker="o", linewidth=2, label="D4sigma_eq")
    plt.xlabel("z [cm]")
    plt.ylabel(f"Beam size [{unit}]")
    plt.title("D4σ-Diameter in z (prepared images)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plot2_path = folder / "beam_size_vs_z_d4sigma_prepared.png"
    plt.savefig(plot2_path, dpi=180)
    plt.show()

    print()
    print("Fertig.")
    print(f"CSV:   {csv_path}")
    print(f"Plot:  {plot_path}")
    print(f"Plot2: {plot2_path}")
    if SAVE_PREPARED_IMAGES:
        print(f"Prepared PNGs: {prepared_dir}")


if __name__ == "__main__":
    main()
