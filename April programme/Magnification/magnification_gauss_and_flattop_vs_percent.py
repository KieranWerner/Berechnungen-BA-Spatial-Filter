from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})


# ============================================================
# KONFIGURATION
# ============================================================

CFG = {
    # Zwei Beam-Bilder:
    #   FRONT_IMAGE   = Referenz / vorher
    #   REDUCED_IMAGE = verkleinerter / nachher
    "FRONT_IMAGE": Path(r"C:\Users\User\Desktop\Lens beam size plot\350.tiff"),
    "REDUCED_IMAGE": Path(r"C:\Users\User\Desktop\2D Beamprofiler free\messung ende.tiff"),

    # Ausgabenergebnisse
    "OUTPUT_DIR": Path.home() / "Desktop" / "magnification_fit_analysis",

    # Pixelgröße:
    # None  -> Ausgabe in Pixel
    # Zahl  -> Ausgabe in mm (über Pixelgröße in µm)
    "FRONT_PIXEL_SIZE_UM": None,
    "REDUCED_PIXEL_SIZE_UM": None,

    # Optionaler Crop um das Zentrum des Beams
    "USE_CROP": False,
    "FRONT_CROP": None,      # z.B. (x0, x1, y0, y1)
    "REDUCED_CROP": None,    # z.B. (x0, x1, y0, y1)

    # Hintergrund entfernen
    "SUBTRACT_BACKGROUND": True,
    "BACKGROUND_PERCENTILE": 5.0,

    # Fit-Bereich
    "PERCENT_VALUES": list(range(10, 100, 10)),   # 10,20,...,90 %

    # Startwerte / Grenzen für Super-Gaussian (Flat-Top)
    "SUPER_GAUSS_ORDER_START": 6.0,
    "SUPER_GAUSS_ORDER_MIN": 2.0,
    "SUPER_GAUSS_ORDER_MAX": 30.0,
}


# ============================================================
# HILFSFUNKTIONEN
# ============================================================

def load_image(path: Path) -> np.ndarray:
    img = Image.open(path)
    arr = np.array(img, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    return arr


def maybe_crop(img: np.ndarray, crop):
    if crop is None:
        return img
    x0, x1, y0, y1 = crop
    return img[y0:y1, x0:x1]


def preprocess_image(img: np.ndarray, subtract_background: bool, bg_percentile: float) -> np.ndarray:
    data = np.asarray(img, dtype=np.float64)

    if subtract_background:
        bg = np.percentile(data, bg_percentile)
        data = data - bg

    data = np.clip(data, 0, None)

    maxv = np.max(data)
    if maxv <= 0:
        raise ValueError("Bild enthält nach Preprocessing keine positive Intensität.")
    data = data / maxv
    return data


def centroid_pixels(I: np.ndarray):
    I = np.clip(np.asarray(I, dtype=float), 0, None)
    P = float(I.sum())
    if P <= 0:
        raise ValueError("Intensitätssumme ist 0.")
    y = np.arange(I.shape[0])
    x = np.arange(I.shape[1])
    X, Y = np.meshgrid(x, y)
    x0 = float((I * X).sum() / P)
    y0 = float((I * Y).sum() / P)
    return x0, y0, P, X, Y


def second_moments(I: np.ndarray):
    x0, y0, P, X, Y = centroid_pixels(I)
    Xc = X - x0
    Yc = Y - y0
    cov_xx = float((I * Xc * Xc).sum() / P)
    cov_yy = float((I * Yc * Yc).sum() / P)
    cov_xy = float((I * Xc * Yc).sum() / P)
    return x0, y0, cov_xx, cov_yy, cov_xy


def ellipse_from_cov(cov_xx: float, cov_yy: float, cov_xy: float):
    cov = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]], dtype=float)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    major_var = max(float(vals[0]), 1e-15)
    minor_var = max(float(vals[1]), 1e-15)
    theta = float(np.arctan2(vecs[1, 0], vecs[0, 0]))
    return np.sqrt(major_var), np.sqrt(minor_var), theta


def rotated_coords(X, Y, x0, y0, theta):
    ct = np.cos(theta)
    st = np.sin(theta)
    Xc = X - x0
    Yc = Y - y0
    Xr = ct * Xc + st * Yc
    Yr = -st * Xc + ct * Yc
    return Xr, Yr


# ============================================================
# FIT-MODELLE
# ============================================================

def gaussian_model(coords, offset, amplitude, x0, y0, w_major, w_minor, theta):
    X, Y = coords
    Xr, Yr = rotated_coords(X, Y, x0, y0, theta)
    expo = -2.0 * ((Xr / w_major) ** 2 + (Yr / w_minor) ** 2)
    return offset + amplitude * np.exp(expo)


def supergaussian_model(coords, offset, amplitude, x0, y0, w_major, w_minor, theta, order):
    X, Y = coords
    Xr, Yr = rotated_coords(X, Y, x0, y0, theta)
    R = (np.abs(Xr / w_major) ** order + np.abs(Yr / w_minor) ** order)
    return offset + amplitude * np.exp(-2.0 * R)


def fit_gaussian_2d(I: np.ndarray) -> dict:
    I = np.clip(np.asarray(I, dtype=float), 0, None)

    y = np.arange(I.shape[0], dtype=float)
    x = np.arange(I.shape[1], dtype=float)
    X, Y = np.meshgrid(x, y)

    x0, y0, cov_xx, cov_yy, cov_xy = second_moments(I)
    sigma_major_px, sigma_minor_px, theta0 = ellipse_from_cov(cov_xx, cov_yy, cov_xy)

    offset0 = float(np.percentile(I, 5))
    amp0 = float(np.max(I) - offset0)
    w_major0 = max(np.sqrt(2.0) * sigma_major_px, 1e-6)
    w_minor0 = max(np.sqrt(2.0) * sigma_minor_px, 1e-6)

    p0 = np.array([offset0, amp0, x0, y0, w_major0, w_minor0, theta0], dtype=float)

    try:
        from scipy.optimize import curve_fit

        lower = [0.0, 0.0, x.min(), y.min(), 1e-6, 1e-6, -np.pi / 2]
        upper = [1.0, 2.0, x.max(), y.max(), I.shape[1], I.shape[0], np.pi / 2]

        popt, _ = curve_fit(
            gaussian_model,
            (X.ravel(), Y.ravel()),
            I.ravel(),
            p0=p0,
            bounds=(lower, upper),
            maxfev=100000,
        )
    except Exception:
        popt = p0

    offset, amplitude, x0, y0, w_major, w_minor, theta = [float(v) for v in popt]

    if w_minor > w_major:
        w_major, w_minor = w_minor, w_major
        theta += np.pi / 2

    while theta <= -np.pi / 2:
        theta += np.pi
    while theta > np.pi / 2:
        theta -= np.pi

    return {
        "offset": offset,
        "amplitude": amplitude,
        "x0_px": x0,
        "y0_px": y0,
        "w_major_px": w_major,
        "w_minor_px": w_minor,
        "theta_rad": theta,
    }


def fit_supergaussian_2d(I: np.ndarray, cfg: dict) -> dict:
    I = np.clip(np.asarray(I, dtype=float), 0, None)

    y = np.arange(I.shape[0], dtype=float)
    x = np.arange(I.shape[1], dtype=float)
    X, Y = np.meshgrid(x, y)

    x0, y0, cov_xx, cov_yy, cov_xy = second_moments(I)
    sigma_major_px, sigma_minor_px, theta0 = ellipse_from_cov(cov_xx, cov_yy, cov_xy)

    offset0 = float(np.percentile(I, 5))
    amp0 = float(np.max(I) - offset0)
    w_major0 = max(2.0 * sigma_major_px, 1e-6)
    w_minor0 = max(2.0 * sigma_minor_px, 1e-6)
    order0 = float(cfg["SUPER_GAUSS_ORDER_START"])

    p0 = np.array([offset0, amp0, x0, y0, w_major0, w_minor0, theta0, order0], dtype=float)

    try:
        from scipy.optimize import curve_fit

        lower = [
            0.0, 0.0, x.min(), y.min(), 1e-6, 1e-6, -np.pi / 2, float(cfg["SUPER_GAUSS_ORDER_MIN"])
        ]
        upper = [
            1.0, 2.0, x.max(), y.max(), I.shape[1], I.shape[0], np.pi / 2, float(cfg["SUPER_GAUSS_ORDER_MAX"])
        ]

        popt, _ = curve_fit(
            supergaussian_model,
            (X.ravel(), Y.ravel()),
            I.ravel(),
            p0=p0,
            bounds=(lower, upper),
            maxfev=150000,
        )
    except Exception:
        popt = p0

    offset, amplitude, x0, y0, w_major, w_minor, theta, order = [float(v) for v in popt]

    if w_minor > w_major:
        w_major, w_minor = w_minor, w_major
        theta += np.pi / 2

    while theta <= -np.pi / 2:
        theta += np.pi
    while theta > np.pi / 2:
        theta -= np.pi

    return {
        "offset": offset,
        "amplitude": amplitude,
        "x0_px": x0,
        "y0_px": y0,
        "w_major_px": w_major,
        "w_minor_px": w_minor,
        "theta_rad": theta,
        "order": order,
    }


# ============================================================
# SIZE-BERECHNUNG AUS FITS
# ============================================================

def gaussian_size_at_percent(fit: dict, percent: float) -> dict:
    p = percent / 100.0
    if not (0.0 < p < 1.0):
        raise ValueError("percent muss zwischen 0 und 100 liegen.")
    factor = np.sqrt(-0.5 * np.log(p))
    d_major = 2.0 * fit["w_major_px"] * factor
    d_minor = 2.0 * fit["w_minor_px"] * factor
    d_mean = 0.5 * (d_major + d_minor)
    return {
        "diameter_major_px": d_major,
        "diameter_minor_px": d_minor,
        "diameter_mean_px": d_mean,
    }


def supergaussian_size_at_percent(fit: dict, percent: float) -> dict:
    p = percent / 100.0
    if not (0.0 < p < 1.0):
        raise ValueError("percent muss zwischen 0 und 100 liegen.")
    factor = (-0.5 * np.log(p)) ** (1.0 / fit["order"])
    d_major = 2.0 * fit["w_major_px"] * factor
    d_minor = 2.0 * fit["w_minor_px"] * factor
    d_mean = 0.5 * (d_major + d_minor)
    return {
        "diameter_major_px": d_major,
        "diameter_minor_px": d_minor,
        "diameter_mean_px": d_mean,
    }


def convert_px_to_output_units(values_px: np.ndarray, pixel_size_um):
    values_px = np.asarray(values_px, dtype=float)
    if pixel_size_um is None:
        return values_px, "px"
    values_mm = values_px * float(pixel_size_um) * 1e-3
    return values_mm, "mm"


def analyse_one_image(img: np.ndarray, cfg: dict) -> dict:
    gfit = fit_gaussian_2d(img)
    ffit = fit_supergaussian_2d(img, cfg)
    return {
        "gaussian_fit": gfit,
        "flat_top_fit": ffit,
    }


def build_result_table(percent_values, fit_front, fit_reduced, front_pixel_size_um, reduced_pixel_size_um, fit_type: str):
    rows = []
    for percent in percent_values:
        if fit_type == "gaussian":
            front_sizes = gaussian_size_at_percent(fit_front, percent)
            reduced_sizes = gaussian_size_at_percent(fit_reduced, percent)
        elif fit_type == "flat_top":
            front_sizes = supergaussian_size_at_percent(fit_front, percent)
            reduced_sizes = supergaussian_size_at_percent(fit_reduced, percent)
        else:
            raise ValueError("fit_type muss 'gaussian' oder 'flat_top' sein.")

        front_mean_out, unit_front = convert_px_to_output_units(front_sizes["diameter_mean_px"], front_pixel_size_um)
        reduced_mean_out, unit_reduced = convert_px_to_output_units(reduced_sizes["diameter_mean_px"], reduced_pixel_size_um)

        # Vergrößerung als Verhältnis der mittleren Durchmesser
        magnification = float(reduced_mean_out / front_mean_out)

        rows.append({
            "percent": float(percent),
            f"front_mean_{unit_front}": float(front_mean_out),
            f"reduced_mean_{unit_reduced}": float(reduced_mean_out),
            "magnification": magnification,
        })
    return rows


def save_csv_like_json(path: Path, rows: list[dict]):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_plot(rows: list[dict], output_path: Path, title: str, ylabel: str, front_label: str, reduced_label: str):
    percent = np.array([r["percent"] for r in rows], dtype=float)

    front_key = [k for k in rows[0].keys() if k.startswith("front_mean_")][0]
    reduced_key = [k for k in rows[0].keys() if k.startswith("reduced_mean_")][0]

    y_front = np.array([r[front_key] for r in rows], dtype=float)
    y_reduced = np.array([r[reduced_key] for r in rows], dtype=float)

    plt.figure(figsize=(10, 7))
    plt.plot(percent, y_front, "o-", linewidth=2, markersize=7, label=front_label)
    plt.plot(percent, y_reduced, "s-", linewidth=2, markersize=7, label=reduced_label)

    plt.xlabel("Intensitätsniveau [% vom Peak]")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def main():
    cfg = dict(CFG)
    output_dir = Path(cfg["OUTPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)

    front_raw = load_image(Path(cfg["FRONT_IMAGE"]))
    reduced_raw = load_image(Path(cfg["REDUCED_IMAGE"]))

    if cfg["USE_CROP"]:
        front_raw = maybe_crop(front_raw, cfg["FRONT_CROP"])
        reduced_raw = maybe_crop(reduced_raw, cfg["REDUCED_CROP"])

    front = preprocess_image(
        front_raw,
        subtract_background=bool(cfg["SUBTRACT_BACKGROUND"]),
        bg_percentile=float(cfg["BACKGROUND_PERCENTILE"]),
    )
    reduced = preprocess_image(
        reduced_raw,
        subtract_background=bool(cfg["SUBTRACT_BACKGROUND"]),
        bg_percentile=float(cfg["BACKGROUND_PERCENTILE"]),
    )

    front_result = analyse_one_image(front, cfg)
    reduced_result = analyse_one_image(reduced, cfg)

    gaussian_rows = build_result_table(
        percent_values=cfg["PERCENT_VALUES"],
        fit_front=front_result["gaussian_fit"],
        fit_reduced=reduced_result["gaussian_fit"],
        front_pixel_size_um=cfg["FRONT_PIXEL_SIZE_UM"],
        reduced_pixel_size_um=cfg["REDUCED_PIXEL_SIZE_UM"],
        fit_type="gaussian",
    )

    flat_top_rows = build_result_table(
        percent_values=cfg["PERCENT_VALUES"],
        fit_front=front_result["flat_top_fit"],
        fit_reduced=reduced_result["flat_top_fit"],
        front_pixel_size_um=cfg["FRONT_PIXEL_SIZE_UM"],
        reduced_pixel_size_um=cfg["REDUCED_PIXEL_SIZE_UM"],
        fit_type="flat_top",
    )

    save_csv_like_json(output_dir / "gaussian_size_vs_percent.csv", gaussian_rows)
    save_csv_like_json(output_dir / "flat_top_size_vs_percent.csv", flat_top_rows)

    meta = {
        "front_image": str(cfg["FRONT_IMAGE"]),
        "reduced_image": str(cfg["REDUCED_IMAGE"]),
        "front_pixel_size_um": cfg["FRONT_PIXEL_SIZE_UM"],
        "reduced_pixel_size_um": cfg["REDUCED_PIXEL_SIZE_UM"],
        "gaussian_fit_front": front_result["gaussian_fit"],
        "gaussian_fit_reduced": reduced_result["gaussian_fit"],
        "flat_top_fit_front": front_result["flat_top_fit"],
        "flat_top_fit_reduced": reduced_result["flat_top_fit"],
        "percent_values": cfg["PERCENT_VALUES"],
        "note": "magnification = reduced_mean_size / front_mean_size",
    }
    with open(output_dir / "fit_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    front_unit = "px" if cfg["FRONT_PIXEL_SIZE_UM"] is None else "mm"
    reduced_unit = "px" if cfg["REDUCED_PIXEL_SIZE_UM"] is None else "mm"
    ylabel = f"Beam size [{front_unit}]"
    if front_unit != reduced_unit:
        ylabel = f"Beam size [{front_unit} / {reduced_unit}]"

    make_plot(
        rows=gaussian_rows,
        output_path=output_dir / "gaussian_size_vs_percent.png",
        title="Beam size vs. Intensitätsniveau – Gaussian Fit",
        ylabel=ylabel,
        front_label="Referenzbild / vorne",
        reduced_label="Verkleinertes Bild",
    )

    make_plot(
        rows=flat_top_rows,
        output_path=output_dir / "flat_top_size_vs_percent.png",
        title="Beam size vs. Intensitätsniveau – Flat-Top Fit (Super-Gaussian)",
        ylabel=ylabel,
        front_label="Referenzbild / vorne",
        reduced_label="Verkleinertes Bild",
    )

    # Zusätzliche Zusammenfassung auf der Konsole
    print("=" * 80)
    print("FIT-ANALYSE FERTIG")
    print("=" * 80)
    print(f"Ausgabeordner: {output_dir}")
    print("Gaussian Plot: ", output_dir / "gaussian_size_vs_percent.png")
    print("Flat-Top Plot:", output_dir / "flat_top_size_vs_percent.png")
    print("Gaussian CSV: ", output_dir / "gaussian_size_vs_percent.csv")
    print("Flat-Top CSV: ", output_dir / "flat_top_size_vs_percent.csv")

    # Kurze Magnification-Zusammenfassung
    g_mag = np.array([r["magnification"] for r in gaussian_rows], dtype=float)
    f_mag = np.array([r["magnification"] for r in flat_top_rows], dtype=float)
    print()
    print(f"Mittlere Vergrößerung aus Gaussian-Fit: {np.mean(g_mag):.4f}")
    print(f"Mittlere Vergrößerung aus Flat-Top-Fit: {np.mean(f_mag):.4f}")


if __name__ == "__main__":
    main()
