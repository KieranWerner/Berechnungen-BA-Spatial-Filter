import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ============================================================
# AUSWAHL: Welche Definitionen sollen geplottet werden?
# ============================================================
PLOT_D4SIGMA = True
PLOT_EE50 = True
PLOT_EE86 = True
PLOT_DEQ50 = True
PLOT_DEQ13_8 = True
PLOT_FWHM = True

# ============================================================
# EINSTELLUNGEN
# ============================================================
# Pixelgröße in mm/Pixel
PIXEL_SIZE_MM = 0.011
UNIT_LABEL = "mm"

# Hintergrund wird aus dem Rand des Bildes geschätzt.
BORDER_WIDTH_PX = 80
BACKGROUND_PERCENTILE = 50  # Median des Randes

# Optional: negative Restwerte nach Hintergrundabzug auf 0 setzen
CLIP_NEGATIVE_TO_ZERO = True

# Dateien und ihre Entfernungen in Metern
MEASUREMENTS = [
    (0.53, "messung 53cm referenz.tiff"),
    (2.00, "messung 200 cm.tiff"),
    (3.50, "messung 350cm.tiff"),
    (5.00, "messung 500cm.tiff"),
    (7.50, "messung 750cm.tiff"),
    (12.0, "messung ende.tiff"),
]

# Ordner der Bilder
BASE_DIR = Path("C:/Users/User/Desktop/Real free propagation")


# ============================================================
# HILFSFUNKTIONEN
# ============================================================
def load_image(path: Path) -> np.ndarray:
    """Lädt ein TIFF-Bild als float64-Array."""
    return np.array(Image.open(path), dtype=np.float64)


def estimate_background(img: np.ndarray, border_width: int = 80, percentile: float = 50) -> float:
    """Schätzt den Hintergrund aus dem Bildrand."""
    bw = min(border_width, img.shape[0] // 4, img.shape[1] // 4)
    top = img[:bw, :]
    bottom = img[-bw:, :]
    left = img[:, :bw]
    right = img[:, -bw:]
    border_pixels = np.concatenate([
        top.ravel(), bottom.ravel(), left.ravel(), right.ravel()
    ])
    return float(np.percentile(border_pixels, percentile))


def preprocess(img: np.ndarray, subtract_background: bool) -> np.ndarray:
    """Optionaler Hintergrundabzug."""
    if not subtract_background:
        return img.copy()

    bg = estimate_background(img, BORDER_WIDTH_PX, BACKGROUND_PERCENTILE)
    corrected = img - bg

    if CLIP_NEGATIVE_TO_ZERO:
        corrected = np.clip(corrected, 0, None)

    return corrected


def centroid(img: np.ndarray) -> tuple[float, float]:
    """Intensitätsgewichteter Schwerpunkt (x, y)."""
    total = img.sum()
    if total <= 0:
        raise ValueError("Bild enthält keine positive Gesamtintensität.")
    y, x = np.indices(img.shape)
    cx = float((img * x).sum() / total)
    cy = float((img * y).sum() / total)
    return cx, cy


def d4sigma_diameter(img: np.ndarray) -> float:
    """
    D4σ-Durchmesser als geometrisches Mittel aus x- und y-Richtung:
    Dx = 4 * sigma_x, Dy = 4 * sigma_y, D = sqrt(Dx * Dy)
    """
    total = img.sum()
    if total <= 0:
        return np.nan

    y, x = np.indices(img.shape)
    cx, cy = centroid(img)
    sigma_x = math.sqrt(float((img * (x - cx) ** 2).sum() / total))
    sigma_y = math.sqrt(float((img * (y - cy) ** 2).sum() / total))

    dx = 4.0 * sigma_x
    dy = 4.0 * sigma_y
    return math.sqrt(dx * dy)


def encircled_energy_diameter(img: np.ndarray, fraction: float) -> float:
    """
    Durchmesser des Kreises um den Schwerpunkt, der 'fraction' der Gesamtenergie enthält.
    """
    total = img.sum()
    if total <= 0:
        return np.nan

    cx, cy = centroid(img)
    y, x = np.indices(img.shape)
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).ravel()
    inten = img.ravel()

    order = np.argsort(r)
    r_sorted = r[order]
    i_sorted = inten[order]

    cum = np.cumsum(i_sorted)
    target = fraction * total

    idx = np.searchsorted(cum, target)
    idx = min(idx, len(r_sorted) - 1)
    radius = float(r_sorted[idx])

    return 2.0 * radius


def equivalent_diameter_from_threshold(img: np.ndarray, rel_threshold: float) -> float:
    """
    Äquivalenter Durchmesser aus der Fläche aller Pixel mit
    I >= rel_threshold * I_max.
    """
    peak = img.max()
    if peak <= 0:
        return np.nan

    threshold = rel_threshold * peak
    mask = img >= threshold
    area_px = float(mask.sum())

    if area_px <= 0:
        return np.nan

    return math.sqrt(4.0 * area_px / math.pi)


def fwhm_diameter(img: np.ndarray) -> float:
    """
    FWHM als äquivalenter Durchmesser der Fläche oberhalb von 50 % des Peaks.
    Für 2D-Bilder ist das praktisch identisch zur Deq50-Definition.
    """
    return equivalent_diameter_from_threshold(img, 0.50)


def px_to_unit(value_px: float) -> float:
    return value_px * PIXEL_SIZE_MM


def evaluate_measurements(subtract_background: bool) -> tuple[list[float], dict[str, list[float]]]:
    """Wertet alle Messungen für eine gewählte Vorverarbeitung aus."""
    distances_m = []
    curves = {
        "D4sigma": [],
        "EE50": [],
        "EE86": [],
        "Deq50": [],
        "Deq13.8": [],
        "FWHM": [],
    }

    mode_text = "MIT Hintergrundabzug" if subtract_background else "OHNE Hintergrundabzug"
    print("\n" + "=" * 80)
    print(f"Auswertung {mode_text}")
    print("=" * 80)

    for distance_m, filename in MEASUREMENTS:
        path = BASE_DIR / filename
        if not path.exists():
            print(f"WARNUNG: Datei nicht gefunden: {path}")
            continue

        raw = load_image(path)
        img = preprocess(raw, subtract_background=subtract_background)

        distances_m.append(distance_m)

        d4 = px_to_unit(d4sigma_diameter(img))
        ee50 = px_to_unit(encircled_energy_diameter(img, 0.50))
        ee86 = px_to_unit(encircled_energy_diameter(img, 0.86))
        deq50 = px_to_unit(equivalent_diameter_from_threshold(img, 0.50))
        deq138 = px_to_unit(equivalent_diameter_from_threshold(img, 0.138))
        fwhm = px_to_unit(fwhm_diameter(img))

        curves["D4sigma"].append(d4)
        curves["EE50"].append(ee50)
        curves["EE86"].append(ee86)
        curves["Deq50"].append(deq50)
        curves["Deq13.8"].append(deq138)
        curves["FWHM"].append(fwhm)

        print(
            f"{filename:28s} | Entfernung = {distance_m:5.2f} m | "
            f"D4σ = {d4:8.3f} {UNIT_LABEL}, "
            f"EE50 = {ee50:8.3f} {UNIT_LABEL}, "
            f"EE86 = {ee86:8.3f} {UNIT_LABEL}, "
            f"Deq50 = {deq50:8.3f} {UNIT_LABEL}, "
            f"Deq13.8 = {deq138:8.3f} {UNIT_LABEL}, "
            f"FWHM = {fwhm:8.3f} {UNIT_LABEL}"
        )

    if not distances_m:
        raise RuntimeError("Keine gültigen Messdateien gefunden.")

    return distances_m, curves


def plot_curves(distances_m: list[float], curves: dict[str, list[float]], title: str, out_path: Path) -> None:
    """Plottet die ausgewählten Kurven und speichert den Plot."""
    plt.figure(figsize=(10, 6))

    if PLOT_D4SIGMA:
        plt.plot(distances_m, curves["D4sigma"], marker="o", label="D4sigma")
    if PLOT_EE50:
        plt.plot(distances_m, curves["EE50"], marker="o", label="EE50")
    if PLOT_EE86:
        plt.plot(distances_m, curves["EE86"], marker="o", label="EE86")
    if PLOT_DEQ50:
        plt.plot(distances_m, curves["Deq50"], marker="o", label="Deq50")
    if PLOT_DEQ13_8:
        plt.plot(distances_m, curves["Deq13.8"], marker="o", label="Deq13.8")
    if PLOT_FWHM:
        plt.plot(distances_m, curves["FWHM"], marker="o", label="FWHM")

    plt.xlabel("Entfernung [m]")
    plt.ylabel(f"Beam Size [{UNIT_LABEL}]")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.show()


# ============================================================
# HAUPTPROGRAMM
# ============================================================
def main() -> None:
    # 1) Mit Hintergrundabzug
    distances_bg, curves_bg = evaluate_measurements(subtract_background=True)
    out_path_bg = BASE_DIR / "beam_size_plot_mit_hintergrundabzug.png"
    plot_curves(
        distances_bg,
        curves_bg,
        "Beam Size über Entfernung (mit Hintergrundabzug)",
        out_path_bg,
    )

    # 2) Ohne Hintergrundabzug
    distances_raw, curves_raw = evaluate_measurements(subtract_background=False)
    out_path_raw = BASE_DIR / "beam_size_plot_ohne_hintergrundabzug.png"
    plot_curves(
        distances_raw,
        curves_raw,
        "Beam Size über Entfernung (ohne Hintergrundabzug)",
        out_path_raw,
    )

    print("\nPlots gespeichert unter:")
    print(f"  {out_path_bg}")
    print(f"  {out_path_raw}")
    print("\nHinweise:")
    print("- FWHM wurde als äquivalenter Durchmesser der Fläche oberhalb von 50 % des Peak-Signals berechnet.")
    print("- Dadurch ist FWHM in dieser Implementierung numerisch identisch zu Deq50.")
    print("- Falls du stattdessen eine profilbasierte FWHM-Definition entlang x/y möchtest, kann man das separat ergänzen.")


if __name__ == "__main__":
    main()