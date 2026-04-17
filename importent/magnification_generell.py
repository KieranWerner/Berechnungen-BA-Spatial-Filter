import os
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread

# =========================================
# PFAD-EINSTELLUNGEN
# =========================================
base_path = r"C:\Users\User\Desktop\Vergroesserung"

original_file = os.path.join(base_path, "original groesse.tiff")
average_file = os.path.join(base_path, "average_INT.csv")
reference53_file = os.path.join(base_path, "messung 53cm referenz.tiff")

# =========================================
# PIXELGRÖSSEN
# =========================================
# original groesse.tiff
pixel_original_value = 11
pixel_original_unit = "um"

# average_INT.csv
pixel_average_value = 24
pixel_average_unit = "um"

# messung 53cm referenz.tiff
pixel_reference53_value = 11
pixel_reference53_unit = "um"

# =========================================
# HILFSFUNKTIONEN
# =========================================
def unit_to_meter(value, unit):
    unit = unit.lower()
    factors = {
        "m": 1.0,
        "mm": 1e-3,
        "um": 1e-6,
        "µm": 1e-6,
        "nm": 1e-9
    }
    if unit not in factors:
        raise ValueError(f"Unbekannte Einheit: {unit}")
    return value * factors[unit]

def load_image(path):
    ext = os.path.splitext(path)[1].lower()

    if ext in [".tif", ".tiff"]:
        img = imread(path)
    elif ext == ".csv":
        try:
            img = np.loadtxt(path, delimiter=",")
        except Exception:
            img = np.loadtxt(path, delimiter=";")
    else:
        raise ValueError(f"Nicht unterstütztes Dateiformat: {ext}")

    if img.ndim > 2:
        img = np.mean(img, axis=0)

    img = img.astype(float)
    img = img - np.nanmin(img)
    img[np.isnan(img)] = 0.0
    return img

def beam_center(img):
    total = np.sum(img)
    if total <= 0:
        raise ValueError("Bild enthält keine positive Intensität.")
    y, x = np.indices(img.shape)
    cx = np.sum(x * img) / total
    cy = np.sum(y * img) / total
    return cx, cy

def radius_map(shape, cx, cy):
    y, x = np.indices(shape)
    return np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

def diameter_fwhm(img, r):
    max_intensity = np.max(img)
    threshold = 0.5 * max_intensity
    mask = img >= threshold
    if not np.any(mask):
        return np.nan
    return 2.0 * np.max(r[mask])

def diameter_1e2(img, r):
    max_intensity = np.max(img)
    threshold = max_intensity / np.exp(2)
    mask = img >= threshold
    if not np.any(mask):
        return np.nan
    return 2.0 * np.max(r[mask])

def diameter_d4sigma(img):
    total = np.sum(img)
    if total <= 0:
        return np.nan, np.nan, np.nan

    y, x = np.indices(img.shape)

    x_mean = np.sum(x * img) / total
    y_mean = np.sum(y * img) / total

    sigma_x = np.sqrt(np.sum(((x - x_mean) ** 2) * img) / total)
    sigma_y = np.sqrt(np.sum(((y - y_mean) ** 2) * img) / total)

    d_x = 4.0 * sigma_x
    d_y = 4.0 * sigma_y
    d_mean = 0.5 * (d_x + d_y)

    return d_mean, d_x, d_y

def diameter_threshold(img, r, frac=0.1):
    max_intensity = np.max(img)
    threshold = frac * max_intensity
    mask = img >= threshold
    if not np.any(mask):
        return np.nan
    return 2.0 * np.max(r[mask])

def diameter_encircled_energy(img, r, fraction=0.5):
    r_flat = r.ravel()
    i_flat = img.ravel()

    idx = np.argsort(r_flat)
    r_sorted = r_flat[idx]
    i_sorted = i_flat[idx]

    cum = np.cumsum(i_sorted)
    if cum[-1] <= 0:
        return np.nan
    cum = cum / cum[-1]

    pos = np.searchsorted(cum, fraction)
    pos = min(pos, len(r_sorted) - 1)
    return 2.0 * r_sorted[pos]

def compute_methods(img, pixel_size_m):
    cx, cy = beam_center(img)
    r = radius_map(img.shape, cx, cy)

    d_fwhm_px = diameter_fwhm(img, r)
    d_1e2_px = diameter_1e2(img, r)
    d_10_px = diameter_threshold(img, r, frac=0.1)
    d_50ee_px = diameter_encircled_energy(img, r, fraction=0.5)
    d_d4sigma_px, d_d4sigma_x_px, d_d4sigma_y_px = diameter_d4sigma(img)

    return {
        "FWHM": d_fwhm_px * pixel_size_m,
        "1/e^2": d_1e2_px * pixel_size_m,
        "D4sigma_mean": d_d4sigma_px * pixel_size_m,
        "D4sigma_x": d_d4sigma_x_px * pixel_size_m,
        "D4sigma_y": d_d4sigma_y_px * pixel_size_m,
        "Threshold_10%": d_10_px * pixel_size_m,
        "EncircledEnergy_50%": d_50ee_px * pixel_size_m,
        "center_x_px": cx,
        "center_y_px": cy,
    }

def format_si(value_m):
    if np.isnan(value_m):
        return "nan"
    abs_val = abs(value_m)
    if abs_val >= 1e-3:
        return f"{value_m * 1e3:.6f} mm"
    elif abs_val >= 1e-6:
        return f"{value_m * 1e6:.6f} µm"
    elif abs_val >= 1e-9:
        return f"{value_m * 1e9:.6f} nm"
    else:
        return f"{value_m:.6e} m"

def print_comparison(title, ref_name, cmp_name, res_ref, res_cmp):
    methods = [
        ("FWHM", "FWHM"),
        ("1/e²", "1/e^2"),
        ("D4σ", "D4sigma_mean"),
        ("10%-Threshold", "Threshold_10%"),
        ("50%-Encircled-Energy", "EncircledEnergy_50%"),
    ]

    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)
    print(f"Referenz : {ref_name}")
    print(f"Vergleich: {cmp_name}\n")

    for label, key in methods:
        d_ref = res_ref[key]
        d_cmp = res_cmp[key]
        vergroesserung = d_cmp / d_ref if d_ref > 0 else np.nan

        print(f"{label}")
        print(f"  D_ref = {format_si(d_ref)}")
        print(f"  D_cmp = {format_si(d_cmp)}")
        print(f"  Vergrößerungsfaktor = {vergroesserung:.6f}")
        print()

    print("Zusatz für D4σ:")
    print(f"  D4σ_x ref = {format_si(res_ref['D4sigma_x'])}")
    print(f"  D4σ_y ref = {format_si(res_ref['D4sigma_y'])}")
    print(f"  D4σ_x cmp = {format_si(res_cmp['D4sigma_x'])}")
    print(f"  D4σ_y cmp = {format_si(res_cmp['D4sigma_y'])}")
    print()

# =========================================
# DATEN LADEN
# =========================================
img_original = load_image(original_file)
img_average = load_image(average_file)
img_reference53 = load_image(reference53_file)

pixel_original_m = unit_to_meter(pixel_original_value, pixel_original_unit)
pixel_average_m = unit_to_meter(pixel_average_value, pixel_average_unit)
pixel_reference53_m = unit_to_meter(pixel_reference53_value, pixel_reference53_unit)

# =========================================
# AUSWERTUNG
# =========================================
res_original = compute_methods(img_original, pixel_original_m)
res_average = compute_methods(img_average, pixel_average_m)
res_reference53 = compute_methods(img_reference53, pixel_reference53_m)

print("\n================ GESAMTAUSWERTUNG ================\n")
print(f"Originaldatei        : {original_file}")
print(f"Average_INT-Datei    : {average_file}")
print(f"53cm-Referenzdatei   : {reference53_file}\n")

print(f"Pixelgröße Original      : {pixel_original_value} {pixel_original_unit}/Pixel")
print(f"Pixelgröße Average_INT   : {pixel_average_value} {pixel_average_unit}/Pixel")
print(f"Pixelgröße Referenz 53cm : {pixel_reference53_value} {pixel_reference53_unit}/Pixel")

# Vergleich 1: original groesse vs average_INT
print_comparison(
    title="Vergleich 1: original groesse vs average_INT",
    ref_name="original groesse",
    cmp_name="average_INT",
    res_ref=res_original,
    res_cmp=res_average
)

# Vergleich 2: original groesse vs messung 53cm referenz
print_comparison(
    title="Vergleich 2: original groesse vs messung 53cm referenz",
    ref_name="original groesse",
    cmp_name="messung 53cm referenz",
    res_ref=res_original,
    res_cmp=res_reference53
)

# =========================================
# OPTIONAL: ANZEIGE
# =========================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].imshow(img_original, cmap="viridis")
axes[0].set_title("original groesse")
axes[0].scatter(res_original["center_x_px"], res_original["center_y_px"], s=30, marker="x")
axes[0].set_aspect("equal")

axes[1].imshow(img_average, cmap="viridis")
axes[1].set_title("average_INT")
axes[1].scatter(res_average["center_x_px"], res_average["center_y_px"], s=30, marker="x")
axes[1].set_aspect("equal")

axes[2].imshow(img_reference53, cmap="viridis")
axes[2].set_title("messung 53cm referenz")
axes[2].scatter(res_reference53["center_x_px"], res_reference53["center_y_px"], s=30, marker="x")
axes[2].set_aspect("equal")

plt.tight_layout()
plt.show()