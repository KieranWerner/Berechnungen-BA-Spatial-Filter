import os
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread

# =========================
# PFAD-EINSTELLUNGEN
# =========================
base_path = r"C:\Users\User\Desktop\Vergroesserung"

input_file = os.path.join(base_path, "original groesse.tiff")
output_file = os.path.join(base_path, "average_INT.csv")

# =========================
# PIXELGRÖSSEN
# =========================
# original groesse: 5.5 µm / Pixel
pixel_in_value = 11
pixel_in_unit = "um"

# average_INT: 41.683 mm / Pixel
pixel_out_value = 24
pixel_out_unit = "um"

# =========================
# HILFSFUNKTIONEN
# =========================
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
        # versucht zuerst Komma-getrennt, dann Semikolon
        try:
            img = np.loadtxt(path, delimiter=",")
        except:
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
    return np.sqrt((x - cx)**2 + (y - cy)**2)

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
    y, x = np.indices(img.shape)

    x_mean = np.sum(x * img) / total
    y_mean = np.sum(y * img) / total

    sigma_x = np.sqrt(np.sum(((x - x_mean) ** 2) * img) / total)
    sigma_y = np.sqrt(np.sum(((y - y_mean) ** 2) * img) / total)

    # ISO-artige D4σ-Durchmesser getrennt in x/y und gemittelt
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

    results = {
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
    return results

def format_si(value_m):
    if np.isnan(value_m):
        return "nan"
    abs_val = abs(value_m)
    if abs_val >= 1e-3:
        return f"{value_m*1e3:.6f} mm"
    elif abs_val >= 1e-6:
        return f"{value_m*1e6:.6f} µm"
    elif abs_val >= 1e-9:
        return f"{value_m*1e9:.6f} nm"
    else:
        return f"{value_m:.6e} m"

# =========================
# DATEN LADEN
# =========================
img_in = load_image(input_file)
img_out = load_image(output_file)

pixel_in_m = unit_to_meter(pixel_in_value, pixel_in_unit)
pixel_out_m = unit_to_meter(pixel_out_value, pixel_out_unit)

# =========================
# AUSWERTUNG
# =========================
res_in = compute_methods(img_in, pixel_in_m)
res_out = compute_methods(img_out, pixel_out_m)

methods = [
    ("FWHM", "FWHM"),
    ("1/e²", "1/e^2"),
    ("D4σ", "D4sigma_mean"),
    ("10%-Threshold", "Threshold_10%"),
    ("50%-Encircled-Energy", "EncircledEnergy_50%"),
]

print("\n================ ERGEBNISSE ================\n")
print(f"Eingangsdatei : {input_file}")
print(f"Ausgangsdatei : {output_file}")
print(f"Pixelgröße in : {pixel_in_value} {pixel_in_unit}/Pixel")
print(f"Pixelgröße out: {pixel_out_value} {pixel_out_unit}/Pixel\n")

for label, key in methods:
    d_in = res_in[key]
    d_out = res_out[key]
    magnification = d_out / d_in if d_in > 0 else np.nan

    print(f"{label}")
    print(f"  D_in  = {format_si(d_in)}")
    print(f"  D_out = {format_si(d_out)}")
    print(f"  M     = {magnification:.6f}")


print("Zusatz für D4σ:")
print(f"  D4σ_x in  = {format_si(res_in['D4sigma_x'])}")
print(f"  D4σ_y in  = {format_si(res_in['D4sigma_y'])}")
print(f"  D4σ_x out = {format_si(res_out['D4sigma_x'])}")
print(f"  D4σ_y out = {format_si(res_out['D4sigma_y'])}")

# =========================
# OPTIONAL: BILDER ANZEIGEN
# =========================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(img_in, cmap="viridis")
axes[0].set_title("Input: original groesse")
axes[0].scatter(res_in["center_x_px"], res_in["center_y_px"], s=30, marker="x")
axes[0].set_aspect("equal")

axes[1].imshow(img_out, cmap="viridis")
axes[1].set_title("Output: average_INT")
axes[1].scatter(res_out["center_x_px"], res_out["center_y_px"], s=30, marker="x")
axes[1].set_aspect("equal")

plt.tight_layout()
plt.show()