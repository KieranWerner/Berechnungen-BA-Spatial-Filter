import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

# ======== SETTINGS ========
DATA_FOLDER = r"C:\Users\User\Desktop\2D Beamprofiler free"

# Dateinamen + zugehörige Abstände in cm
file_info = [
  # falls "ende" nicht 0 cm ist, hier anpassen
    ("messung 53cm.tiff", 53),
    ("messung 200 cm.tiff", 200),
    ("messung 350cm.tiff", 350),
    ("messung 500cm.tiff", 500),
    ("messung 750cm.tiff", 750),
    ("messung ende.tiff", 1350), 
]

# Energiegrenzwerte
energy_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# ==========================

def load_tiff_image(path):
    img = Image.open(path)
    return np.array(img, dtype=float)

def compute_core_halo(I, energy_threshold=0.5):
    I = I.astype(float)
    I[I < 0] = 0

    total_intensity = I.sum()
    if total_intensity == 0:
        raise ValueError("Bild enthält keine Intensität.")

    # Schwerpunkt des gesamten Strahls
    y, x = np.indices(I.shape)
    x0 = (I * x).sum() / total_intensity
    y0 = (I * y).sum() / total_intensity

    # Abstand jedes Pixels zum Schwerpunkt
    r = np.sqrt((x - x0)**2 + (y - y0)**2)

    # Flatten und nach Radius sortieren
    r_flat = r.flatten()
    I_flat = I.flatten()

    idx = np.argsort(r_flat)
    r_sorted = r_flat[idx]
    I_sorted = I_flat[idx]

    # Kumulative Energie
    cum_energy = np.cumsum(I_sorted)
    cum_energy /= cum_energy[-1]

    # Radius, bei dem energy_threshold erreicht wird
    i_core = np.searchsorted(cum_energy, energy_threshold)
    i_core = min(i_core, len(r_sorted) - 1)
    r_core = r_sorted[i_core]

    # Core / Halo Masken
    core_mask = r <= r_core
    halo_mask = r > r_core

    return core_mask, halo_mask, r_core, (x0, y0)

def beam_radius(I, mask):
    y, x = np.indices(I.shape)
    I_masked = I * mask

    total = I_masked.sum()
    if total == 0:
        return 0.0

    x0 = (I_masked * x).sum() / total
    y0 = (I_masked * y).sum() / total

    r2 = (x - x0)**2 + (y - y0)**2
    return np.sqrt((I_masked * r2).sum() / total)

# ======== DATEIEN LADEN ========
distances = []
images = []

for fname, z in file_info:
    path = os.path.join(DATA_FOLDER, fname)

    if not os.path.exists(path):
        print(f"Datei nicht gefunden: {path}")
        continue

    try:
        I = load_tiff_image(path)
        images.append(I)
        distances.append(z)
        print(f"Geladen: {fname}")
    except Exception as e:
        print(f"Fehler beim Laden von {fname}: {e}")

if len(images) == 0:
    raise RuntimeError("Keine Bilder geladen. Bitte Dateipfade prüfen.")

# ======== AUSWERTUNG ========
core_results = {thr: [] for thr in energy_thresholds}
halo_results = {thr: [] for thr in energy_thresholds}

for I in images:
    for thr in energy_thresholds:
        core_mask, halo_mask, r_core, center = compute_core_halo(I, energy_threshold=thr)

        core_r = beam_radius(I, core_mask)
        halo_r = beam_radius(I, halo_mask)

        core_results[thr].append(core_r)
        halo_results[thr].append(halo_r)

# ======== PLOT: ALLES IN EINEM GRAPHEN ========
plt.figure(figsize=(12, 7))

for thr in energy_thresholds:
    percent = int(thr * 100)

    # Core = durchgezogen
    plt.plot(
        distances,
        core_results[thr],
        marker='o',
        linestyle='-',
        label=f"Core {percent}%"
    )

    # Halo = gestrichelt
    plt.plot(
        distances,
        halo_results[thr],
        marker='x',
        linestyle='--',
        label=f"Halo {percent}%"
    )

plt.xlabel("Propagation / Abstand (cm)")
plt.ylabel("Radius (Pixel)")
plt.title("Core- und Halo-Propagation für verschiedene Energiegrenzwerte")
plt.grid(True)
plt.legend(ncol=2, fontsize=8)
plt.tight_layout()
plt.show()