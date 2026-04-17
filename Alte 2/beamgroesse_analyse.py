from pathlib import Path
import re
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# =========================
# ORDNER
# =========================
folder = Path(r"C:\Users\User\Desktop\Eingabe2")

# =========================
# TIFF laden
# =========================
def load_image(path):
    img = Image.open(path)
    arr = np.array(img, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    return arr

# =========================
# Hintergrund entfernen
# =========================
def preprocess(image):
    border = 20
    bg = np.median(np.concatenate([
        image[:border, :].ravel(),
        image[-border:, :].ravel(),
        image[:, :border].ravel(),
        image[:, -border:].ravel()
    ]))
    image = image - bg
    image[image < 0] = 0
    return image

# =========================
# Schwerpunkt
# =========================
def centroid(image):
    y, x = np.indices(image.shape)
    total = image.sum()
    cx = (x * image).sum() / total
    cy = (y * image).sum() / total
    return cx, cy

# =========================
# D4sigma
# =========================
def beam_size(image):
    cx, cy = centroid(image)
    y, x = np.indices(image.shape)

    sigma_x2 = ((x - cx)**2 * image).sum() / image.sum()
    sigma_y2 = ((y - cy)**2 * image).sum() / image.sum()

    d4x = 4 * math.sqrt(sigma_x2)
    d4y = 4 * math.sqrt(sigma_y2)

    return 0.5 * (d4x + d4y)

# =========================
# Distanz aus Dateiname
# =========================
def extract_distance(name):
    m = re.search(r'(\d+)\s*cm', name.lower())
    if m:
        return float(m.group(1))
    return None

# =========================
# Dateien sammeln
# =========================
files = sorted(folder.glob("*.tiff"))

distances = []
sizes = []

for file in files:
    img = load_image(file)
    img = preprocess(img)

    size = beam_size(img)
    dist = extract_distance(file.name)

    if dist is not None:
        distances.append(dist)
        sizes.append(size)

# =========================
# Sortieren
# =========================
order = np.argsort(distances)
distances = np.array(distances)[order]
sizes = np.array(sizes)[order]

# =========================
# Plot speichern
# =========================
plt.figure(figsize=(8,5))
plt.plot(distances, sizes, marker='o')

plt.xlabel("Distanz [cm]")
plt.ylabel("Strahlgröße D4σ [Pixel]")
plt.title("Strahlgröße über Distanz")
plt.grid(True)

plot_path = folder / "strahlgroesse_plot.png"
plt.savefig(plot_path, dpi=150)

print("Plot gespeichert:", plot_path)
