import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# USER SETTINGS
# =========================
FOLDER = os.path.expanduser("~/Desktop/focus scan 5 new")
PIXEL_SIZE_UM = None

CANDIDATE_PIXEL_OFFSETS = [9348, 13924, 20000, 25000]
IMG_SHAPE = (1024, 1024)

# =========================
# WCF READER
# =========================
def try_read_image_from_wcf(path):
    with open(path, "rb") as f:
        data = f.read()

    for offset in CANDIDATE_PIXEL_OFFSETS:
        nbytes = IMG_SHAPE[0] * IMG_SHAPE[1] * 2
        if offset + nbytes > len(data):
            continue

        arr = np.frombuffer(
            data[offset:offset+nbytes],
            dtype=np.uint16
        ).reshape(IMG_SHAPE)

        if arr.max() > 50 and arr.std() > 5:
            return arr.astype(float)

    raise RuntimeError(f"Could not parse image from {path}")

# =========================
# METRICS
# =========================
def centroid(img):
    y, x = np.indices(img.shape)
    total = img.sum()
    cx = (x * img).sum() / total
    cy = (y * img).sum() / total
    return cx, cy

def d4sigma(img):
    y, x = np.indices(img.shape)
    total = img.sum()
    cx, cy = centroid(img)

    sx2 = ((x - cx)**2 * img).sum() / total
    sy2 = ((y - cy)**2 * img).sum() / total

    dx = 4 * np.sqrt(sx2)
    dy = 4 * np.sqrt(sy2)

    deq = np.sqrt(dx * dy)
    return deq, dx, dy

def encircled_energy_diameter(img, fraction):
    cx, cy = centroid(img)
    y, x = np.indices(img.shape)

    r = np.sqrt((x-cx)**2 + (y-cy)**2).ravel()
    inten = img.ravel()

    idx = np.argsort(r)
    r_sorted = r[idx]
    i_sorted = inten[idx]

    cum = np.cumsum(i_sorted)
    target = fraction * cum[-1]

    j = np.searchsorted(cum, target)
    radius = r_sorted[min(j, len(r_sorted)-1)]

    return 2 * radius

def area_equivalent_diameter(img, threshold_fraction):
    thr = img.max() * threshold_fraction
    area = np.sum(img >= thr)
    return 2 * np.sqrt(area / np.pi)

def extract_z(filename):
    m = re.search(r'(\d+)', os.path.basename(filename))
    if m:
        return float(m.group(1))
    return np.nan

# =========================
# MAIN
# =========================
files = sorted(glob.glob(os.path.join(FOLDER, "*.wcf")), key=extract_z)

rows = []

for f in files:
    z = extract_z(f)
    img = try_read_image_from_wcf(f)

    deq, dx, dy = d4sigma(img)
    d86 = encircled_energy_diameter(img, 0.86)
    d95 = encircled_energy_diameter(img, 0.95)
    d50_area = area_equivalent_diameter(img, 0.50)
    d13_area = area_equivalent_diameter(img, 0.135)

    rows.append({
        "z_cm": z,
        "D4sigma_eq": deq,
        "D4sigma_x": dx,
        "D4sigma_y": dy,
        "D_EE86": d86,
        "D_EE95": d95,
        "D_area_50pct": d50_area,
        "D_area_13p5pct": d13_area
    })

df = pd.DataFrame(rows).sort_values("z_cm")

if PIXEL_SIZE_UM is not None:
    scale = PIXEL_SIZE_UM / 1000.0
    for col in df.columns:
        if col != "z_cm":
            df[col] *= scale
    ylabel = "Beam size [mm]"
else:
    ylabel = "Beam size [px]"

# save CSV
df.to_csv(os.path.join(FOLDER, "beam_size_vs_z.csv"), index=False)

# =========================
# PLOT 1: D4SIGMA ONLY
# =========================
plt.figure(figsize=(8, 5))

plt.plot(df["z_cm"], df["D4sigma_eq"], marker="o", label="D4sigma_eq")
plt.plot(df["z_cm"], df["D4sigma_x"], marker="o", label="D4sigma_x")
plt.plot(df["z_cm"], df["D4sigma_y"], marker="o", label="D4sigma_y")

plt.xlabel("z [cm]")
plt.ylabel(ylabel)
plt.title("D4sigma beam size")
plt.legend()
plt.grid(True)

plt.savefig(os.path.join(FOLDER, "beam_size_d4sigma.png"), dpi=150)

# =========================
# PLOT 2: OTHER METHODS
# =========================
plt.figure(figsize=(8, 5))

plt.plot(df["z_cm"], df["D_EE86"], marker="o", label="D_EE86")
plt.plot(df["z_cm"], df["D_EE95"], marker="o", label="D_EE95")
plt.plot(df["z_cm"], df["D_area_50pct"], marker="o", label="D_area_50pct")
plt.plot(df["z_cm"], df["D_area_13p5pct"], marker="o", label="D_area_13p5pct")

plt.xlabel("z [cm]")
plt.ylabel(ylabel)
plt.title("Other beam size definitions")
plt.legend()
plt.grid(True)

plt.savefig(os.path.join(FOLDER, "beam_size_other_methods.png"), dpi=150)

plt.show()
