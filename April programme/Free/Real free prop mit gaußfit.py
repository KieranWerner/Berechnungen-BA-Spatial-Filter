import numpy as np
import tifffile as tiff
import csv
import matplotlib.pyplot as plt
from pathlib import Path

# ================= CONFIG =================

BASE_PATH = Path(r"C:\Users\User\Desktop\Free beam size plot")

FILES = [
    (BASE_PATH / "messung 53cm.tiff", 0.53),
    (BASE_PATH / "messung 200 cm.tiff", 2.00),
    (BASE_PATH / "messung 350cm.tiff", 3.50),
    (BASE_PATH / "messung 500cm.tiff", 5.00),
    (BASE_PATH / "messung 750cm.tiff", 7.50),
    (BASE_PATH / "messung ende.tiff", 10.00),
]

OUTPUT_CSV = BASE_PATH / "real_propagation_gaussfit.csv"
OUTPUT_PNG = BASE_PATH / "real_propagation_gaussfit.png"

plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "legend.fontsize": 16,
})

# =========================================

def load_tiff(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return tiff.imread(path).astype(float)

# ================= GAUSS FIT =================

def gaussian_2d(coords, offset, amplitude, x0, y0, sigma_x, sigma_y):
    x, y = coords
    return offset + amplitude * np.exp(
        -2 * ((x - x0)**2 / sigma_x**2 + (y - y0)**2 / sigma_y**2)
    )

def fit_gaussian(I):
    from scipy.optimize import curve_fit

    I = np.clip(I, 0, None)

    ny, nx = I.shape
    y = np.arange(ny)
    x = np.arange(nx)
    X, Y = np.meshgrid(x, y)

    # initial guesses
    offset0 = np.percentile(I, 5)
    amplitude0 = np.max(I) - offset0

    P = I.sum()
    x0 = (I*X).sum()/P
    y0 = (I*Y).sum()/P

    sigma_x = np.sqrt(((X-x0)**2 * I).sum()/P)
    sigma_y = np.sqrt(((Y-y0)**2 * I).sum()/P)

    p0 = [offset0, amplitude0, x0, y0, sigma_x, sigma_y]

    try:
        popt, _ = curve_fit(
            gaussian_2d,
            (X.ravel(), Y.ravel()),
            I.ravel(),
            p0=p0,
            maxfev=20000
        )
    except Exception:
        popt = p0  # fallback

    _, _, x0, y0, sigma_x, sigma_y = popt

    # 1/e² diameter
    diameter_x = 2 * sigma_x
    diameter_y = 2 * sigma_y
    diameter_mean = 0.5 * (diameter_x + diameter_y)

    return diameter_mean

# ================= MAIN =================

results = []

for file, z in FILES:
    print(f"Processing: {file.name}")

    I = load_tiff(file)
    size = fit_gaussian(I)

    results.append({
        "z_m": z,
        "Gauss_diameter_1e2_px": size
    })

# sort
results = sorted(results, key=lambda x: x["z_m"])

# ================= SAVE CSV =================

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["z_m", "Gauss_diameter_1e2_px"])
    writer.writeheader()
    writer.writerows(results)

print(f"CSV saved: {OUTPUT_CSV}")

# ================= PLOT =================

z_vals = [r["z_m"] for r in results]
sizes = [r["Gauss_diameter_1e2_px"] for r in results]

plt.figure(figsize=(10,6))

plt.plot(
    z_vals,
    sizes,
    "o-",
    linewidth=2,
    markersize=8,
    label="Measurement (Gaussian fit 1/e²)"
)

plt.xlabel("Propagation distance z [m]")
plt.ylabel("Beam diameter (1/e²) [pixels]")
plt.title("Measured beam propagation (Gaussian fit)")

plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=300)
plt.show()

print(f"PNG saved: {OUTPUT_PNG}")