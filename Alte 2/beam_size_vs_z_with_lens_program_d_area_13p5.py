import os
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Einstellungen
# ============================================================

folder = r"C:\Users\User\Desktop\Theory free propagation"
int_file = os.path.join(folder, "average_INT.csv")
pha_file = os.path.join(folder, "average_PHA.csv")

wavelength = 800e-9          # [m]
dx = 8.0e-6                  # [m]
dy = 8.0e-6                  # [m]

z_values = np.array([0.0, 0.53, 2.00, 3.50, 5.00, 7.50, 12.00])  # [m]

phase_in_waves = True        # False testen, falls nötig
pad_factor = 4               # 2 oder 4 ist sinnvoll
clip_background = True

# ============================================================
# Hilfsfunktionen
# ============================================================

def pad_center(arr, pad_factor=4):
    ny, nx = arr.shape
    new_ny = pad_factor * ny
    new_nx = pad_factor * nx

    out = np.zeros((new_ny, new_nx), dtype=arr.dtype)

    y0 = (new_ny - ny) // 2
    x0 = (new_nx - nx) // 2
    out[y0:y0+ny, x0:x0+nx] = arr
    return out, x0, y0, nx, ny


def crop_center(arr, x0, y0, nx, ny):
    return arr[y0:y0+ny, x0:x0+nx]


def angular_spectrum(U0, dx, dy, wavelength, z):
    ny, nx = U0.shape
    k = 2 * np.pi / wavelength

    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dy)
    FX, FY = np.meshgrid(fx, fy)

    root = 1 - (wavelength * FX)**2 - (wavelength * FY)**2
    H = np.exp(1j * k * z * np.sqrt(root + 0j))

    Uz = np.fft.ifft2(np.fft.fft2(U0) * H)
    return Uz


def beam_centroid_and_d4sigma(I, dx, dy):
    I = np.asarray(I, dtype=float)
    power = I.sum()
    if power <= 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    ny, nx = I.shape
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    X, Y = np.meshgrid(x, y)

    xc = np.sum(I * X) / power
    yc = np.sum(I * Y) / power

    sigma_x = np.sqrt(np.sum(I * (X - xc)**2) / power)
    sigma_y = np.sqrt(np.sum(I * (Y - yc)**2) / power)

    d4x = 4 * sigma_x
    d4y = 4 * sigma_y
    d4avg = 0.5 * (d4x + d4y)

    return xc, yc, d4x, d4y, d4avg


# ============================================================
# Daten laden
# ============================================================

I = np.loadtxt(int_file, delimiter=",")
pha = np.loadtxt(pha_file, delimiter=",")

if I.shape != pha.shape:
    raise ValueError(f"Shape mismatch: {I.shape} vs {pha.shape}")

I = np.nan_to_num(I, nan=0.0, posinf=0.0, neginf=0.0)
pha = np.nan_to_num(pha, nan=0.0, posinf=0.0, neginf=0.0)

if clip_background:
    # einfachen Hintergrund entfernen
    bg = np.percentile(I, 5)
    I = I - bg
    I[I < 0] = 0

# normieren
if I.max() > 0:
    I = I / I.max()

if phase_in_waves:
    phase_rad = 2 * np.pi * pha
else:
    phase_rad = pha

U0 = np.sqrt(I) * np.exp(1j * phase_rad)

# Padding
U0_pad, x0, y0, nx0, ny0 = pad_center(U0, pad_factor=pad_factor)

# ============================================================
# z-Scan
# ============================================================

d4x_list = []
d4y_list = []
d4avg_list = []

for z in z_values:
    Uz_pad = angular_spectrum(U0_pad, dx, dy, wavelength, z)
    Iz_pad = np.abs(Uz_pad)**2

    # optional: nur Originalfeld zurückcropen
    Iz = crop_center(Iz_pad, x0, y0, nx0, ny0)

    # Hintergrundclip nach Propagation
    Iz[Iz < 0] = 0

    _, _, d4x, d4y, d4avg = beam_centroid_and_d4sigma(Iz, dx, dy)

    d4x_list.append(d4x * 1e3)     # mm
    d4y_list.append(d4y * 1e3)
    d4avg_list.append(d4avg * 1e3)

# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(9, 5))
plt.plot(z_values * 100, d4x_list, "o-", label="D4σ_x")
plt.plot(z_values * 100, d4y_list, "o-", label="D4σ_y")
plt.plot(z_values * 100, d4avg_list, "o-", linewidth=2, label="D4σ_avg")
plt.xlabel("z [cm]")
plt.ylabel("Beam size [mm]")
plt.title("Beam size vs propagation distance")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

print("z [cm]   D4σ_x [mm]   D4σ_y [mm]   D4σ_avg [mm]")
for z, a, b, c in zip(z_values * 100, d4x_list, d4y_list, d4avg_list):
    print(f"{z:6.1f}   {a:10.4f}   {b:10.4f}   {c:12.4f}")