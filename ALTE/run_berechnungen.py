import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import struct
import os

# WICHTIG: Der Dateiname muss exakt so heißen wie links im Ordner
FILENAME = "BeamProfile1.wcf"


def main():
    """Hauptfunktion: Einlesen, mitteln, croppen und Super-Gauß-Fit durchführen."""
    # 1. Sicherstellen, dass die Datei da ist
    if not os.path.exists(FILENAME):
        print(f"❌ FEHLER: Die Datei '{FILENAME}' wurde nicht gefunden!")
        print("Lösung: Bitte ziehen Sie die Datei links in das Dateimenü (Ordner-Symbol).")
        return

    print(f"✅ Datei '{FILENAME}' gefunden. Lese Daten...")

    # 2. Daten einlesen
    try:
        frames = read_wcf_frames(FILENAME)
    except Exception as e:
        print(f"Fehler beim Lesen der Datei: {e}")
        return

    if len(frames) == 0:
        print("Fehler: Die Datei scheint leer oder beschädigt zu sein.")
        return

    print(f"Anzahl Frames: {len(frames)}, Frame-Shape: {frames[0].shape}")

    # 3. Mitteln (Average) über alle Frames
    avg_image = np.mean(frames, axis=0)

    # --- Fit Vorbereitung (Cropping) ---
    # Wir schneiden das Bild auf den Strahl zu, damit der Fit stabil läuft
    h, w = avg_image.shape
    y_c, x_c = np.unravel_index(np.argmax(avg_image), avg_image.shape)

    crop_sz = 400
    x_min, x_max = max(0, x_c - crop_sz//2), min(w, x_c + crop_sz//2)
    y_min, y_max = max(0, y_c - crop_sz//2), min(h, y_c + crop_sz//2)

    img_crop = avg_image[y_min:y_max, x_min:x_max]
    hc, wc = img_crop.shape

    # Sanity-Checks
    if hc < 8 or wc < 8:
        print("Crop zu klein für zuverlässigen Fit (mind. 8x8).")
        return
    if hc * wc > 5_000_000:
        print("Bild zu groß für Fit (hc*wc > 5e6). Verkleinere Crop oder downsample.")
        return

    # Gitter für Fit
    X, Y = np.meshgrid(np.arange(wc), np.arange(hc))
    xdata = np.vstack((X.ravel(), Y.ravel()))
    ydata = img_crop.ravel()

    # Bessere Startwerte mittels Momente
    offset0 = np.min(img_crop)
    amp0 = np.max(img_crop) - offset0
    total = img_crop.sum()
    if total > 0:
        x_idx, y_idx = np.meshgrid(np.arange(wc), np.arange(hc))
        x0 = (x_idx * img_crop).sum() / total
        y0 = (y_idx * img_crop).sum() / total
        # Varianz (second moments)
        sx2 = (( (x_idx - x0)**2 * img_crop ).sum() / total)
        sy2 = (( (y_idx - y0)**2 * img_crop ).sum() / total)
        sx = np.sqrt(max(sx2, 1e-6))
        sy = np.sqrt(max(sy2, 1e-6))
        # start widths not too small or too large
        wx0 = max(1.0, sx)
        wy0 = max(1.0, sy)
    else:
        x0, y0 = wc/2.0, hc/2.0
        wx0, wy0 = max(1.0, wc/8.0), max(1.0, hc/8.0)

    p0 = [offset0, amp0, x0, y0, wx0, wy0, 2.0]
    bounds = ([0, 0, 0, 0, 0.5, 0.5, 1.0], [65535, 65535, wc, hc, wc, hc, 30.0])

    print("Führe Super-Gauß-Fit durch...")
    try:
        popt, pcov = curve_fit(super_gaussian_2d, xdata, ydata, p0=p0, bounds=bounds, maxfev=20000)
        n_res = popt[6]

        # Prüfe Kovarianzmatrix
        if np.isnan(pcov).any():
            print("Warnung: Kovarianzmatrix enthält NaNs. Unsichere Fehlerabschätzung.")

        # --- Plotten ---
        fit_model = super_gaussian_2d(xdata, *popt).reshape(hc, wc)

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        # Original
        im1 = ax[0].imshow(img_crop, cmap='jet')
        ax[0].set_title(f"Messdaten (Gemittelt)")
        plt.colorbar(im1, ax=ax[0])

        # Fit
        im2 = ax[1].imshow(fit_model, cmap='jet')
        ax[1].set_title(f"Super-Gauß Fit (n={n_res:.2f})")
        plt.colorbar(im2, ax=ax[1])

        plt.show()

        print("-" * 30)
        print(f"ERGEBNIS:")
        print(f"Ordnung n: {n_res:.2f}")
        print(f" -> n=2.0 wäre ein perfekter Gauß")
        print(f" -> n>2.0 bedeutet Flat-Top (steile Flanken)")
        print("-" * 30)

    except Exception as e:
        print(f"Fit konnte nicht konvergieren: {e}")
        print("Startwerte (p0):", p0)
        print("Bildshape (hc,wc):", (hc, wc))
        return

# --- Hilfsfunktionen (Technischer Teil) ---
def read_wcf_frames(filepath):
    """Liest Frames aus einer DataRay WCF Datei.
    Erwartet feste Offsets (FILE_HEADER=5592, FRAME_HEADER=944) wie in der originalen Implementierung.
    Gibt ein numpy-array der Form (n_frames, h, w) zurück."""
    frames = []
    # Offsets für DataRay WCF Format
    FILE_HEADER = 5592
    FRAME_HEADER = 944

    with open(filepath, 'rb') as f:
        f.seek(FILE_HEADER)
        while True:
            fh = f.read(FRAME_HEADER)
            if len(fh) < FRAME_HEADER:
                break
            # Lese Breite/Höhe (little-endian unsigned int)
            w = struct.unpack_from('<I', fh, 20)[0]
            h = struct.unpack_from('<I', fh, 24)[0]
            # Validierungen
            if not (1 <= w <= 10000 and 1 <= h <= 10000):
                # Unplausible Werte im Header -- überspringe Rest oder breche ab
                print(f"Warnung: Unplausible Bildgröße im Frame-Header w={w}, h={h}. Breche Einlese-Schleife ab.")
                break
            n_bytes = w * h * 2  # 16-bit Daten
            if n_bytes > 50_000_000:
                raise MemoryError("Frame-Größe > 50MB, Abbruch zum Schutz des Speichers.")
            data = f.read(n_bytes)
            if len(data) < n_bytes:
                # Unvollständiger Frame am Dateiende
                break
            frame = np.frombuffer(data, dtype=np.uint16).reshape((h, w))
            frames.append(frame)

    return np.array(frames)


def super_gaussian_2d(xy, offset, amp, x0, y0, wx, wy, n):
    """Super-Gauß Modell.
    Parameters: xy: stacked x,y arrays (2, N)
    Returns: 1D array mit Modellwerten."""
    x, y = xy
    # Schutz gegen Division durch Null
    wx = max(wx, 1e-6)
    wy = max(wy, 1e-6)
    rad = ((x - x0) / wx) ** 2 + ((y - y0) / wy) ** 2
    return offset + amp * np.exp(-2.0 * rad ** (np.abs(n) / 2.0))


if __name__ == "__main__":
    main()
