from pathlib import Path
import numpy as np

# ============================================================
# EINSTELLUNGEN
# ============================================================
INPUT_FOLDER = Path(r"C:\Users\User\Desktop\average PHA front csv")
BACKGROUND_FILE = Path(r"C:\Users\User\Desktop\Background average front PHA\PHA SID4 21h53m20s121ms.csv")
OUTPUT_FOLDER = Path(r"C:\Users\User\Desktop\average PHA front csv single background substracted")

FILE_PATTERNS = ["PHA*.csv", "*.csv"]


# ============================================================
# CSV LADEN
# ============================================================
def load_csv_auto(path: Path) -> np.ndarray:
    for delim in [",", ";", "\t"]:
        try:
            arr = np.loadtxt(path, delimiter=delim)
            if arr.ndim == 2 and arr.size > 0:
                return np.asarray(arr, dtype=np.float64)
        except Exception:
            pass
    raise ValueError(f"Could not parse CSV: {path}")


# ============================================================
# HAUPTPROGRAMM
# ============================================================
def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    # Hintergrund laden
    background = load_csv_auto(BACKGROUND_FILE)
    print(f"Hintergrund geladen: {BACKGROUND_FILE}")

    # Dateien sammeln
    files = []
    for pattern in FILE_PATTERNS:
        files.extend(INPUT_FOLDER.glob(pattern))

    # Duplikate entfernen und sortieren
    files = sorted(set(files))

    # Hintergrunddatei nicht mitverarbeiten, falls sie zufällig im Input-Ordner liegt
    files = [f for f in files if f.resolve() != BACKGROUND_FILE.resolve()]

    if len(files) == 0:
        raise FileNotFoundError("Keine passenden CSV-Dateien gefunden.")

    print(f"{len(files)} Dateien gefunden.")

    for f in files:
        print(f"Verarbeite: {f.name}")
        arr = load_csv_auto(f)

        if arr.shape != background.shape:
            raise ValueError(
                f"Shape mismatch: {f.name} -> {arr.shape} != Hintergrund {background.shape}"
            )

        corrected = arr - background

        output_path = OUTPUT_FOLDER / f"{f.stem}_background_subtracted.csv"
        np.savetxt(output_path, corrected, delimiter=",", fmt="%.10g")
        print(f"Gespeichert: {output_path.name}")

    print("\nFertig. Alle Bilder wurden hintergrundbereinigt.")


if __name__ == "__main__":
    main()