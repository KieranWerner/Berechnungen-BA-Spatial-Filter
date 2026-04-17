from pathlib import Path
import numpy as np

INPUT_FOLDER = Path(r"C:\Users\User\Desktop\average PHA front csv")
FILE_PATTERNS = ["PHA*.csv"]

OUTPUT_CSV = INPUT_FOLDER / "average_PHA.csv"


def load_csv_auto(path: Path) -> np.ndarray:
    for delim in [",", ";", "\t"]:
        try:
            arr = np.loadtxt(path, delimiter=delim)
            if arr.ndim == 2 and arr.size > 0:
                return np.asarray(arr, dtype=np.float64)
        except Exception:
            pass
    raise ValueError(f"Could not parse CSV: {path}")


def main():
    files = []
    for pattern in FILE_PATTERNS:
        files.extend(INPUT_FOLDER.glob(pattern))

    files = sorted(files)

    if len(files) == 0:
        raise FileNotFoundError("Keine passenden Dateien gefunden.")

    print(f"{len(files)} Dateien gefunden.")

    arrays = []
    shape_ref = None

    for f in files:
        print(f"Lade: {f.name}")
        arr = load_csv_auto(f)

        if shape_ref is None:
            shape_ref = arr.shape
        elif arr.shape != shape_ref:
            raise ValueError(f"Shape mismatch: {f.name} -> {arr.shape} != {shape_ref}")

        arrays.append(arr)

    stack = np.stack(arrays, axis=0)
    average = np.mean(stack, axis=0)

    np.savetxt(OUTPUT_CSV, average, delimiter=",", fmt="%.10g")

    print("\nFertig.")
    print(f"Gespeichert: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()