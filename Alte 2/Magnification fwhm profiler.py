import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from scipy.ndimage import map_coordinates


def load_image(path):
    img = imread(path)
    if img.ndim == 3:
        img = rgb2gray(img)
    img = img.astype(np.float64)
    img -= np.min(img)
    maxv = np.max(img)
    if maxv > 0:
        img /= maxv
    return img


def find_center(img):
    y, x = np.indices(img.shape)
    total = np.sum(img)
    if total <= 0:
        return img.shape[1] / 2, img.shape[0] / 2
    cx = np.sum(x * img) / total
    cy = np.sum(y * img) / total
    return cx, cy


def extract_line_profile(img, cx, cy, angle_rad, half_length=None, num_points=4000):
    h, w = img.shape

    if half_length is None:
        half_length = np.hypot(w, h)

    s = np.linspace(-half_length, half_length, num_points)
    x = cx + s * np.cos(angle_rad)
    y = cy + s * np.sin(angle_rad)

    valid = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    x_valid = x[valid]
    y_valid = y[valid]
    s_valid = s[valid]

    profile = map_coordinates(img, [y_valid, x_valid], order=1, mode="nearest")
    return s_valid, profile


def width_at_threshold(s, profile, threshold):
    if len(profile) < 2:
        return np.nan

    peak = np.max(profile)
    if peak <= 0:
        return np.nan

    level = threshold * peak
    indices = np.where(profile >= level)[0]

    if len(indices) < 2:
        return np.nan

    return s[indices[-1]] - s[indices[0]]


def compute_diagonal_widths(img, thresholds):
    cx, cy = find_center(img)

    # +45°: top-left to bottom-right
    # -45°: top-right to bottom-left
    angles = [np.pi / 4, -np.pi / 4]
    labels = ["diag_left", "diag_right"]

    results = {}

    for angle, label in zip(angles, labels):
        s, profile = extract_line_profile(img, cx, cy, angle)
        widths = [width_at_threshold(s, profile, t) for t in thresholds]
        results[label] = {
            "s": s,
            "profile": profile,
            "widths": np.array(widths, dtype=float),
        }

    return results, (cx, cy)


def print_table(thresholds, res1, res2=None, name1="Image 1", name2="Image 2"):
    if res2 is None:
        print(f"\n{name1}")
        print("Threshold [%] | Width diag_left [px] | Width diag_right [px]")
        print("-" * 62)
        for t, w1, w2 in zip(
            thresholds,
            res1["diag_left"]["widths"],
            res1["diag_right"]["widths"],
        ):
            print(f"{100*t:12.0f} | {w1:20.2f} | {w2:21.2f}")
    else:
        print(f"\nComparison: {name1} / {name2}")
        print("Threshold [%] | Magnification diag_left | Magnification diag_right")
        print("-" * 70)
        for t, a1, a2, b1, b2 in zip(
            thresholds,
            res1["diag_left"]["widths"],
            res1["diag_right"]["widths"],
            res2["diag_left"]["widths"],
            res2["diag_right"]["widths"],
        ):
            m1 = a1 / b1 if np.isfinite(a1) and np.isfinite(b1) and b1 != 0 else np.nan
            m2 = a2 / b2 if np.isfinite(a2) and np.isfinite(b2) and b2 != 0 else np.nan
            print(f"{100*t:12.0f} | {m1:23.3f} | {m2:24.3f}")


def plot_results(thresholds, res1, res2=None, name1="Image 1", name2="Image 2"):
    plt.figure(figsize=(8, 5))
    plt.plot(100 * thresholds, res1["diag_left"]["widths"], "o-", label=f"{name1} diag_left")
    plt.plot(100 * thresholds, res1["diag_right"]["widths"], "o-", label=f"{name1} diag_right")

    if res2 is not None:
        plt.plot(100 * thresholds, res2["diag_left"]["widths"], "s--", label=f"{name2} diag_left")
        plt.plot(100 * thresholds, res2["diag_right"]["widths"], "s--", label=f"{name2} diag_right")

    plt.xlabel("Threshold [% of maximum]")
    plt.ylabel("Width along diagonal [px]")
    plt.title("Beam size along both diagonals")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    if res2 is not None:
        m_left = res1["diag_left"]["widths"] / res2["diag_left"]["widths"]
        m_right = res1["diag_right"]["widths"] / res2["diag_right"]["widths"]

        plt.figure(figsize=(8, 5))
        plt.plot(100 * thresholds, m_left, "o-", label="Magnification diag_left")
        plt.plot(100 * thresholds, m_right, "o-", label="Magnification diag_right")
        plt.xlabel("Threshold [% of maximum]")
        plt.ylabel("Magnification factor")
        plt.title("Magnification factor along both diagonals")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    folder = r"C:\Users\User\Desktop\Vergroesserung"

    path1 = os.path.join(folder, "original groesse.tiff")
    path2 = os.path.join(folder, "profiler begin.tiff")

    img1 = load_image(path1)
    img2 = load_image(path2)

    thresholds = np.arange(0.05, 1.00, 0.05)  # 5% to 95%

    res1, center1 = compute_diagonal_widths(img1, thresholds)
    res2, center2 = compute_diagonal_widths(img2, thresholds)

    print(f"Center Image 1: x={center1[0]:.2f}, y={center1[1]:.2f}")
    print(f"Center Image 2: x={center2[0]:.2f}, y={center2[1]:.2f}")

    print_table(thresholds, res1, name1="original groesse")
    print_table(thresholds, res2, name1="profiler begin")
    print_table(thresholds, res1, res2, name1="original groesse", name2="profiler begin")

    plot_results(thresholds, res1, res2, name1="original groesse", name2="profiler begin")