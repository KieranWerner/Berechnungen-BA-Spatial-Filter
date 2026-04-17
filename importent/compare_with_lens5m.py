# NEW COMPARISON FILE USING theory_lens_5m_only

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# IMPORT THEORY FUNCTION
# ============================================================

from theory_lens_5m_only import propagate_theory_only, CFG as THEORY_CFG

# ============================================================
# CONFIG
# ============================================================

CFG = {
    "MEAS_WCF_DIR": Path(r"C:\Users\User\Desktop\Real lens propagation"),
    "MEAS_PIXEL_SIZE_UM": 11,   # <-- SET THIS

    "OUTPUT_DIR": Path(r"C:\Users\User\Desktop\Comparison_lens_5m"),

    "COMPARISON_METRIC": "D4sigma_avg_mm",
}

# ============================================================
# SIMPLE WCF READER (reuse your previous logic if needed)
# ============================================================

import re
import numpy as np

def parse_z_from_filename(path: Path):
    m = re.search(r'(\d+(?:\.\d+)?)', path.stem)
    return float(m.group(1)) / 100.0  # cm → m


def load_wcf_dummy(path: Path):
    # simplified: assumes you already have processed images as numpy or adapt
    raise NotImplementedError("Use your existing WCF reader here.")


# ============================================================
# MEASUREMENT (simplified placeholder)
# ============================================================

def run_measurement():
    rows = []
    for f in sorted(CFG["MEAS_WCF_DIR"].glob("*.wcf")):
        z = parse_z_from_filename(f)

        # TODO: replace with your real loader
        # img = load_wcf(...)
        continue

    return pd.DataFrame(rows)


# ============================================================
# COMPARISON
# ============================================================

def run_comparison(df_meas, df_theory):
    metric = CFG["COMPARISON_METRIC"]

    z_meas = df_meas["z_m"].values
    meas = df_meas[metric].values

    z_theory = df_theory["z_m"].values
    theory = df_theory[metric].values

    interp = np.interp(z_meas, z_theory, theory)

    diff = meas - interp
    ratio = meas / interp

    plt.figure()
    plt.plot(z_theory, theory, label="Theory")
    plt.plot(z_meas, meas, "o", label="Measurement")
    plt.legend()
    plt.grid()
    plt.savefig(CFG["OUTPUT_DIR"] / "overlay.png")

    plt.figure()
    plt.plot(z_meas, diff, "o-")
    plt.axhline(0)
    plt.title("Difference")
    plt.savefig(CFG["OUTPUT_DIR"] / "diff.png")

    plt.figure()
    plt.plot(z_meas, ratio, "o-")
    plt.axhline(1)
    plt.title("Ratio")
    plt.savefig(CFG["OUTPUT_DIR"] / "ratio.png")


# ============================================================
# MAIN
# ============================================================

def main():
    CFG["OUTPUT_DIR"].mkdir(exist_ok=True, parents=True)

    # run theory (your new script)
    propagate_theory_only()

    theory_csv = THEORY_CFG["OUTPUT_DIR"] / "theory_lens_5m_beam_size_vs_z.csv"
    df_theory = pd.read_csv(theory_csv)

    # measurement (use your existing code instead)
    df_meas = run_measurement()

    run_comparison(df_meas, df_theory)


if __name__ == "__main__":
    main()
