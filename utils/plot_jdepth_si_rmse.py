#!/usr/bin/env python3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


EPOCHS = list(range(1, 41))
TRAIN = [
    0.639271, 0.575258, 0.551227, 0.534865, 0.529458,
    0.518556, 0.515439, 0.510325, 0.506853, 0.503848,
    0.500566, 0.500036, 0.496311, 0.495792, 0.493229,
    0.494905, 0.491181, 0.472721, 0.471920, 0.470995,
    0.470325, 0.470897, 0.470071, 0.469553, 0.469328,
    0.468995, 0.468896, 0.468715, 0.468485, 0.468204,
    0.468357, 0.468161, 0.468035, 0.467355, 0.467334,
    0.467159, 0.466756, 0.466412, 0.466166, 0.466121,
]
VAL = [
    0.578803, 0.560970, 0.563933, 0.537460, 0.524659,
    0.534071, 0.522884, 0.516023, 0.512446, 0.509065,
    0.503120, 0.501661, 0.502146, 0.503952, 0.506642,
    0.501295, 0.502389, 0.473410, 0.469587, 0.469920,
    0.470416, 0.470594, 0.469487, 0.469694, 0.469668,
    0.469134, 0.469357, 0.468425, 0.469265, 0.468479,
    0.470654, 0.468479, 0.468895, 0.467761, 0.468319,
    0.467715, 0.467795, 0.467889, 0.468369, 0.467187,
]


def smooth(values, window=3):
    smoothed = []
    half = window // 2
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        smoothed.append(sum(values[lo:hi]) / (hi - lo))
    return smoothed


def main():
    out = Path("/home/msayfiddinov/mono/jdepth_si_rmse_40_epochs.png")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=220)

    ax.plot(EPOCHS, smooth(TRAIN), color="#1f77b4", linewidth=2.8, label="Train SI-RMSE")
    ax.plot(EPOCHS, smooth(VAL), color="#d62728", linewidth=2.8, label="Val SI-RMSE")

    ax.set_title("JDepth SI-RMSE over 40 Epochs", fontsize=15, pad=14)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("SI-RMSE", fontsize=12)
    ax.set_xlim(1, 40)
    ax.set_ylim(0.46, 0.645)
    ax.set_xticks(range(1, 41, 2))
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=10)
    ax.grid(True, color="#d9dde3", linewidth=0.8, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
