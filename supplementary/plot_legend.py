"""Script for visualizing the plot legend separately."""

# Import packages
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from utils import plotting as up


if __name__ == "__main__":
    # ---------- Directories ---------- #
    BASE_DIR = "/well/woolrich/users/olt015/Cho2026_Tokenizer"
    PLOT_DIR = os.path.join(BASE_DIR, "plots")

    token_nums = np.load(f"{BASE_DIR}/models/tokenizer/token_nums.npy")
    color_palette = {
        "Baseline": "#787878FF",
        f"Causal (n={token_nums[0]})": "#E69F00",
        f"Noncausal (n={token_nums[1]})": "#56B4E9",
        f"Mu (n={token_nums[2]})": "#009E73",
        f"Mu (n={token_nums[3]})": "#F0E442",
        f"Mu (n={token_nums[4]})": "#0072B2",
        f"Mu (n={token_nums[5]})": "#D55E00",
        f"SQ (n={token_nums[6]})": "#CC79A7",
    }

    # Create handles for filled color patches
    patch_handles = [
        Patch(facecolor=color, label=label)
        for label, color in color_palette.items()
    ]

    # Create handles for lines
    line_handles = [
        Line2D(
            [0], [0], color="k",
            linewidth=2, linestyle="--",
            label="Random",
        )
    ]

    # Combine all handles
    handles = line_handles + patch_handles

    # Make a legend-only figure
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.axis('off')
    legend = ax.legend(
        handles=handles,
        loc="center",
        frameon=True,
        fancybox=True,
        ncol=1,
        handlelength=2.5,
        borderpad=1.2,
        fontsize=14,
    )
    plt.tight_layout()
    up.save(fig, f"{PLOT_DIR}/legend.png", transparent=True)

    print("Visualization complete.")
