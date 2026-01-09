"""Script for visualizing the training history of generator models."""

# Import packages
import os
import numpy as np
from utils import analysis as ua
from utils import plotting as up


if __name__ == "__main__":
    # ---------- Directories ---------- #
    BASE_DIR = "/well/woolrich/users/olt015/Cho2026_Tokenizer"
    MODEL_DIR = os.path.join(BASE_DIR, "models/generator")
    PLOT_DIR = os.path.join(BASE_DIR, "plots/generator")

    fig_dir = os.path.join(PLOT_DIR, "history")
    os.makedirs(fig_dir, exist_ok=True)

    # ---------- Visualization ---------- #
    # Define model names
    model_names = [
        "causal", "noncausal",
        "mu_transform", "mu_transform_big",
        "mu_transform_small", "mu_transform_tiny",
        "standard_quantile",
    ]

    # Set color palette
    token_nums = np.load(f"{BASE_DIR}/models/tokenizer/token_nums.npy")
    color_palette = {
        f"Causal\n(n={token_nums[0]})": "#E69F00",
        f"Noncausal\n(n={token_nums[1]})": "#56B4E9",
        f"Mu\n(n={token_nums[2]})": "#009E73",
        f"Mu\n(n={token_nums[3]})": "#F0E442",
        f"Mu\n(n={token_nums[4]})": "#0072B2",
        f"Mu\n(n={token_nums[5]})": "#D55E00",
        f"SQ\n(n={token_nums[6]})": "#CC79A7",
    }

    # Plot training and validation performance metrics
    # (i.e., loss and top 1 accuracy curves)
    for model_name in model_names:
        up.plot_generator_history(
            model_type=model_name,
            run_id=1,
            model_dir=MODEL_DIR,
            save_dir=PLOT_DIR,
        )

    # Compute log-relative loss and measure convergence rates
    log_relative_loss = []
    convergence_rates = []
    for model_name in model_names:
        lrl, cr = ua.compute_log_relative_loss(
            model_type=model_name,
            run_id=1,
            model_dir=MODEL_DIR,
            loss_name="val_loss",  # use "train_loss" for training loss
            plot_fit=True,
            plot_dir=fig_dir,
        )
        log_relative_loss.append(lrl)
        convergence_rates.append(cr)

    log_relative_loss = np.array(log_relative_loss)
    # shape: (n_models, n_epochs)
    convergence_rates = np.array(convergence_rates)
    # shape: (n_models, n_epochs - 1)

    # Plot log-relative loss and convergence rates
    up.plot_convergence_metrics(
        metrics=log_relative_loss,
        label="Log-Relative Loss",
        color_palette=color_palette,
        filename=f"{fig_dir}/log_relative_loss.png",
    )

    up.plot_convergence_metrics(
        metrics=convergence_rates,
        label="Convergence Rate",
        color_palette=color_palette,
        filename=f"{fig_dir}/convergence_rate.png",
    )

    print("Visualization complete.")
