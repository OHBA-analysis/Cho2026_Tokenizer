"""Script for visualizing the token prediction analysis results."""

# Import packages
import os
import numpy as np
import pandas as pd
from utils import data as ud
from utils import plotting as up
from utils import statistics as us


if __name__ == "__main__":
    # ---------- Directories ---------- #
    BASE_DIR = "/well/woolrich/users/olt015/Cho2026_Tokenizer"
    MODEL_DIR = os.path.join(BASE_DIR, "models/generator")
    PLOT_DIR = os.path.join(BASE_DIR, "plots/generator")

    # ---------- User Inputs ---------- #
    loss_sequence_length = 8
    sequence_length = 81

    remove_outlier_flag = False
    print("Remove outliers:", remove_outlier_flag)

    # ---------- Load Data ---------- #
    model_names = [
        "causal", "noncausal",
        "mu_transform", "mu_transform_big",
        "mu_transform_small", "mu_transform_tiny",
        "standard_quantile",
    ]
    n_models = len(model_names)

    recon_mses, recon_pves = [], []
    for i, name in enumerate(model_names):
        print(f"Tokenizer ({model_names[i]})")

        # Load and unpack outputs
        outputs = ud.load(
            os.path.join(MODEL_DIR, f"{name}/1/token_pred_result.pkl")
        )
        accuracy_score, raw_inputs, recon_preds = outputs.values()

        # Validate inputs
        print("\tInput shape:", raw_inputs.shape)
        print("\tPrediction shape:", recon_preds.shape)

        if raw_inputs.shape != recon_preds.shape:
            raise ValueError("Input and prediction shapes do not match.")

        # Reshape to 2D arrays
        raw_inputs = raw_inputs.reshape(-1, raw_inputs.shape[-1])
        recon_preds = recon_preds.reshape(-1, recon_preds.shape[-1])
        T = raw_inputs.shape[0]  # number of time points

        print(f"\tInput shape: {raw_inputs.shape}")
        print(f"\tPrediction shape: {recon_preds.shape}")

        # Report accuracy
        print(f"Accuracy Score: {accuracy_score}")

        # Calculate mean squared errors
        sse = np.sum((raw_inputs - recon_preds) ** 2, axis=0)
        recon_mse = sse / T  # shape: (n_channels,)

        # Calculate percentage of variance explained
        recon_pve = 100 * (
            1 - (sse / np.sum(raw_inputs ** 2, axis=0))
        )  # shape: (n_channels,)

        if remove_outlier_flag:
            recon_mse = ud.remove_outliers(recon_mse)
            recon_pve = ud.remove_outliers(recon_pve)    

        # Gather metrics
        recon_mses.append(recon_mse)
        recon_pves.append(recon_pve)
        # shape: (n_models, n_channels)

    # ---------- Reorder by Performance ---------- #
    pve_order = np.argsort([np.mean(pve) for pve in recon_pves])[::-1]
    mse_order = np.argsort([np.mean(mse) for mse in recon_mses])

    recon_pves = [recon_pves[i] for i in pve_order]
    recon_mses = [recon_mses[i] for i in mse_order]

    # ---------- Statistical Analysis ---------- #
    # Get pairwise combinations
    pairs = [(i, i + 1) for i in range(n_models - 1)]
    # pairs = list(combinations(range(n_models), 2)) (use for all pairwise comparisons)

    # Set threshold
    alpha = 0.05
    n_tests = len(pairs)
    print(f"Number of tests: {n_tests}")

    # Perform statistical tests
    for name, metric, order in zip(
        ["PVE", "MSE"], [recon_pves, recon_mses], [pve_order, mse_order]
    ):
        print("\nStatistical Analysis for", name)
        mod_names = [model_names[i] for i in order]  # reorder model names
        for i, j in pairs:
            print(f"{mod_names[i].title()} vs {mod_names[j].title()}")
            stat, pval, sig_indicator = us.stat_ind_two_samples(
                metric[i], metric[j],
                alpha=alpha,
                bonferroni_ntest=n_tests,
                test="welch",
            )  # Welch's t-test was chosen after checking assumptions; use test=None to check

    # ---------- Plotting ---------- #
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
    color_keys = list(color_palette.keys())

    # Reorder color palette
    pve_color_palette = {
        color_keys[i]: color_palette[color_keys[i]]
        for i in pve_order
    }
    mse_color_palette = {
        color_keys[i]: color_palette[color_keys[i]]
        for i in mse_order
    }

    # Plot PVEs
    df_pve = pd.DataFrame({
        "PVE": np.concatenate(recon_pves),
        "Model": np.concatenate([
            [name] * len(pve) for name, pve in zip(
                pve_color_palette.keys(), recon_pves
            )
        ]),
    })

    up.plot_token_prediction_loss(
        df_pve,
        metric="pve",
        palette=pve_color_palette,
        filepath=os.path.join(PLOT_DIR, "token_pred_pves.png"),
        ylim=[55, 100],
    )

    # Plot MSEs
    df_mse = pd.DataFrame({
        "MSE": np.concatenate(recon_mses),
        "Model": np.concatenate([
            [name] * len(mse) for name, mse in zip(
                mse_color_palette.keys(), recon_mses
            )
        ]),
    })

    up.plot_token_prediction_loss(
        df_mse,
        metric="mse",
        palette=mse_color_palette,
        filepath=os.path.join(PLOT_DIR, "token_pred_mses.png"),
        ylim=[0.045, 0.405],
    )

    print("Analysis complete.")
