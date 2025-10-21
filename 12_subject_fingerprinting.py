"""Script for subject fingerprinting analysis of the Cam-CAN dataset."""

# Import packages
import os
import numpy as np
from utils import analysis as ua
from utils import plotting as up
from utils import statistics as us


if __name__ == "__main__":
    # ---------- User Inputs ---------- #
    # Set model names
    model_names = [
        "causal", "noncausal",
        "mu_transform", "mu_transform_big",
        "mu_transform_small", "mu_transform_tiny",
        "standard_quantile",
    ]
    n_models = len(model_names)

    # Set hyperparameters
    gt_run_id = 1  # generator model run ID
    n_generations = 10  # number of generations per model
    Fs = 250  # sampling frequency (Hz)

    # Whether to load the features and pairwise distance
    load = True

    # ---------- Directories ---------- #
    BASE_DIR = "/well/woolrich/users/olt015/Cho2025_Tokenizer"
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODEL_DIR = os.path.join(BASE_DIR, "models/generator")
    PLOT_DIR = os.path.join(BASE_DIR, "plots/generator")

    # ---------- Fingerprinting Analysis ---------- #
    # Get real data path
    real_data_path = os.path.join(DATA_DIR, "original_data.pkl")

    # Initialize dictionaries to store results
    pred_accuracy = {}
    top_1_accuracy = {}
    consistency_scores = {}

    # Perform fingerprinting analysis for each model
    for mod in model_names:
        print(f"Processing model [{mod}] for fingerprinting analysis ...")

        # Set directories
        generator_dir = f"{MODEL_DIR}/{mod}/{gt_run_id}"
        plot_dir = f"{PLOT_DIR}/{mod}/{gt_run_id}/fingerprint"
        save_dir = f"{MODEL_DIR}/{mod}/{gt_run_id}/fingerprint"

        os.makedirs(plot_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)

        # Initialize lists to store results
        pred_accuracy[mod] = []
        top_1_accuracy[mod] = []
        top_5_accuracies = []  # temporary list
        consistency_scores[mod] = []

        # Get generated data paths
        generated_data_paths = [
            f"{generator_dir}/generated_data_{i}.pkl"
            for i in range(n_generations)
        ]

        for i in range(n_generations):
            # Get TDE covariance matrices
            real_features, generated_features = ua.get_fingerprint_features(
                feature_type="tde",
                real_data_path=real_data_path,
                generated_data_path=generated_data_paths[i],
                save_dir=f"{save_dir}/gen{i}",
                Fs=Fs,
                load=load,
            )

            # Save the pairwise distance
            pairwise_distance = ua.get_fingerprint_pairwise_distance(
                real_features,
                generated_features,
                metric_types="correlation",
                save_dir=f"{save_dir}/gen{i}",
                load=load,
            )
            pairwise_distance = pairwise_distance[0]  # only one metric type

            # Get consistency score
            consistency_scores[mod].append(
                ua.get_fingerprint_consistency_score(pairwise_distance)
            )

            # Get the number of subjects
            n_subjects = pairwise_distance.shape[0] // 2
            print(f"Number of subjects: {n_subjects}")

            # Get accuracy
            accuracy = [
                ua.get_fingerprint_accuracy(pairwise_distance, k)
                for k in range(1, n_subjects + 1)
            ]
            pred_accuracy[mod].append(accuracy)
            top_1_accuracy[mod].append(accuracy[0])
            top_5_accuracies.append(accuracy[4])

        # Report metrics
        print(f"Model [{mod}]")
        print(f"\tTop-1 Accuracy: {np.mean(top_1_accuracy[mod]):.6f} ± {np.std(top_1_accuracy[mod]):.6f}")
        print(f"\tTop-5 Accuracy: {np.mean(top_5_accuracies):.6f} ± {np.std(top_5_accuracies):.6f}")
        print(f"\tConsistency Score: {np.mean(consistency_scores[mod]):.6f} ± {np.std(consistency_scores[mod]):.6f}")

    # ---------- Visualization ---------- #
    # Set color palettes
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

    # Reorder metrics by performance
    cs_order = np.argsort([
        np.mean(consistency_scores[mod]) for mod in model_names
        # average over data generations
    ])[::-1]
    consistency_scores = {
        mod: consistency_scores[mod]
        for mod in [model_names[i] for i in cs_order]
    }

    top_1_acc_order = np.argsort([
        np.mean(top_1_accuracy[mod]) for mod in model_names
        # average over data generations
    ])[::-1]
    top_1_accuracy = {
        mod: top_1_accuracy[mod]
        for mod in [model_names[i] for i in top_1_acc_order]
    }

    # Reorder color palettes
    cs_color_palette = {
        color_keys[i]: color_palette[color_keys[i]]
        for i in cs_order
    }
    cs_model_names = [model_names[i] for i in cs_order]

    top_1_acc_color_palette = {
        color_keys[i]: color_palette[color_keys[i]]
        for i in top_1_acc_order
    }
    top_1_acc_model_names = [model_names[i] for i in top_1_acc_order]

    # Plot top-k accuracy curves
    up.plot_top_k_accuracy(
        pred_accuracy,
        model_names,
        palette=color_palette,
        filename=f"{PLOT_DIR}/fp_top_k_accuracy.png",
    )

    # Plot top-1 accuracy scores
    up.plot_fingerprint_box(
        top_1_accuracy,
        metric_name="Top-1 Accuracy",
        model_names=top_1_acc_model_names,
        palette=top_1_acc_color_palette,
        filename=f"{PLOT_DIR}/fp_top_1_accuracies.png",
        strip=True,
        ylim=[0.05, 0.42],
    )

    # Plot consistency scores
    up.plot_fingerprint_box(
        consistency_scores,
        metric_name="Consistency Score",
        model_names=cs_model_names,
        palette=cs_color_palette,
        filename=f"{PLOT_DIR}/fp_consistency_scores.png",
        strip=True,
        ylim=[0.74, 0.82],
    )

    # ---------- Statistical Analysis ---------- #
    # Get pairwise combinations
    pairs = [(i, i + 1) for i in range(n_models - 1)]

    # Set threshold
    alpha = 0.05
    n_tests = len(pairs)
    print(f"Number of tests: {n_tests}")

    # Perform statistical tests
    for name, metric, order in zip(
        ["Top-1 Accuracy", "Consistency Score"],
        [top_1_accuracy, consistency_scores],
        [top_1_acc_order, cs_order],
    ):
        print("\nStatistical Analysis for", name)
        mod_names = [model_names[i] for i in order]  # reorder model names
        for i, j in pairs:
            print(f"{mod_names[i].title()} vs {mod_names[j].title()}")
            stat, pval, sig_indicator = us.stat_ind_two_samples(
                metric[mod_names[i]], metric[mod_names[j]],
                alpha=alpha,
                bonferroni_ntest=n_tests,
                test="welch",
            )  # Welch's t-test was chosen after checking assumptions; use test=None to check

    print("Analysis complete.")
