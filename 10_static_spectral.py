"""Script for the post-hoc analysis of static spectral features."""

# Import packages
import os
import numpy as np
import pandas as pd
import seaborn as sns

from collections import defaultdict
from osl_dynamics.analysis import static
from utils import analysis as ua
from utils import data as ud
from utils import plotting as up


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

    # ---------- Directories ---------- #
    BASE_DIR = "/well/woolrich/users/olt015/Cho2025_Tokenizer"
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODEL_DIR = os.path.join(BASE_DIR, "models/generator")
    PLOT_DIR = os.path.join(BASE_DIR, "plots/generator")

    # ---------- Load Data ---------- #
    print("Loading data ...")

    # Load original data
    original_data = ud.load(f"{DATA_DIR}/original_data.pkl")
    # shape: (n_subjects, n_samples, n_channels)

    # Load generated data
    generated_data = defaultdict(dict)
    for name in model_names:
        for gen_id in range(n_generations):
            generated_data[name][gen_id] = ud.load(
                f"{MODEL_DIR}/{name}/{gt_run_id}/generated_data_{gen_id}.pkl"
            )  # shape: (n_subjects, n_generated_samples, n_channels)

    # Trim original data to match generated data length
    n_generated_samples = generated_data[model_names[0]][0][0].shape[0]
    original_data = [d[:n_generated_samples] for d in original_data]

    # ---------- Compute Static Spectral Features ---------- #
    save_path = os.path.join(DATA_DIR, f"static_psds_gt{gt_run_id}.pkl")

    if not os.path.exists(save_path):
        print("Computing static spectral features ...")

        # Compute subject-level static PSDs
        freq, psd_real = static.welch_spectra(
            data=original_data,
            sampling_frequency=Fs,
            frequency_range=[1, 45],
            n_jobs=12,
        )  # psd_real.shape = (n_subjects, n_channels, n_frequencies)

        psd_gen = defaultdict(dict)
        for name in model_names:
            for gen_id in range(n_generations):
                _, p_gen = static.welch_spectra(
                    data=generated_data[name][gen_id],
                    sampling_frequency=Fs,
                    frequency_range=[1, 45],
                    n_jobs=12,
                )  # p_gen.shape = (n_subjects, n_channels, n_frequencies)
                psd_gen[name][gen_id] = p_gen

        # Save computed PSDs
        ud.save((freq, psd_real, psd_gen), save_path)
    else:
        print("Loading pre-computed static spectral features ...")
        freq, psd_real, psd_gen = ud.load(save_path)

    # ---------- Set visualization parameters ---------- #
    # Set color palette
    token_nums = np.load(f"{BASE_DIR}/models/tokenizer/token_nums.npy")
    color_palette = {
        f"Causal (n={token_nums[0]})": "#E69F00",
        f"Noncausal (n={token_nums[1]})": "#56B4E9",
        f"Mu (n={token_nums[2]})": "#009E73",
        f"Mu (n={token_nums[3]})": "#F0E442",
        f"Mu (n={token_nums[4]})": "#0072B2",
        f"Mu (n={token_nums[5]})": "#D55E00",
        f"SQ (n={token_nums[6]})": "#CC79A7",
    }

    x_labels = list(k.replace(" ", "\n") for k in color_palette.keys())

    color_palette_1 = {
        k: v for k, v in zip(model_names, color_palette.values())
    }

    cmap = sns.color_palette("flare", as_cmap=True)
    cval = np.linspace(0, 1, n_generations)
    color_palette_2 = {
        k: cmap(v) for k, v in zip(list(range(n_generations)), cval)
    }

    # Define parcellation file
    parcellation_file = "Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz"

    # ---------- Visualize Static Spectral Features ---------- #
    print("Visualizing static spectral features (for single data generation)...")

    # Plot group-level PSDs
    up.plot_psd(
        psd_real,
        *[psd_gen[name][0] for name in model_names],
        freq=freq,
        parcellation_file=parcellation_file,
        plot_dir=f"{PLOT_DIR}/psds",
        titles=["Original"] + list(color_palette.keys()),
    )

    # Plot group-level static power maps
    map_names = ["original"] + model_names
    for i, psd in enumerate(
        [psd_real] + [psd_gen[name][0] for name in model_names]
    ):
        up.plot_static_power_maps(
            psd,
            freq=freq,
            filename=f"{PLOT_DIR}/power_maps/power_{map_names[i]}.png",
            fontsize=40,
        )

    # ---------- Compare Original and Generated PSDs ---------- #
    # Compute distance metrics between original and generated PSDs
    l2_dist = defaultdict(dict)
    cos_sim = defaultdict(dict)
    for name in model_names:
        for gen_id in range(n_generations):
            l2_dist[name][gen_id] = ua.compute_l2_distance(
                psd_real, psd_gen[name][gen_id], axis=-1
            )  # shape: (n_subjects, n_channels)
            cos_sim[name][gen_id] = ua.compute_cosine_similarity(
                psd_real, psd_gen[name][gen_id], axis=-1
            )  # shape: (n_subjects, n_channels)

    # Make .csv file and save distance metrics
    print("Saving distance metrics ...")
    save_path = f"{DATA_DIR}/static_gt{gt_run_id}_.csv"
    
    df_l2 = ud.metric_dict_to_long(l2_dist)
    df_l2.to_csv(save_path.replace(".csv", "l2.csv"), index=False)

    df_cos = ud.metric_dict_to_long(cos_sim)
    df_cos.to_csv(save_path.replace(".csv", "cos.csv"), index=False)

    # ---------- Visualize Metrics ---------- #

    ############ IMPORTANT ############
    # You need to run the R script `10-1_static_spectral_lmm.r` 
    # before this step in order to proceed with the visualizations.
    ###################################

    METRIC_DIR = os.path.join(DATA_DIR, "static_spectral")
    if not os.path.exists(METRIC_DIR):
        raise ValueError(
            "Statistical test results do not exist. Please run the R script first."
        )

    # Load L2 distance metrics and estimated marginal means (EMMs)
    l2_distances = pd.read_csv(f"{METRIC_DIR}/subject_means_l2.csv")  # L2 distances per subject, averaged over datasets
    emm_model_l2 = pd.read_csv(f"{METRIC_DIR}/emm_model_l2.csv")  # estimated marginal means (EMMs) per model

    l2_distances["model"] = pd.Categorical(
        l2_distances["model"], categories=model_names, ordered=True
    )
    emm_model_l2["model"] = pd.Categorical(
        emm_model_l2["model"], categories=model_names, ordered=True
    )

    # Plot subject-level distributions of L2 distance per model
    print("Visualizing distributions of L2 distance metrics ...")

    up.plot_metric_violin(
        df=l2_distances,
        x="model",
        y="mean_metric",
        hue="model",
        emm_df=emm_model_l2,
        metric_name="L2 Distance",
        palette=color_palette_1,
        x_labels=x_labels,
        filename=os.path.join(PLOT_DIR, "l2_distances.png"),
    )

    # Print the pairwise EMM test results (post-hoc tests with multiple comparison correction)
    pairs_l2 = pd.read_csv(f"{METRIC_DIR}/pairwise_model_contrasts_l2.csv")
    print("\nPairwise EMM comparisons (Tukey-adjusted):")
    print(pairs_l2[["contrast", "estimate", "SE", "z.ratio", "p.value"]])  # adjusted p-values

    # Plot model × dataset interaction for L2 distances (EMMs)
    print("Visualizing model-dataset interaction for L2 distances ...")
    emm_md_l2 = pd.read_csv(f"{METRIC_DIR}/emm_model_by_dataset_l2.csv")
    emm_md_l2["model"] = pd.Categorical(emm_md_l2["model"], categories=model_names, ordered=True)

    up.plot_model_dataset_interaction(
        emm_df=emm_md_l2,
        metric_name="L2 Distance",
        palette=color_palette_2,
        x_labels=x_labels,
        filename=os.path.join(PLOT_DIR, "l2_distances_interaction.png"),
    )

    # Load cosine similarity metrics and estimated marginal means (EMMs)
    cos_sim = pd.read_csv(f"{METRIC_DIR}/subject_means_cos.csv")  # metrics per subject, averaged over datasets
    emm_model_cos = pd.read_csv(f"{METRIC_DIR}/emm_model_cos.csv")  # estimated marginal means (EMMs) per model

    cos_sim["model"] = pd.Categorical(
        cos_sim["model"], categories=model_names, ordered=True
    )
    emm_model_cos["model"] = pd.Categorical(
        emm_model_cos["model"], categories=model_names, ordered=True
    )

    # Plot subject-level distributions of cosine similarity per model
    print("Visualizing distributions of cosine similarities ...")

    up.plot_metric_violin(
        df=cos_sim,
        x="model",
        y="mean_metric",
        hue="model",
        emm_df=emm_model_cos,
        metric_name="Cosine Similarity",
        palette=color_palette_1,
        x_labels=x_labels,
        filename=os.path.join(PLOT_DIR, "cosine_similarities.png"),
    )

    # Print the pairwise EMM test results (post-hoc tests with multiple comparison correction)
    pairs_cos = pd.read_csv(f"{METRIC_DIR}/pairwise_model_contrasts_cos.csv")
    print("\nPairwise EMM comparisons (Tukey-adjusted):")
    print(pairs_cos[["contrast", "estimate", "SE", "z.ratio", "p.value"]])  # adjusted p-values

    # Plot model × dataset interaction for cosine similarities (EMMs)
    print("Visualizing model-dataset interaction for cosine similarities ...")
    emm_md_cos = pd.read_csv(f"{METRIC_DIR}/emm_model_by_dataset_cos.csv")
    emm_md_cos["model"] = pd.Categorical(emm_md_cos["model"], categories=model_names, ordered=True)

    up.plot_model_dataset_interaction(
        emm_df=emm_md_cos,
        metric_name="Cosine Similarity",
        palette=color_palette_2,
        x_labels=x_labels,
        filename=os.path.join(PLOT_DIR, "cosine_similarities_interaction.png"),
    )
    
    print("Analysis complete.")
