"""Script for HMM post-hoc analysis."""

# Import packages
import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from collections import defaultdict
from osl_dynamics.inference import modes
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

    # Set channel indices
    ch_indices = [0, 4, 13, 18, 22]

    # Set hyperparameters
    gt_run_id = 1  # generator model run ID
    n_generations = 10  # number of generations per model
    Fs = 100  # sampling frequency (Hz)

    # ---------- Directories ---------- #
    BASE_DIR = "/well/woolrich/users/olt015/Cho2025_Tokenizer"
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODEL_DIR = os.path.join(BASE_DIR, "models/hmm")
    PLOT_DIR = os.path.join(BASE_DIR, "plots/generator")

    fig_dir = os.path.join(PLOT_DIR, "burst_detection")
    os.makedirs(fig_dir, exist_ok=True)

    # ---------- Load Parameters ---------- #
    print("Loading parameters ...")

    inf_params = defaultdict(dict)
    for n, name in enumerate(["real", *model_names]):
        for c in ch_indices:
            print(f"Processing model - {name}, channel - {c} ...")

            inf_params[name][f"ch{c}"] = {}
            if name == "real":
                mod_dir = f"{MODEL_DIR}/real/{gt_run_id}/ch{c}"

                # Model parameters
                with open(f"{mod_dir}/inf_params/alp.pkl", "rb") as f:
                    alpha = pickle.load(f)
                inf_params[name][f"ch{c}"]["alpha"] = alpha
                inf_params[name][f"ch{c}"]["covs"] = np.load(f"{mod_dir}/inf_params/covs.npy")

                # Training history
                with open(f"{mod_dir}/model/history.pkl", "rb") as f:
                    history = pickle.load(f)
                inf_params[name][f"ch{c}"]["loss"] = history["loss"]
                inf_params[name][f"ch{c}"]["free_energy"] = history["free_energy"]

                # Spectral properties
                inf_params[name][f"ch{c}"]["f"] = np.load(f"{mod_dir}/spectra/f.npy")
                inf_params[name][f"ch{c}"]["psd"] = np.squeeze(np.load(f"{mod_dir}/spectra/psd.npy"))
                inf_params[name][f"ch{c}"]["coh"] = np.squeeze(np.load(f"{mod_dir}/spectra/coh.npy"))

                # Summary statistics
                stc = modes.argmax_time_courses(alpha)
                fo, lt, intv, sr = ua.calculate_summary_stats(stc, sampling_frequency=Fs)
                inf_params[name][f"ch{c}"]["fo"] = fo
                inf_params[name][f"ch{c}"]["lt"] = lt
                inf_params[name][f"ch{c}"]["intv"] = intv
                inf_params[name][f"ch{c}"]["sr"] = sr
                # shape: (n_subjects, n_states)

            else:
                inf_params[name][f"ch{c}"] = {}
                for gen_id in range(n_generations):
                    mod_dir = f"{MODEL_DIR}/{name}/{gt_run_id}/ch{c}/{gen_id}"

                    # Load Model parameters
                    inf_params[name][f"ch{c}"][gen_id] = {}
                    with open(f"{mod_dir}/inf_params/alp.pkl", "rb") as f:
                        alpha = pickle.load(f)  # shape: (n_subjects, n_samples, n_states)
                    gen_covs = np.load(f"{mod_dir}/inf_params/covs.npy")
                    # shape: (n_states, n_channels, n_channels)

                    # Match order to real data
                    real_covs = inf_params["real"][f"ch{c}"]["covs"]
                    order = modes.match_covariances(real_covs, gen_covs, return_order=True)[1]
                    # NOTE: This is an order between real and generated data for specific channel and data generation.
                    # It does not ensure the same order across different channels or data generations.

                    # Model parameters
                    alpha_reordered = [alp[:, order] for alp in alpha]
                    inf_params[name][f"ch{c}"][gen_id]["alpha"] = alpha_reordered
                    inf_params[name][f"ch{c}"][gen_id]["covs"] = gen_covs[order]

                    # Training history
                    with open(f"{mod_dir}/model/history.pkl", "rb") as f:
                        history = pickle.load(f)
                    inf_params[name][f"ch{c}"][gen_id]["loss"] = history["loss"]
                    inf_params[name][f"ch{c}"][gen_id]["free_energy"] = history["free_energy"]

                    # Spectral properties
                    inf_params[name][f"ch{c}"][gen_id]["f"] = np.load(f"{mod_dir}/spectra/f.npy")
                    # shape: (n_frequencies,)
                    psd = np.squeeze(np.load(f"{mod_dir}/spectra/psd.npy"))
                    coh = np.squeeze(np.load(f"{mod_dir}/spectra/coh.npy"))
                    # shape: (n_subjects, n_states, n_frequencies); only one channel

                    inf_params[name][f"ch{c}"][gen_id]["psd"] = psd[:, order, :]
                    inf_params[name][f"ch{c}"][gen_id]["coh"] = coh[:, order, :]

                    # Summary statistics
                    stc = modes.argmax_time_courses(alpha_reordered)
                    fo, lt, intv, sr = ua.calculate_summary_stats(stc, sampling_frequency=Fs)
                    inf_params[name][f"ch{c}"][gen_id]["fo"] = fo
                    inf_params[name][f"ch{c}"][gen_id]["lt"] = lt
                    inf_params[name][f"ch{c}"][gen_id]["intv"] = intv
                    inf_params[name][f"ch{c}"][gen_id]["sr"] = sr
                    # shape: (n_subjects, n_states)

        # ---------- Visualization for Single Channel (and One Data Generation) ---------- #
        print(f"Visualizing results for model [{name}] ...")

        # Select parameters to visualize
        if name == "real":
            ex_params = inf_params[name][f"ch{ch_indices[1]}"]
        else:
            ex_params = inf_params[name][f"ch{ch_indices[1]}"][0]  # first data generation

        # Plot state time courses
        print("Plotting state time courses ...")

        stc = modes.argmax_time_courses(ex_params["alpha"])
        n_samples = int(Fs * 5)  # 5 seconds
        up.plot_alpha(
            stc[0][:n_samples, :],  # single subject
            n_samples=n_samples,
            sampling_frequency=Fs,
            cmap="Set3",
            y_labels=["State Time Course"],
            fig_kwargs={"figsize": (11, 3)},
            filename=os.path.join(fig_dir, f"stc_{name}.png"),
        )  # plots for one example subject

        # Plot state spectra
        print("Plotting state spectra ...")

        up.plot_dynamic_psds(
            freq=ex_params["f"],
            psds=ex_params["psd"],
            filename=os.path.join(fig_dir, f"psd_{name}.png"),
            colors=["#e2685c", "#b13c6c", "#6c2b6d"],
            xlim=[-0.1, 46],
            ylim=[-0.005, 0.145],
            legend=True,
        )

        # Plot training loss curves
        print("Plotting training loss curves ...")

        up.plot_hmm_loss(
            loss=ex_params["loss"],
            filename=os.path.join(fig_dir, f"loss_{name}.png"),
        )

        # Plot summary statistics
        print("Plotting summary statistics ...")

        summary_stats = (
            pd.concat({
                metric: pd.DataFrame(arr).stack()
                for metric, arr in ex_params.items()
                if metric in ["fo", "lt", "intv", "sr"]
            }, names=["Metric", "Subject", "State"])
            .reset_index()
            .rename(columns={0: "Value"})
        )

        up.plot_summary_stats(
            summary_stats,
            metric_name="sr",
            filename=os.path.join(fig_dir, f"sr_{name}.png"),
            palette={
                0: "#e2685c",
                1: "#b13c6c",
                2: "#6c2b6d",
            },
            ylim=[-0.4, 2.2],
        )

    # ---------- Compute Evaluation Metrics ---------- #
    # Compute distance metrics between original and generated PSDs
    l2_psd = defaultdict(dict)
    l2_sr = defaultdict(dict)

    for name in model_names:
        for c in ch_indices:
            l2_psd[name][f"ch{c}"] = {}
            real_psd = inf_params["real"][f"ch{c}"]["psd"]
            # shape: (n_subjects, n_states, n_frequencies)

            l2_sr[name][f"ch{c}"] = {}
            real_sr = inf_params["real"][f"ch{c}"]["sr"]
            # shape: (n_subjects, n_states)

            for gen_id in range(n_generations):
                gen_psd = inf_params[name][f"ch{c}"][gen_id]["psd"]
                dist = ua.compute_l2_distance(real_psd, gen_psd, axis=(1, 2))
                l2_psd[name][f"ch{c}"][gen_id] = dist  # shape: (n_subjects,)

                gen_sr = inf_params[name][f"ch{c}"][gen_id]["sr"]
                dist = ua.compute_l2_distance(real_sr, gen_sr, axis=1)
                l2_sr[name][f"ch{c}"][gen_id] = dist  # shape: (n_subjects,)

    # Make .csv file and save distance metrics
    print("Saving evaluation metrics ...")
    
    save_path = f"{DATA_DIR}/dynamic_gt{gt_run_id}_.csv"

    df_l2_psd = ud.dynamic_metric_dict_to_long(l2_psd)
    df_l2_psd.to_csv(save_path.replace(".csv", "l2_psd.csv"), index=False)

    df_l2_sr = ud.dynamic_metric_dict_to_long(l2_sr)
    df_l2_sr.to_csv(save_path.replace(".csv", "l2_sr.csv"), index=False)

    # ---------- Set visualization parameters ---------- #
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

    x_labels = list(color_palette.keys())

    color_palette_1 = {
        k: v for k, v in zip(model_names, color_palette.values())
    }

    cmap = sns.color_palette("flare", as_cmap=True)
    cval = np.linspace(0, 1, n_generations)
    color_palette_2 = {
        k: cmap(v) for k, v in zip(list(range(n_generations)), cval)
    }

    # ---------- Visualize Metrics ---------- #

    ############ IMPORTANT ############
    # You need to run the R script `11-1_dynamic_spectral_lmm.r` 
    # before this step in order to proceed with the visualizations.
    ###################################

    METRIC_DIR = os.path.join(DATA_DIR, "dynamic_spectral")
    if not os.path.exists(METRIC_DIR):
        raise ValueError(
            "Statistical test results do not exist. Please run the R script first."
        )

    # Load L2 distance metrics and estimated marginal means (EMMs) for PSDs
    l2_distances = pd.read_csv(f"{METRIC_DIR}/subject_means_l2_psd.csv")  # L2 distances per subject, averaged over datasets and channels
    emm_model_l2 = pd.read_csv(f"{METRIC_DIR}/emm_model_l2_psd.csv")  # estimated marginal means (EMMs) per model

    l2_distances["model"] = pd.Categorical(
        l2_distances["model"], categories=model_names, ordered=True
    )
    emm_model_l2["model"] = pd.Categorical(
        emm_model_l2["model"], categories=model_names, ordered=True
    )

    # Plot subject-level distributions of L2 distance per model for PSDs
    print("Visualizing distributions of L2 distance metrics for PSDs ...")

    up.plot_metric_violin(
        df=l2_distances,
        x="model",
        y="mean_metric",
        hue="model",
        emm_df=emm_model_l2,
        metric_name="L2 Distance",
        palette=color_palette_1,
        x_labels=x_labels,
        filename=os.path.join(fig_dir, "l2_dynamic_psds.png"),
    )

    # Print the pairwise EMM test results (post-hoc tests with multiple comparison correction)
    pairs_l2 = pd.read_csv(f"{METRIC_DIR}/pairwise_model_contrasts_l2_psd.csv")
    print("\nPairwise EMM comparisons (Tukey-adjusted):")
    print(pairs_l2[["contrast", "estimate", "SE", "z.ratio", "p.value"]])  # adjusted p-values

    # Plot model × dataset interaction for L2 distances (EMMs) of PSDs
    print("Visualizing model-dataset interaction for L2 distances of PSDs ...")
    emm_md_l2 = pd.read_csv(f"{METRIC_DIR}/emm_model_by_dataset_l2_psd.csv")
    emm_md_l2["model"] = pd.Categorical(emm_md_l2["model"], categories=model_names, ordered=True)

    up.plot_model_dataset_interaction(
        emm_df=emm_md_l2,
        metric_name="L2 Distance",
        palette=color_palette_2,
        x_labels=x_labels,
        filename=os.path.join(fig_dir, "l2_dynamic_psds_interaction.png"),
    )

    # Load L2 distance metrics and estimated marginal means (EMMs) for burst rates
    l2_distances = pd.read_csv(f"{METRIC_DIR}/subject_means_l2_sr.csv")  # L2 distances per subject, averaged over datasets
    emm_model_l2 = pd.read_csv(f"{METRIC_DIR}/emm_model_l2_sr.csv")  # estimated marginal means (EMMs) per model

    l2_distances["model"] = pd.Categorical(
        l2_distances["model"], categories=model_names, ordered=True
    )
    emm_model_l2["model"] = pd.Categorical(
        emm_model_l2["model"], categories=model_names, ordered=True
    )

    # Plot subject-level distributions of L2 distance per model for burst rates
    print("Visualizing distributions of L2 distance metrics for burst rates ...")

    up.plot_metric_violin(
        df=l2_distances,
        x="model",
        y="mean_metric",
        hue="model",
        emm_df=emm_model_l2,
        metric_name="L2 Distance",
        palette=color_palette_1,
        x_labels=x_labels,
        filename=os.path.join(fig_dir, "l2_dynamic_sr.png"),
    )

    # Print the pairwise EMM test results (post-hoc tests with multiple comparison correction)
    pairs_l2 = pd.read_csv(f"{METRIC_DIR}/pairwise_model_contrasts_l2_sr.csv")
    print("\nPairwise EMM comparisons (Tukey-adjusted):")
    print(pairs_l2[["contrast", "estimate", "SE", "z.ratio", "p.value"]])  # adjusted p-values

    # Plot model × dataset interaction for L2 distances (EMMs) of burst rates
    print("Visualizing model-dataset interaction for L2 distances of burst rates ...")
    emm_md_l2 = pd.read_csv("data/dynamic_spectral/emm_model_by_dataset_l2_sr.csv")
    emm_md_l2["model"] = pd.Categorical(emm_md_l2["model"], categories=model_names, ordered=True)

    up.plot_model_dataset_interaction(
        emm_df=emm_md_l2,
        metric_name="L2 Distance",
        palette=color_palette_2,
        x_labels=x_labels,
        filename=os.path.join(fig_dir, "l2_dynamic_sr_interaction.png"),
    )

    print("Analysis complete.")