"""Script for Wakeman-Henson task decoding."""

# Import packages
import gc
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from glob import glob
from osl_dynamics.inference import tf_ops
from osl_foundation import create_model
from utils import analysis as ua
from utils import data as ud
from utils import plotting as up
from utils import statistics as us


if __name__ == "__main__":
    # ---------- User Inputs ---------- #
    # Set GPU memory growth
    tf_ops.gpu_growth()

    # Set hyperparameters
    model_names = [
        "causal", "noncausal",
        "mu_transform", "mu_transform_big",
        "mu_transform_small", "mu_transform_tiny",
        "standard_quantile",
    ]
    tk_run_ids = [25, 27, 0, 0, 0, 0, 0]
    gt_run_id = 1  # pre-trained model run ID
    ft_mode = "visualize"

    n_subjects = 19  # number of subjects in the Wakeman-Henson dataset
    n_sessions = 6  # number of sessions per subject
    sequence_length = 80  # sequence length for task trials

    # Validate inputs
    if ft_mode not in ["baseline", "zero_shot", "fine_tune", "visualize"]:
        raise ValueError(
            "Fine tuning mode must be either 'baseline', " +
            "'zero_shot', 'fine_tune', or 'visualize'."
        )

    # Define random seed for Python random, NumPy, and TensorFlow
    BASE_SEED = 813

    # ---------- Directories ---------- #
    BASE_DIR = "/well/woolrich/users/olt015/Cho2026_Tokenizer"
    PROJ_DIR = "/well/woolrich/projects/wakeman_henson/summer23/src"
    MODEL_DIR = os.path.join(BASE_DIR, f"models/decoding_models/{ft_mode}")
    DATA_DIR = os.path.join(BASE_DIR, "tokenized_data_fif_wh")

    # ---------- Feature Extraction & Task Decoding ---------- #
    if ft_mode != "visualize":
        # Create directories to save features and figures
        save_dir = f"{BASE_DIR}/data/wh_decoding/{ft_mode}"
        os.makedirs(save_dir, exist_ok=True)

        # Get subject IDs
        subject_ids = [f"sub{i:02d}" for i in range(1, n_subjects + 1)]

        # Extract features
        if ft_mode == "baseline":
            # Define save paths
            decoding_ws_save_path = f"{save_dir}/decoding_accuracies_ws.pkl"
            decoding_ns_save_path = f"{save_dir}/decoding_accuracies_ns.pkl"

            # Compute session-wise decoding accuracy for each model
            if not os.path.exists(decoding_ws_save_path) or not os.path.exists(decoding_ns_save_path):
                print("Extracting features using baseline model ...")

                # Get data files
                data_files = sorted(glob(
                    f"{PROJ_DIR}/sub*_run*/sflip_parc-raw.fif"
                ))
                n_total_sessions = len(data_files)

                # Extract trials and task labels
                features, labels = ud.get_event_trials_and_labels(
                    data_files, sequence_length=80
                )
                # NOTE: We use parcel MEG time courses as our features for the baseline model.

                # Verify total number of sessions
                if n_total_sessions != n_subjects * n_sessions:
                    raise ValueError(
                        f"Expected {n_subjects * n_sessions} sessions, but found {n_total_sessions} sessions."
                    )
                print(f"\tTotal number of sessions: {n_total_sessions}")

                # Save features and labels              
                data_dict = {}
                for file, feature, label in zip(data_files, features, labels):
                    session_id = file.split("/")[-2]
                    data_dict[session_id] = (feature, label)

                # Compute decoding accuracy
                print(f"Computing decoding accuracy for baseline model ...")
                
                decoding_accuracy_ws = ua.compute_task_decoding_accuracy(
                    data_dict,
                    config_path=f"{MODEL_DIR}/within_subject/config.yml",
                    test_session="run06",
                    seed=BASE_SEED,
                )
                print(f"\tDecoding accuracy (within subject): {decoding_accuracy_ws}")
                print(f"\tShape: {decoding_accuracy_ws.shape}")

                decoding_accuracy_ns = ua.compute_task_decoding_accuracy(
                    data_dict,
                    config_path=f"{MODEL_DIR}/new_subject/config.yml",
                    test_subject="sub19",
                    seed=BASE_SEED,
                )
                print(f"\tDecoding accuracy (new subject): {decoding_accuracy_ns}")
                print(f"\tShape: {decoding_accuracy_ns.shape}")

                # Save decoding accuracy
                ud.save(decoding_accuracy_ws, decoding_ws_save_path)
                ud.save(decoding_accuracy_ns, decoding_ns_save_path)

            else:
                print(f"\tDecoding accuracies for the baseline model already exist. Loading accuracies.")

        else:
            for i, name in enumerate(model_names):
                # Define save paths
                decoding_ws_save_path = f"{save_dir}/decoding_accuracies_ws_{name}.pkl"
                decoding_ns_save_path = f"{save_dir}/decoding_accuracies_ns_{name}.pkl"

                # Compute session-wise decoding accuracy for each model
                if not os.path.exists(decoding_ws_save_path) or not os.path.exists(decoding_ns_save_path):
                    print(f"Extracting features using {name} model ...")

                    # Get data files
                    data_files = sorted(glob(f"{DATA_DIR}/{name}/{tk_run_ids[i]}/*.fif"))
                    n_total_sessions = len(data_files)

                    # Verify total number of sessions
                    if n_total_sessions != n_subjects * n_sessions:
                        raise ValueError(
                            f"Expected {n_subjects * n_sessions} sessions, but found {n_total_sessions} sessions."
                        )
                    print(f"\tTotal number of sessions: {n_total_sessions}")

                    # Load models
                    if ft_mode == "zero_shot":
                        model_path = f"{MODEL_DIR}/{name}/{gt_run_id}"
                        decoding_model = create_model(f"{model_path}/config.yml")

                    # Extract task event trials and labels
                    task_trials, task_labels = ud.get_event_trials_and_labels(
                        data_files, sequence_length=sequence_length,
                    )
                    # task_trials.shape: (n_sessions, n_trials, n_samples, n_channels)
                    # task_labels.shape: (n_sessions, n_trials)

                    # Extract features for each session
                    if ft_mode == "fine_tune":
                        task_features = task_trials
                    else:
                        task_features = ud.get_features(
                            decoding_model,
                            task_trials,
                            subject_ids=[None] * n_sessions * n_subjects,
                            batch_size=64,
                        )
                        # task_features.shape: (n_sessions, n_trials, n_samples, n_channels, model_dim)

                        # Clear previous model and secure memory
                        del decoding_model
                        tf.keras.backend.clear_session()
                        gc.collect()

                    # Save features and labels
                    data_dict = {}
                    for file, feature, label in zip(data_files, task_features, task_labels):
                        session_id = file.split("/")[-1]
                        data_dict[session_id] = (feature, label)

                    # Compute decoding accuracy
                    print(f"Computing decoding accuracy for {name} model ...")
                    
                    if ft_mode == "fine_tune":
                        decoding_accuracy_ws = ua.evaluate_fine_tuned_model(
                            data_dict,
                            config_dir=f"{MODEL_DIR}/within_subject/{name}/{gt_run_id}",
                            test_session="run06",
                        )
                        decoding_accuracy_ns = ua.evaluate_fine_tuned_model(
                            data_dict,
                            config_dir=f"{MODEL_DIR}/new_subject/{name}/{gt_run_id}",
                            test_subject="sub19",
                        )
                    else:
                        decoding_accuracy_ws = ua.compute_task_decoding_accuracy(
                            data_dict,
                            config_path=f"{model_path}/within_subject/config.yml",
                            test_session="run06",
                            seed=BASE_SEED,
                            use_tfrecord=True,
                            save_dir=f"{model_path}/within_subject/tfrecords"
                        )
                        decoding_accuracy_ns = ua.compute_task_decoding_accuracy(
                            data_dict,
                            config_path=f"{model_path}/new_subject/config.yml",
                            test_subject="sub19",
                            seed=BASE_SEED,
                            use_tfrecord=True,
                            save_dir=f"{model_path}/new_subject/tfrecords"
                        )

                    print(f"\tDecoding accuracy (within subject): {decoding_accuracy_ws}")
                    print(f"\tShape: {decoding_accuracy_ws.shape}")

                    print(f"\tDecoding accuracy (new subject): {decoding_accuracy_ns}")
                    print(f"\tShape: {decoding_accuracy_ns.shape}")

                    # Save decoding accuracy
                    ud.save(decoding_accuracy_ws, decoding_ws_save_path)
                    ud.save(decoding_accuracy_ns, decoding_ns_save_path)

                else:
                    print(f"\tDecoding accuracies for {name} model already exist. Skipping computations.")

    # ---------- Visualization ---------- #
    if ft_mode == "visualize":
        # Plot model training history
        # (i.e., training/validation loss and top 1 accuracy curves)
        for mode in ["baseline", "zero_shot", "fine_tune"]:
            for task_type in ["within_subject", "new_subject"]:
                model_dir = f"{BASE_DIR}/models/decoding_models/{mode}"
                save_dir = f"{BASE_DIR}/plots/decoding_models/{mode}"

                if mode == "baseline":
                    up.plot_generator_history(
                        model_dir=f"{model_dir}/{task_type}",
                        save_dir=f"{save_dir}/{task_type}",
                    )
                else:
                    for name in model_names:
                        up.plot_generator_history(
                            model_dir=(
                                f"{model_dir}/{name}/{gt_run_id}/{task_type}" if mode != "fine_tune"
                                else f"{model_dir}/{task_type}/{name}/{gt_run_id}"
                            ),
                            save_dir=(
                                f"{save_dir}/{name}/{gt_run_id}/{task_type}" if mode != "fine_tune"
                                else f"{save_dir}/{task_type}/{name}/{gt_run_id}"
                            ),
                        )

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

        color_palette_1 = {
            k: v for k, v in zip(model_names, color_palette.values())
        }

        # Load decoding accuracies
        acc_base_ws, acc_zs_ws, acc_zs_se_ws, acc_ft_ws = {}, {}, {}, {}  # for within subject decoding
        acc_base_ns, acc_zs_ns, acc_zs_se_ns, acc_ft_ns = {}, {}, {}, {}  # for new subject decoding

        save_dir = f"{BASE_DIR}/data/wh_decoding"

        acc_base_ws["baseline"] = ud.load(f"{save_dir}/baseline/decoding_accuracies_ws.pkl")
        acc_base_ns["baseline"] = ud.load(f"{save_dir}/baseline/decoding_accuracies_ns.pkl")

        for name in model_names:
            ws_path = f"{save_dir}/{{0}}/decoding_accuracies_ws_{{1}}.pkl"
            ns_path = f"{save_dir}/{{0}}/decoding_accuracies_ns_{{1}}.pkl"

            acc_zs_ws[name] = ud.load(ws_path.format("zero_shot", name))
            acc_ft_ws[name] = ud.load(ws_path.format("fine_tune", name))

            acc_zs_ns[name] = ud.load(ns_path.format("zero_shot", name))
            acc_ft_ns[name] = ud.load(ns_path.format("fine_tune", name))

        # Build dataframe
        dict_to_df = lambda d: pd.DataFrame.from_dict(d).melt(var_name="Model", value_name="Accuracy")

        df_acc_base_ws = dict_to_df(acc_base_ws)
        df_acc_zs_ws = dict_to_df(acc_zs_ws)
        df_acc_ft_ws = dict_to_df(acc_ft_ws)

        df_acc_base_ns = dict_to_df(acc_base_ns)
        df_acc_zs_ns = dict_to_df(acc_zs_ns)
        df_acc_ft_ns = dict_to_df(acc_ft_ns)

        # Visualize bar plots for decoding accuracies
        dfs = [df_acc_base_ws, df_acc_base_ns]
        filenames = ["acc_base_ws.png", "acc_base_ns.png"]

        for df, filename in zip(dfs, filenames):
            up.plot_decoding_bars(
                df,
                mode="Baseline",
                palette={"baseline": "#787878FF"},
                filename=filename,
                ylim=[0.0, 0.8],
            )

        dfs = [df_acc_zs_ws, df_acc_ft_ws, df_acc_zs_ns, df_acc_ft_ns]
        modes = ["Zero-Shot", "Fine-Tuned"] * 2
        filenames = [
            "acc_zs_ws.png", "acc_ft_ws.png", "acc_zs_ns.png", "acc_ft_ns.png"
        ]

        for df, mode, filename in zip(dfs, modes, filenames):
            up.plot_decoding_bars(
                df,
                mode=mode,
                palette=color_palette_1,
                filename=filename,
                ylim=[0.0, 0.8],
            )

    # ---------- Statistical Testing ---------- #
    # Get pairwise combinations
    pairs = [(i, i + 1) for i in range(len(model_names) - 1)]

    # Set threshold
    alpha = 0.05
    n_tests = len(pairs)
    print(f"Number of tests: {n_tests}")

    # Perform statistical tests
    for name, df in zip(
        ["Within-Subject Zero-Shot", "New Subject Zero-Shot",
         "Within-Subject Fine-Tuned", "New Subject Fine-Tuned"],
        [df_acc_zs_ws, df_acc_zs_ns,
         df_acc_ft_ws, df_acc_ft_ns],
    ):
        # Reorder metrics by performance
        mean_accuracies = df.groupby("Model")["Accuracy"].mean()
        order = mean_accuracies.sort_values(ascending=False).index.tolist()
        order = [model_names.index(model) for model in order]

        print("\nStatistical Analysis for", name)
        mod_names = [model_names[i] for i in order]  # reorder model names
        for i, j in pairs:
            print(f"{mod_names[i].title()} vs {mod_names[j].title()}")
            samples1 = df[df["Model"] == mod_names[i]]["Accuracy"].values
            samples2 = df[df["Model"] == mod_names[j]]["Accuracy"].values
            stat, pval, sig_indicator = us.stat_ind_two_samples(
                samples1,
                samples2,
                alpha=alpha,
                bonferroni_ntest=n_tests,
            )

    print("Decoding completed.")
