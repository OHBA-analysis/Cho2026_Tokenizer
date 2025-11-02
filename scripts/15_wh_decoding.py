"""Script for Wakeman-Henson task decoding."""

# Import packages
import os
import numpy as np
import pandas as pd
from glob import glob
from osl_dynamics.inference import tf_ops
from osl_foundation import load_model
from utils import analysis as ua
from utils import data as ud
from utils import plotting as up


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
    n_subjects = 19  # number of subjects in the Wakeman-Henson dataset
    n_sessions = 6  # number of sessions per subject
    tk_run_ids = [25, 27, 0, 0, 0, 0, 0]
    gt_run_id = 1  # pre-trained model run ID
    dc_run_ids = np.arange(n_subjects)
    ft_mode = "visualize"

    # Validate inputs
    if ft_mode not in ["zero_shot_subject_emb", "fine_tune", "visualize"]:
        raise ValueError(
            "Fine tuning mode must be either 'zero_shot_subject_emb', 'fine_tune', or 'visualize'."
        )

    # ---------- Directories ---------- #
    BASE_DIR = "/well/woolrich/users/olt015/Cho2025_Tokenizer"
    MODEL_DIR = os.path.join(BASE_DIR, f"models/decoding_models/{ft_mode}")
    DATA_DIR = os.path.join(BASE_DIR, "tokenized_data_fif_wh")

    # ---------- Feature Extraction & Task Decoding ---------- #
    if ft_mode in ["zero_shot_subject_emb", "fine_tune"]:
        # Create directories to save features and figures
        save_dir = f"{BASE_DIR}/data/wh_decoding/{ft_mode}"
        fig_dir = f"{BASE_DIR}/plots/decoding_models/{ft_mode}"
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(fig_dir, exist_ok=True)

        # Plot fine-tuning history
        # (i.e., training/validation loss and top 1 accuracy curves)
        for name in model_names:
            for id in dc_run_ids:
                up.plot_generator_history(
                    model_dir=f"{MODEL_DIR}/{name}/{gt_run_id}/{id}",
                    save_dir=f"{fig_dir}/{name}/{gt_run_id}/{id}",
                )

        # Get subject IDs
        subject_ids = [f"sub{i:02d}" for i in range(1, n_subjects + 1)]

        for i, name in enumerate(model_names):
            # Define save paths
            feature_save_path = f"{save_dir}/decoding_features_{name}.pkl"
            decoding_ws_save_path = f"{save_dir}/decoding_accuracies_ws_{name}.pkl"
            decoding_ns_save_path = f"{save_dir}/decoding_accuracies_ns_{name}.pkl"

            # Extract features for each model
            if not os.path.exists(feature_save_path):
                print(f"Extracting features using {name} model ...")
                
                features, labels = [], []
                n_total_sessions = 0

                for s, subject_id in enumerate(subject_ids):
                    # Get data files
                    data_files = sorted(glob(
                        f"{DATA_DIR}/{name}/{tk_run_ids[i]}/{subject_id}/*.fif"
                    ))
                    n_total_sessions += len(data_files)

                    # Load fine-tuned model
                    decoding_model = load_model(
                        f"{MODEL_DIR}/{name}/{gt_run_id}/{s}", checkpoint="latest"
                    )
                    sequence_length = decoding_model.config.model_config.sequence_length
                    print(f"\tSequence length: {sequence_length}")

                    # Extract trials and task labels
                    trials, subject_labels = ud.get_event_trials_and_labels(data_files, sequence_length)

                    # Extract features for each session
                    subject_features = ud.get_features(
                        decoding_model,
                        trials,
                        subject_ids=[0] * n_sessions,
                        batch_size=64,
                    )
                    # NOTE: We use subject_id=0, because we have different model instances 
                    # for each subject and hence one subject embedding per decoding model.
                    features.extend(subject_features)
                    labels.extend(subject_labels)

                # Verify total number of sessions
                if n_total_sessions != n_subjects * n_sessions:
                    raise ValueError(
                        f"Expected {n_subjects * n_sessions} sessions, but found {n_total_sessions} sessions."
                    )
                print(f"\tTotal number of sessions: {n_total_sessions}")

                # Save features and labels
                data_files = sorted(glob(
                    f"{DATA_DIR}/{name}/{tk_run_ids[i]}/*/*.fif"
                ))
                
                data_dict = {}
                for file, feature, label in zip(data_files, features, labels):
                    session_id = file.split("/")[-1]
                    data_dict[session_id] = (feature, label)

                ud.save(data_dict, feature_save_path)

            else:
                print(f"\tFeatures for {name} model already exist. Skipping extraction.")

            # Compute session-wise decoding accuracy for each model
            if not os.path.exists(decoding_ws_save_path) or not os.path.exists(decoding_ns_save_path):
                print(f"Computing decoding accuracy for {name} model ...")
                
                decoding_accuracy_ws = ua.compute_task_decoding_accuracy(
                    feature_save_path, test_session="run06"
                )
                print(f"\tDecoding accuracy (within subject): {decoding_accuracy_ws}")
                print(f"\tShape: {decoding_accuracy_ws.shape}")

                decoding_accuracy_ns = ua.compute_task_decoding_accuracy(
                    feature_save_path, test_subject="sub19"
                )
                print(f"\tDecoding accuracy (new subject): {decoding_accuracy_ns}")
                print(f"\tShape: {decoding_accuracy_ns.shape}")

                # Save decoding accuracy
                ud.save(decoding_accuracy_ws, decoding_ws_save_path)
                ud.save(decoding_accuracy_ns, decoding_ns_save_path)

            else:
                print(f"\tDecoding accuracies for {name} model already exist. Loading accuracies.")

    # ---------- Visualization ---------- #
    if ft_mode == "visualize":
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
        acc_zs_ws, acc_ft_ws = {}, {}  # for within subject decoding
        acc_zs_ns, acc_ft_ns = {}, {}  # for new subject decoding
        for name in model_names:
            save_dir = f"{BASE_DIR}/data/wh_decoding"
            ws_path = f"{save_dir}/{{0}}/decoding_accuracies_ws_{{1}}.pkl"
            ns_path = f"{save_dir}/{{0}}/decoding_accuracies_ns_{{1}}.pkl"

            acc_zs_ws[name] = ud.load(ws_path.format("zero_shot_subject_emb", name))
            acc_ft_ws[name] = ud.load(ws_path.format("fine_tune", name))

            acc_zs_ns[name] = ud.load(ns_path.format("zero_shot_subject_emb", name))
            acc_ft_ns[name] = ud.load(ns_path.format("fine_tune", name))

        # Build dataframe
        dict_to_df = lambda d: pd.DataFrame.from_dict(d).melt(var_name="Model", value_name="Accuracy")

        df_acc_zs_ws = dict_to_df(acc_zs_ws)
        df_acc_ft_ws = dict_to_df(acc_ft_ws)

        df_acc_zs_ns = dict_to_df(acc_zs_ns)
        df_acc_ft_ns = dict_to_df(acc_ft_ns)

        # Visualize bar plots for decoding accuracies
        up.plot_decoding_bars(
            df_acc_zs_ws,
            mode="Zero-Shot",
            palette=color_palette_1,
            filename=f"acc_zs_ws.png",
            ylim=[0.0, 0.8],
        )

        up.plot_decoding_bars(
            df_acc_ft_ws,
            mode="Fine-Tuned",
            palette=color_palette_1,
            filename=f"acc_ft_ws.png",
            ylim=[0.0, 0.8],
        )

        up.plot_decoding_bars(
            df_acc_zs_ns,
            mode="Zero-Shot",
            palette=color_palette_1,
            filename=f"acc_zs_ns.png",
            ylim=[0.0, 0.8],
        )

        up.plot_decoding_bars(
            df_acc_ft_ns,
            mode="Fine-Tuned",
            palette=color_palette_1,
            filename=f"acc_ft_ns.png",
            ylim=[0.0, 0.8],
        )

    # ---------- Statistical Testing ---------- #

    print("Decoding completed.")
