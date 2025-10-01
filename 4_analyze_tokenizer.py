"""Script for the post-hoc analysis on the best tokenizer model runs."""

# Import packages
import os
import numpy as np
import pandas as pd
from glob import glob
from osl_dynamics.data import Data
from osl_dynamics.inference import tf_ops
from osl_foundation import load_model
from utils import data as ud
from utils import plotting as up


if __name__ == "__main__":
    # ---------- User Inputs ---------- #
    # Set GPU memory growth
    tf_ops.gpu_growth()

    # Set best tokenizer run IDs
    causal_id = 25
    noncausal_id = 27
    base_id = 0  # for baseline tokenizers

    # Define hyperparameters
    Fs = 250  # sampling rate (in Hz)
    n_models = 7  # number of tokenizer models

    # Set model names
    model_names = [
        "causal", "noncausal",
        "mu_transform", "mu_transform_big",
        "mu_transform_small", "mu_transform_tiny",
        "standard_quantile",
    ]
    model_ids = [causal_id, noncausal_id] + [base_id] * (n_models - 2)

    # Set whether to re-calculate metrics
    saved = False

    # ---------- Directories ---------- #
    data_dir = "/well/win-camcan/shared/spring23/src"
    model_dir = "models/tokenizer"
    plot_dir = "plots/tokenizer"

    # ---------- Training Loss (Causal & Noncausal) ---------- #
    # Get training history
    causal_loss, causal_temp = ud.get_tokenizer_history(
        f"{model_dir}/causal/{causal_id}"
    )
    noncausal_loss, noncausal_temp = ud.get_tokenizer_history(
        f"{model_dir}/noncausal/{noncausal_id}"
    )

    # Get annealed temperature
    if np.all(causal_temp == noncausal_temp):
        temperature = causal_temp
    else:
        raise ValueError("temperature not consistent across model types.")
    
    # Plot training loss
    up.plot_tokenizer_loss(
        (causal_loss, noncausal_loss),
        filepath=os.path.join(plot_dir, "training_loss.png"),
        temperature=temperature,
    )

    # ---------- Tokenizer Metrics ---------- #
    # Get data files
    data_files = sorted(glob(f"{data_dir}/*/sflip_parc-raw.fif"))

    train_idx = np.array([
        38, 57, 421, 534, 413, 146, 245, 152, 410, 139, 79, 583, 489,
        67, 218, 260, 342, 118, 372, 51, 592, 289, 598, 504, 538, 171,
        320, 137, 41, 157, 341, 596, 375, 502, 32, 590, 560, 37, 155,
        495, 142, 183, 332, 339, 353, 518, 194, 475, 93, 64,
    ]) # selected using the numpy random generator with seed=813
    test_idx = np.setdiff1d(np.arange(len(data_files)), train_idx)

    train_files = [data_files[i] for i in train_idx]
    test_files = [data_files[i] for i in test_idx]

    n_train = len(train_files)
    n_test = len(test_files)
    
    # Load data
    train_data = Data(
        train_files,
        n_jobs=8,
        picks="misc",
        use_tfrecord=True,
        reject_by_annotation="omit",
        store_dir=f"tmp_tk_train",
    )
    test_data = Data(
        test_files,
        n_jobs=8,
        picks="misc",
        use_tfrecord=True,
        reject_by_annotation="omit",
        store_dir=f"tmp_tk_test",
    )
    
    # Standardize data
    train_data.standardize()
    test_data.standardize()

    # Compute tokenizer metrics
    if saved:
        print("Loading previously saved tokenizer metrics...")

        # Load outputs
        train_pves = np.load(f"{model_dir}/train_pves.npy")
        test_pves = np.load(f"{model_dir}/test_pves.npy")

        train_counts = ud.load(f"{model_dir}/train_counts.pkl")
        test_counts = ud.load(f"{model_dir}/test_counts.pkl")
    else:
        print("Calculating tokenizer metrics...")

        train_pves, test_pves = [], []  # percentage of variance explained
        train_counts, test_counts = [], []  # total token counts

        for i, name in enumerate(model_names):
            tk_dir = f"{model_dir}/{name}/{model_ids[i]}"
            tokenizer = load_model(tk_dir)
            n_tokens = tokenizer.config.model_config.n_tokens

            # Calculate percentage of variance explained
            train_pves.append(tokenizer.get_pve(train_data))
            test_pves.append(tokenizer.get_pve(test_data))

            # Calculate total token counts
            if i < 2:  # for learnable tokenizers
                vocab = ud.load(f"{tk_dir}/vocab.pkl")
                train_count = np.array(vocab["total_token_counts"])

                test_tokens = tokenizer.tokenize_data(test_data)
                test_count = np.array(
                    [np.bincount(t.flatten(), minlength=n_tokens) for t in test_tokens]
                )
                test_count = np.sum(test_count, axis=0)

            else:  # for baseline tokenizers
                train_count = tokenizer.get_token_counts(train_data)
                test_count = tokenizer.get_token_counts(test_data)
            
            train_counts.append(train_count)
            test_counts.append(test_count)

        # Convert to numpy arrays
        train_pves = np.array(train_pves)
        test_pves = np.array(test_pves)

        # Save metrics
        np.save(f"{model_dir}/train_pves.npy", train_pves)
        np.save(f"{model_dir}/test_pves.npy", test_pves)
        # shape: (n_models, n_subjects)

        ud.save(train_counts, f"{model_dir}/train_counts.pkl")
        ud.save(test_counts, f"{model_dir}/test_counts.pkl")
        # shape: (n_models, n_tokens)

    # Get total number of tokens (using test set)
    token_nums = [len(c) for c in test_counts]

    # ---------- Visualization ---------- #
    # Set color palette
    color_palette = {
        f"Causal (n={token_nums[0]})": "#E69F00",
        f"Noncausal (n={token_nums[1]})": "#56B4E9",
        f"Mu (n={token_nums[2]})": "#009E73",
        f"Mu (n={token_nums[3]})": "#F0E442",
        f"Mu (n={token_nums[4]})": "#0072B2",
        f"Mu (n={token_nums[5]})": "#D55E00",
        f"SQ (n={token_nums[6]})": "#CC79A7",
    }

    # Plot percentage of variance explained
    pves = np.concatenate((train_pves, test_pves), axis=1)
    df = pd.DataFrame({
        "PVE": pves.flatten(),
        "Dataset": np.tile(
            [f"Train (n={n_train})"] * train_pves.shape[1] +
            [f"Test (n={n_test})"] * test_pves.shape[1],
            n_models,
        ),
        "Model": np.repeat(list(color_palette.keys()), pves.shape[1]),
    })

    up.plot_pve(
        df,
        palette=color_palette,
        filepath=os.path.join(plot_dir, "pves_comparison.png"),
        ylim=[94.7, None],
    )

    # Plot token count histograms for each tokenizer
    for name, train_c, test_c in zip(model_names, train_counts, test_counts):
        if name not in ["causal", "noncausal"]:
            # Sort token counts in descending order
            train_order = np.argsort(train_c)[::-1]
            train_c = train_c[train_order]

            test_order = np.argsort(test_c)[::-1]
            test_c = test_c[test_order]
        else:
            # Remove tokens with zero counts
            test_c = test_c[test_c > 0]

        up.plot_token_count_histogram(
            (train_c, test_c),
            filepath=os.path.join(plot_dir, f"{name}/token_count_hist.png"),
        )

    # Plot original and reconstructed signals (for single subject/channel)
    original_ts = test_data.time_series()[0]
    recon_ts = []
    for name, id in zip(model_names, model_ids):
        tk_dir = f"{model_dir}/{name}/{id}"
        token_files = sorted(glob(f"tokenized_data/{name}/{id}/x*.npy"))
        tokens = np.load(token_files[test_idx[0]])

        tokenizer = load_model(tk_dir)
        if name not in ["causal", "noncausal"]:
            tokenized_ts = tokenizer.tokenize_data(original_ts)
            recon_ts.append(tokenizer.reconstruct_data(tokenized_ts))
        else:
            tokenized_ts = tokenizer._tokenize_data(original_ts)
            recon_ts.append(tokenizer._reconstruct_data(tokenized_ts))

    up.plot_reconstructed_signals(
        original_ts, recon_ts,
        filepath=os.path.join(plot_dir, "reconstructed_signals.png"),
        sampling_frequency=Fs,
        titles=list(color_palette.keys()),
    )

    # Delete temporary data directories
    train_data.delete_dir()
    test_data.delete_dir()

    # ---------- Generalization Analysis ---------- #
    # Get task data files (Wakeman-Henson)
    task_dir = "/well/woolrich/projects/wakeman_henson/summer23/src"
    task_files = sorted(glob(f"{task_dir}/sub*_run*/sflip_parc-raw.fif"))
    n_tasks = len(task_files)
    print(f"Number of subjects in Wakeman-Henson: {n_tasks}")

    # Get files for data with different scanner (Nottingham MEGUK - CTF)
    scanner_dir = "/well/woolrich/users/olt015/Cho2025_Tokenizer/notts_mrc_meguk_glasser"
    scanner_files = sorted(glob(f"{scanner_dir}/*.npy"))
    n_scanners = len(scanner_files)
    print(f"Number of subjects in Nottingham MEGUK: {n_scanners}")

    # Load data
    task_data = Data(
        task_files,
        sampling_frequency=Fs,
        picks="misc",
        reject_by_annotation="omit",
        n_jobs=12,
    )
    task_data.standardize()

    scanner_data = Data(
        scanner_files,
        sampling_frequency=Fs,
        n_jobs=12,
    )
    scanner_data.standardize()

    # Compute percentage of variance explained
    if saved:
        print("Loading previously saved PVEs...")

        # Load outputs
        task_pves = np.load(f"{model_dir}/task_pves.npy")
        scanner_pves = np.load(f"{model_dir}/scanner_pves.npy")
    else:
        print("Calculating PVEs...")

        task_pves, scanner_pves = [], []
        for i, name in enumerate(model_names):
            tk_dir = f"{model_dir}/{name}/{model_ids[i]}"
            tokenizer = load_model(tk_dir)
            task_pves.append(tokenizer.get_pve(task_data))
            scanner_pves.append(tokenizer.get_pve(scanner_data))
        
        # Convert to numpy arrays
        task_pves = np.array(task_pves)
        scanner_pves = np.array(scanner_pves)

        # Save PVEs
        np.save(f"{model_dir}/task_pves.npy", task_pves)
        np.save(f"{model_dir}/scanner_pves.npy", scanner_pves)
        # shape: (n_models, n_subjects)

    # Plot percentage of variance explained
    pves = np.concatenate((task_pves, scanner_pves), axis=1)
    df = pd.DataFrame({
        "PVE": pves.flatten(),
        "Dataset": np.tile(
            [f"Task (n={n_tasks})"] * task_pves.shape[1] +
            [f"Scanner (n={n_scanners})"] * scanner_pves.shape[1],
            n_models,
        ),
        "Model": np.repeat(list(color_palette.keys()), pves.shape[1]),
    })

    up.plot_pve(
        df,
        palette=color_palette,
        filepath=os.path.join(plot_dir, "pves_generalization.png"),
        ylim=[94.7, None],
    )

    # Delete temporary data directories
    task_data.delete_dir()
    scanner_data.delete_dir()

    print("Tokenizer analysis complete.")
