"""Script for training a tokenizer on the subset of Cam-CAN data."""

# Import packages
import os
import numpy as np
from glob import glob
from sys import argv
from osl_dynamics.data import Data
from osl_dynamics.inference import tf_ops
from osl_foundation import create_model, load_model


if __name__ == "__main__":
    # ---------- User Inputs ---------- #
    # Set GPU memory growth
    tf_ops.gpu_growth()

    # Set user arguments
    if len(argv) != 3:
        raise ValueError("Please provide the model type and run ID as arguments.")
    model_type = argv[1]
    run_id = int(argv[2])
    print(f"[INFO] Model type: {model_type}, Run ID: {run_id}")

    # ---------- Directories ---------- #
    data_dir = "/well/win-camcan/shared/spring23/src"
    plot_dir = f"plots/tokenizer/{model_type}/{run_id}"
    tokenizer_dir = f"models/tokenizer/{model_type}/{run_id}"
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(tokenizer_dir, exist_ok=True)

    # ---------- Load Data ---------- #
    # Select data subset
    train_idx = np.array([
        38, 57, 421, 534, 413, 146, 245, 152, 410, 139, 79, 583, 489,
        67, 218, 260, 342, 118, 372, 51, 592, 289, 598, 504, 538, 171,
       320, 137, 41, 157, 341, 596, 375, 502, 32, 590, 560, 37, 155,
       495, 142, 183, 332, 339, 353, 518, 194, 475, 93, 64,
    ]) # selected using the numpy random generator with seed=813
    print(f"Number of training subjects: {len(train_idx)}")

    # Load training data
    data_files = sorted(glob(f"{data_dir}/*/sflip_parc-raw.fif"))
    data_files = [data_files[i] for i in train_idx]

    data = Data(
        data_files,
        n_jobs=8,
        picks="misc",
        reject_by_annotation="omit",
        use_tfrecord=True,
        store_dir=f"tmp_{model_type}_{run_id}",
    )
    data.standardize()

    # ---------- Build Tokenizer ---------- #
    # Set model hyperparameters
    if model_type in ["causal", "noncausal"]:

        n_epochs = 10

        if model_type == "noncausal":
            token_kernel_padding = "same"
        if model_type == "causal":
            token_kernel_padding = "causal"

        config = f"""
            model_config:
                name: osl_tokenizer
                sequence_length: 200
                n_channels: 52
                n_tokens: 128
                token_dim: 10
                token_kernel_padding: {token_kernel_padding}
                rnn_n_units: 128
            training_config:
                optimizer:
                    learning_rate: 1.0e-05
                batch_size: 32
                n_epochs: {n_epochs}
                temperature_annealing:
                    n_stages: {n_epochs}
                    n_annealing_epochs: {n_epochs}
                    end_temperature: 0.0
        """
    elif model_type == "standard_quantile":
        config = f"""
            model_config:
                name: standard_quantile_tokenizer
                n_tokens: 108
                standardize: False
        """
        # NOTE: data already standardized
    else:
        if model_type == "mu_transform":
            n_tokens = 256
        if model_type == "mu_transform_big":
            n_tokens = 182
        if model_type == "mu_transform_small":
            n_tokens = 108
        if model_type == "mu_transform_tiny":
            n_tokens = 54

        config = f"""
            model_config:
                name: mu_transform_tokenizer
                n_tokens: {n_tokens}
                mu: {n_tokens - 1}
                normalization: max_abs
        """

    # Build model
    if os.path.exists(os.path.join(tokenizer_dir, "config.yml")):
        print(f"Loading existing tokenizer from {tokenizer_dir} ...")
        tokenizer = load_model(tokenizer_dir)
    else:
        tokenizer = create_model(config)
    
    if model_type in ["causal", "noncausal"]:
        tokenizer.summary()
    
    # ---------- Fit and Save Tokenizer ---------- #
    if not os.path.exists(os.path.join(tokenizer_dir, "config.yml")):
        # Fit tokenizer
        if model_type in ["causal", "noncausal"]:
            tokenizer.fit(data)
        else:
            # If tokenizer is not learnable, clip the data
            tokenizer.fit(data, clip=4)  # clip to 4 standard deviations

        # Save tokenizer
        tokenizer.save(tokenizer_dir)

    # ---------- Summary Plots ---------- #
    # Plot percentage of explained variance
    tokenizer.plot_pve(data=data, plot_dir=plot_dir)

    # Plot token counts and stimulus response of token kernels
    if model_type in ["causal", "noncausal"]:
        tokenizer.plot_token_counts(plot_dir=plot_dir)
        tokenizer.plot_token_response(plot_dir=plot_dir)
    else:
        tokenizer.plot_token_counts(data=data, plot_dir=plot_dir)
    
    # Plot signals reconstructed from tokenized data (for one session)
    tokenizer.plot_fitted_signal(
        data_path=data.time_series()[0],
        plot_dir=plot_dir,
    )

    # Clean up temporary data directory
    data.delete_dir()

    print("Tokenizer training complete.")
