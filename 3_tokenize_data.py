"""Script for tokenizing data using a trained tokenizer model."""

# Import packages
import os
import numpy as np
from glob import glob
from sys import argv
from tqdm.auto import tqdm
from osl_dynamics.data import Data
from osl_dynamics.inference import tf_ops
from osl_dynamics.utils import set_random_seed
from osl_foundation import load_model


if __name__ == "__main__":
    # ---------- User Inputs ---------- #
    # Set random seed for Python random, NumPy, and TensorFlow
    set_random_seed(813, op_determinism=True)

    # Set GPU memory growth
    tf_ops.gpu_growth()

    # Set user arguments
    if len(argv) != 4:
        raise ValueError("Please provide the model type, run ID, and save mode as arguments.")
    model_type = argv[1]
    run_id = int(argv[2])
    save_mode = argv[3]
    print(f"[INFO] Model type: {model_type}, Run ID: {run_id}, Save Mode: {save_mode}")

    # Set hyperparameters
    tk_seq_len = 200  # sequence length for learnable tokenizers

    # ---------- Directories ---------- #
    data_dir = "/well/win-camcan/shared/spring23/src"
    tokenizer_dir = f"models/tokenizer/{model_type}/{run_id}"
    tokenized_data_dir = f"tokenized_data/{model_type}/{run_id}"
    tokenized_data_tf_dir = f"tokenized_data_tfrecords/{model_type}/{run_id}"
    os.makedirs(tokenized_data_dir, exist_ok=True)
    os.makedirs(tokenized_data_tf_dir, exist_ok=True)

    # ---------- Load Data ---------- #
    data_files = sorted(glob(f"{data_dir}/*/sflip_parc-raw.fif"))
    data = Data(
        data_files,
        n_jobs=8,
        picks="misc",
        reject_by_annotation="omit",
        use_tfrecord=True,
        store_dir=f"tmp_{model_type}_{run_id}",
    )
    data.standardize()

    # Trim data to be multiple of tokenizer sequence length
    raw_data = data.time_series()
    for i, array in enumerate(tqdm(raw_data, desc="Trimming data")):
        n_trim = array.shape[0] % tk_seq_len
        if n_trim > 0:
            raw_data[i] = array[:-n_trim, :]

    data.delete_dir()
    data = Data(
        raw_data,
        n_jobs=8,
        use_tfrecord=True,
        store_dir=f"tmp_{model_type}_{run_id}",
    )

    if save_mode == "numpy":
        # Load tokenizer and tokenize data
        tokenizer = load_model(tokenizer_dir)
        tokenized_data = tokenizer.tokenize_data(data)

        # Save numpy arrays
        for i, token_data in enumerate(tqdm(tokenized_data, desc="Saving tokenized data")):
            np.save(
                f"{tokenized_data_dir}/x_{i:0{len(str(len(tokenized_data)))}d}",
                token_data,
            )
    elif save_mode == "tfrecord":
        # Load tokenized data
        token_files = sorted(glob(f"{tokenized_data_dir}/x_*.npy"))
        tokenized_data = []
        for file in tqdm(token_files, desc="Loading tokenized data"):
            token_data = np.load(file)
            tokenized_data.append(token_data)

        # Re-seed for TFRecord shuffling
        set_random_seed(813, op_determinism=True)
        # NOTE: This ensures same train/val split across different tokenizers and runs.

        # Save TFRecord dataset
        tokenized_data = Data(tokenized_data, n_jobs=16)
        tokenized_data.add_session_labels(
            "session_id", np.arange(tokenized_data.n_sessions), "categorical"
        )
        tokenized_data.add_extra_channel("raw_data", raw_data)
        tokenized_data.save_tfrecord_dataset(
            tfrecord_dir=tokenized_data_tf_dir,
            sequence_length=81,
            validation_split=0.1,
            overwrite=True,
        )

    # Clean up temporary data directory
    data.delete_dir()

    print("Tokenization complete.")
