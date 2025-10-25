"""Script for tokenizing Wakeman-Henson task dataset using a pre-trained tokenizer model."""

# Import packages
import os
import numpy as np
from glob import glob
from sys import argv
from tqdm.auto import tqdm
from osl_dynamics.data import Data
from osl_dynamics.inference import tf_ops, modes
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
        raise ValueError(
            "Please provide the model type, best tokenizer run ID, and save mode as arguments."
        )
    model_type = argv[1]
    tk_run_id = int(argv[2])
    save_mode = argv[3]
    print(f"[INFO] Model type: {model_type}, Tokenizer Run ID: {tk_run_id}, Save Mode: {save_mode}")

    # Validate inputs
    if save_mode not in ["numpy", "fif", "tfrecord"]:
        raise ValueError("Save mode must be either 'numpy', 'fif', or 'tfrecord'.")

    # Set hyperparameters
    tk_seq_len = 200  # sequence length for learnable tokenizers
    Fs = 250  # sampling frequency
    n_sessions = 6  # number of sessions per subject (19 subjects in total)

    # ---------- Directories ---------- #
    BASE_DIR = "/well/woolrich/users/olt015/Cho2025_Tokenizer"
    DATA_DIR = "/well/woolrich/projects/wakeman_henson/summer23/src"

    tokenizer_dir = f"{BASE_DIR}/models/tokenizer/{model_type}/{tk_run_id}"
    tokenized_data_dir = f"{BASE_DIR}/tokenized_data_wh/{model_type}/{tk_run_id}"
    tokenized_data_fif_dir = f"{BASE_DIR}/tokenized_data_fif_wh/{model_type}/{tk_run_id}"
    tokenized_data_tf_dir = f"{BASE_DIR}/tokenized_data_tfrecords_wh/{model_type}/{tk_run_id}"
    os.makedirs(tokenized_data_dir, exist_ok=True)
    os.makedirs(tokenized_data_fif_dir, exist_ok=True)
    os.makedirs(tokenized_data_tf_dir, exist_ok=True)

    # ---------- Load Data ---------- #
    data_files = sorted(glob(f"{DATA_DIR}/sub*_run*/sflip_parc-raw.fif"))
    data = Data(
        data_files,
        sampling_frequency=Fs,
        picks="misc",
        reject_by_annotation="omit",
        use_tfrecord=True,
        store_dir=f"tmp_wh_{model_type}_{tk_run_id}",
        n_jobs=12,
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
        sampling_frequency=Fs,
        use_tfrecord=True,
        store_dir=f"tmp_wh_{model_type}_{tk_run_id}",
        n_jobs=12,
    )

    # ---------- Tokenize Data ---------- #
    if save_mode == "numpy":
        # Load tokenizer and tokenize data
        tokenizer = load_model(tokenizer_dir)
        tokenized_data = tokenizer.tokenize_data(data)

        # Save numpy arrays
        for i, token_data in enumerate(tqdm(tokenized_data, desc="Saving tokenized data")):
            session_id = data_files[i].split("/")[-2]
            subject_id = session_id.split("_")[0]
            os.makedirs(f"{tokenized_data_dir}/{subject_id}", exist_ok=True)
            np.save(
                f"{tokenized_data_dir}/{subject_id}/{session_id}.npy",
                token_data,
            )

    elif save_mode == "fif":
        # Load tokenizer and tokenize data
        tokenizer = load_model(tokenizer_dir)
        tokenized_data = tokenizer.tokenize_data(data)

        # Save FIF files
        tokenized_data = Data(
            tokenized_data,
            store_dir=f"tmp_fif_{model_type}_{tk_run_id}",
            n_jobs=16,
        )
        tokenized_ts = tokenized_data.time_series()

        for ts, file in zip(tokenized_ts, data_files):
            session_id = file.split("/")[-2]
            subject_id = session_id.split("_")[0]
            os.makedirs(f"{tokenized_data_fif_dir}/{subject_id}", exist_ok=True)
            raw = modes.convert_to_mne_raw(ts, file)
            raw.save(
                f"{tokenized_data_fif_dir}/{subject_id}/{session_id}_raw.fif", overwrite=True
            )

        # Clean up temporary data directory
        tokenized_data.delete_dir()

    elif save_mode == "tfrecord":
        # Re-seed for TFRecord shuffling (safeguard)
        set_random_seed(813, op_determinism=True)

        # Get unique subject IDs
        subject_ids = sorted(list(set(
            [file.split("/")[-2].split("_")[0] for file in data_files]
        )))

        for i, subject_id in enumerate(subject_ids):
            # Create subject-specific directory
            os.makedirs(f"{tokenized_data_tf_dir}/{subject_id}", exist_ok=True)

            # Load tokenized data
            token_files = sorted(glob(f"{tokenized_data_dir}/{subject_id}/*.npy"))
            tokenized_data = []
            for file in tqdm(token_files, desc="Loading tokenized data"):
                token_data = np.load(file)
                tokenized_data.append(token_data)

            # Save TFRecord dataset
            tokenized_data = Data(tokenized_data, n_jobs=16)
            tokenized_data.add_session_labels(
                "session_id",
                np.zeros((n_sessions,), dtype=np.int32),
                "categorical",
            )  # NOTE: While the labels are called session IDs here, they correspond
               #       to subject labels for each session.
            tokenized_data.add_extra_channel(
                "raw_data", raw_data[i * n_sessions:(i + 1) * n_sessions]
            )
            tokenized_data.save_tfrecord_dataset(
                tfrecord_dir=f"{tokenized_data_tf_dir}/{subject_id}",
                sequence_length=81,
                overwrite=True,
            )

    # Clean up temporary data directory
    data.delete_dir()

    print("Tokenization complete.")
