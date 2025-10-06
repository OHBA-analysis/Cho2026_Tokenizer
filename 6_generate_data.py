"""Script for saving the original and generated data."""

# Import packages
import os
import math
import pickle
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
    # Set GPU memory growth
    tf_ops.gpu_growth()

    # Set user arguments
    if len(argv) != 5:
        raise ValueError(
            "Please provide the model type, generator run ID, generation ID, " + 
            "and batch number as arguments."
        )
    model_type = argv[1]
    run_id = int(argv[2])
    gen_id = int(argv[3])
    batch_id = int(argv[4])  # if -1, combines existing batches
    print(f"[INFO] Model type: {model_type} | Generator Run ID: {run_id}, " + 
          f"Generation #: {gen_id}, Batch ID: {batch_id}")

    # Define generation parameters
    n_sessions = 612  # total number of sessions (subjects) to generate
    batch_size = 64  # number of sessions per batch
    n_batches = math.ceil(n_sessions / batch_size)  # total number of batches

    # Set random seed for Python random, NumPy, TensorFlow, and TFP sampling
    if batch_id >= 0:
        BASE_SEED = 813
        seed = BASE_SEED + gen_id * n_batches + batch_id   # unique per (gen_id, batch_id)
        set_random_seed(seed, op_determinism=False)
        # NOTE: This ensures same data generation across different model types.

    # ---------- Directories ---------- #
    BASE_DIR = "/well/woolrich/users/olt015/Cho2025_Tokenizer"
    MODEL_DIR = os.path.join(BASE_DIR, "models/generator")

    data_dir = "/well/win-camcan/shared/spring23/src"
    generator_dir = os.path.join(MODEL_DIR, f"{model_type}/{run_id}")
    save_dir = os.path.join(BASE_DIR, "data")
    os.makedirs(save_dir, exist_ok=True)

    # ---------- Load and save data ---------- #
    org_data_path = f"{save_dir}/original_data.pkl"

    if os.path.exists(org_data_path):
        print(f"Original data already exists at {org_data_path}. Skipping saving.")
    else:
        print("Saving original data ...")
        
        data_files = sorted(glob(f"{data_dir}/*/sflip_parc-raw.fif"))
        data = Data(
            data_files,
            picks="misc",
            reject_by_annotation="omit",
            use_tfrecord=True,
            n_jobs=16,
            store_dir=f"tmp_{model_type}_{run_id}",
        )
        data.standardize()
        original_data = data.time_series()
        data.delete_dir()

        with open(f"{save_dir}/original_data.pkl", "wb") as f:
            pickle.dump(original_data, f)

    # ---------- Load generator ---------- #
    generator = load_model(generator_dir, checkpoint="latest")

    # ---------- Generate data using the generator ---------- #
    if batch_id >= 0:
        # Define session labels
        session_labels = np.array_split(
            np.arange(n_sessions), n_batches
        )  # shape: (batch_number, batch_size); list of arrays

        # Specify which batch to process
        batch_number = len(session_labels)
        if batch_id >= batch_number:
            raise ValueError(f"Batch number must be less than {batch_number}.")
        session_labels = session_labels[batch_id]  # shape: (batch_size,)

        # Generate and save data
        print("Generating data ...")
        save_path = f"{generator_dir}/generated_data_{gen_id}_batch{batch_id}.pkl"

        generate_data = generator.generate_data(
            n_samples=15000,  # 60 seconds (250 Hz sampling rate)
            top_p=0.99,
            batch_size=len(session_labels),
            extra_labels={"session_id": session_labels},
            seed=seed,
        )
        with open(save_path, "wb") as f:
            pickle.dump(generate_data, f)
    else:
        # Combine existing batches
        print("Combining existing batches ...")
        save_path = f"{generator_dir}/generated_data_{gen_id}.pkl"
        
        combined_data = []
        batch_files = sorted(glob(f"{generator_dir}/generated_data_{gen_id}_batch*.pkl"))
        for batch_file in tqdm(batch_files):
            with open(batch_file, "rb") as f:
                batch_data = pickle.load(f)
            combined_data.extend(batch_data)
            # os.remove(batch_file)

        if not isinstance(combined_data, list):
            raise ValueError("The combined data is not a list.")

        if len(combined_data) != n_sessions:
            raise ValueError("The number of combined sessions does not match n_sessions.")
        
        with open(save_path, "wb") as f:
            pickle.dump(combined_data, f)

    print("Data saving complete.")
