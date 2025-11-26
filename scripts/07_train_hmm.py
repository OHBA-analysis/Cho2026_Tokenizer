"""Script for training HMMs on MEG-GPT-generated data and the real Cam-CAN data."""

# Import packages
import os
import pickle
from sys import argv
from osl_dynamics import run_pipeline
from osl_dynamics.data import Data
from osl_dynamics.inference import tf_ops
from osl_dynamics.utils import set_random_seed
from utils.plotting import plot_channel_location


if __name__ == "__main__":
    # ---------- User Inputs ---------- #
    # Set GPU memory growth
    tf_ops.gpu_growth()

    # Set user arguments
    if len(argv) < 3:
        raise ValueError(
            "Please provide the model type, channel index, (and generated data ID) as arguments."
        )
    model_type = argv[1]  # model type
    channel = int(argv[2])  # channel index

    # Channel Information: [
    #   0 (visual),
    #   4 (motor),
    #   13 (temporal),
    #   18 (parietal),
    #   22 (prefrontal),
    # ]

    if len(argv) == 4:
        gen_id = int(argv[3])  # generated data index

    gt_run_id = 1  # trained model index
    n_generations = 10  # number of data generations
    n_generated_samples = 15000  # 60 seconds (250 Hz sampling rate)

    # Set random seed for Python random, NumPy, and TensorFlow
    BASE_SEED = 813
    seed = BASE_SEED + (gen_id if len(argv) == 4 else 10) * 10_000 + channel
    set_random_seed(seed, op_determinism=False)
    # NOTE: This ensures same HMM training across different model types
    #       per channel and data generation (or real data).

    # ---------- Directories ---------- #
    BASE_DIR = "/well/woolrich/users/olt015/Cho2025_Tokenizer"
    MODEL_DIR = os.path.join(BASE_DIR, "models/generator")
    HMM_DIR = os.path.join(BASE_DIR, "models/hmm")

    generator_dir = os.path.join(MODEL_DIR, f"{model_type}/{gt_run_id}")
    hmm_dir = os.path.join(HMM_DIR, f"{model_type}/{gt_run_id}/ch{channel}")
    plot_dir = os.path.join(BASE_DIR, "plots/generator/burst_detection")
    os.makedirs(hmm_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # ---------- Load and prepare data ---------- #
    # Plot channel location
    plot_channel_location(channel, plot_dir)

    # Load data
    if model_type == "real":
        # Load real data
        with open(f"{BASE_DIR}/data/original_data.pkl", "rb") as f:
            original_data = pickle.load(f)
            # shape: (n_subjects, n_samples, n_channels)
        
        # Trim original data to match generated data length and
        # select the channels of interest
        data = [d[:n_generated_samples, [channel]] for d in original_data]
    else:
        # Load generated data
        with open(f"{generator_dir}/generated_data_{gen_id}.pkl", "rb") as f:
            generated_data = pickle.load(f)
            # shape: (n_subjects, n_generated_samples, n_channels)

        # Select the channels of interest
        data = [d[:, [channel]] for d in generated_data]

    # Prepare data
    prepare_methods = {
        "tde": {"n_embeddings": 21},
        "standardize": {},
    }
    input_data = Data(data, sampling_frequency=250, n_jobs=8)
    input_data.downsample(100)
    input_data.delete_dir()
    input_data = Data(
        input_data.time_series(), sampling_frequency=100, n_jobs=8
    )
    input_data.prepare(prepare_methods)

    # ---------- Train univariate TDE-HMM ---------- #
    # Set HMM hyperparameter configurations
    config = """
        train_hmm:
            config_kwargs:
                n_states: 3
                sequence_length: 200
                batch_size: 128
                learn_means: False
                learn_covariances: True
        multitaper_spectra:
            kwargs:
                frequency_range: [1, 45]
    """

    # Run HMM training
    if model_type != "real":
        hmm_dir = f"{hmm_dir}/{gen_id}"
    
    run_pipeline(
        config,
        output_dir=hmm_dir,
        data=input_data,
    )

    # Delete temporary data directory
    input_data.delete_dir()

    print("Training complete.")
