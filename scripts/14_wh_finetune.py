"""Script for fine-tuning pre-trained MEG-GPT models on the Wakeman-Henson task dataset."""

# Import packages
import os
import numpy as np
from sys import argv
from osl_dynamics.data import load_tfrecord_dataset
from osl_dynamics.inference import tf_ops
from osl_dynamics.utils import set_random_seed
from osl_foundation import create_model


if __name__ == "__main__":
    # ---------- User Inputs ---------- #
    # Set GPU memory growth
    tf_ops.gpu_growth()

    # Set user arguments
    if len(argv) != 6:
        raise ValueError(
            "Please provide the model type, best tokenizer run ID, pre-trained generator run ID, " +
            "decoding model run ID, and fine tuning mode as arguments."
        )
    model_type = argv[1]
    tk_run_id = int(argv[2])
    pt_run_id = int(argv[3])
    dc_run_id = int(argv[4])
    ft_mode = argv[5]
    print(f"[INFO] Model Type: {model_type} | Best Tokenizer Run ID: {tk_run_id} " + 
          f"| Pre-Trained Generator Run ID: {pt_run_id} | Decoding Model Run ID: {dc_run_id} " + 
          f"| Fine Tuning Mode: {ft_mode}")

    # Validate inputs
    if ft_mode not in ["fine_tune", "zero_shot_subject_emb"]:
        raise ValueError("Fine tuning mode must be either 'fine_tune' or 'zero_shot_subject_emb'.")

    # Set random seed for Python random, NumPy, and TensorFlow
    BASE_SEED = 813
    set_random_seed(BASE_SEED + 100 * dc_run_id, op_determinism=False)
    # NOTE: This ensures same initialization across different model types
    #       at epoch 0.

    # Set hyperparameters
    n_sessions = 6  # number of sessions per subject (19 subjects total)

    # ---------- Directories ---------- #
    BASE_DIR = "/well/woolrich/users/olt015/Cho2025_Tokenizer"
    MODEL_DIR = os.path.join(BASE_DIR, f"models/decoding_models/{ft_mode}")

    model_dir = f"{MODEL_DIR}/{model_type}/{pt_run_id}/{dc_run_id}"
    tokenized_data_dir = f"{BASE_DIR}/tokenized_data_wh/{model_type}/{tk_run_id}"
    tokenized_data_tf_dir = f"{BASE_DIR}/tokenized_data_tfrecords_wh/{model_type}/{tk_run_id}"
    os.makedirs(model_dir, exist_ok=True)

    # ---------- Build Generator ---------- #
    decoding_model = create_model(f"{model_dir}/config.yml")
    if ft_mode == "fine_tune":
        decoding_model.model.get_layer("decoder").trainable = True
        decoding_model.model.get_layer("prediction_head").trainable=True
        decoding_model.compile()
    decoding_model.summary()

    # ---------- Load Data ---------- #
    # Determine train/val split
    session_id = np.arange(1, n_sessions + 1)
    train_mask = [False] * n_sessions
    for i in range(n_sessions):
        if session_id[i] == 6:
            continue
        train_mask[i] = True
    # sessions 1-5 of all subjects are used for training
    # session 6 is used for validation
    # NOTE: We fine-tune GPT models separately for each subject.

    # Load training data
    train_data = load_tfrecord_dataset(
        f"{tokenized_data_tf_dir}/sub{dc_run_id + 1:02d}",
        batch_size=decoding_model.config.training_config.batch_size,
        shuffle=True,
        concatenate=True,
        drop_last_batch=True,
        buffer_size=2000,
        keep=list(np.where(train_mask)[0]),
    )

    # Load validation data
    val_data = load_tfrecord_dataset(
        f"{tokenized_data_tf_dir}/sub{dc_run_id + 1:02d}",
        batch_size=decoding_model.config.training_config.batch_size,
        shuffle=False,
        concatenate=True,
        drop_last_batch=True,
        buffer_size=2000,
        keep=list(np.where(np.logical_not(train_mask))[0]),
    )

    # ---------- Fine-tune pre-trained generator ---------- #
    decoding_model.fit(
        train_data,
        validation_data=val_data,
        tokenize=False,
    )

    print("Generator fine-tuning complete.")
