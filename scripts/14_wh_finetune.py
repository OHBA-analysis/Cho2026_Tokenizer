"""Script for fine-tuning pre-trained MEG-GPT models on the Wakeman-Henson task dataset."""

# Import packages
import os
import numpy as np
import tensorflow as tf
from sys import argv
from glob import glob
from osl_dynamics.data import load_tfrecord_dataset
from osl_dynamics.inference import tf_ops
from osl_dynamics.utils import set_random_seed
from models import create_model
from utils import data as ud


if __name__ == "__main__":
    # ---------- User Inputs ---------- #
    # Set GPU memory growth
    tf_ops.gpu_growth()

    # Set user arguments
    if len(argv) != 5:
        raise ValueError(
            "Please provide the model type, best tokenizer run ID, pre-trained generator run ID, " +
            "and fine tuning mode as arguments."
        )
    model_type = argv[1]
    tk_run_id = int(argv[2])
    pt_run_id = int(argv[3])
    ft_mode = argv[4]
    print(f"[INFO] Model Type: {model_type} | Best Tokenizer Run ID: {tk_run_id} " + 
          f"| Pre-Trained Generator Run ID: {pt_run_id} | Fine Tuning Mode: {ft_mode}")

    # Validate inputs
    if ft_mode not in ["subject_emb", "within_subject", "new_subject"]:
        raise ValueError("Fine tuning mode must be either 'subject_emb', 'within_subject' or 'new_subject'.")
    # NOTE: Three different fine-tuning modes are supported:
    #       1) 'subject_emb': learn subject embeddings for the dataset to be used during zero-shot learning and fine-tuning
    #       2) 'within_subject': fine-tune on first 5 sessions of each subject, validate on 6th session
    #       3) 'new_subject': fine-tune on all sessions of subjects 1-18, validate on subject 19

    # Set random seed for Python random, NumPy, and TensorFlow
    BASE_SEED = 813
    set_random_seed(BASE_SEED, op_determinism=False)
    # NOTE: This ensures same initialization across different model types
    #       at epoch 0.

    # Set hyperparameters
    n_subjects = 19  # number of subjects
    n_sessions = 6  # number of sessions per subject (19 subjects total)

    # ---------- Directories ---------- #
    BASE_DIR = "/well/woolrich/users/olt015/Cho2025_Tokenizer"
    PROJ_DIR = "/well/woolrich/projects/wakeman_henson/summer23/src"
    MODEL_DIR = os.path.join(BASE_DIR, f"models/decoding_models/fine_tune/{ft_mode}")

    model_dir = f"{MODEL_DIR}/{model_type}/{pt_run_id}"
    tokenized_data_tf_dir = f"{BASE_DIR}/tokenized_data_tfrecords_wh/{model_type}/{tk_run_id}"
    tokenized_data_fif_dir = f"{BASE_DIR}/tokenized_data_fif_wh/{model_type}/{tk_run_id}"
    os.makedirs(model_dir, exist_ok=True)

    # ---------- Build Generator ---------- #
    # Build the decoding model
    decoding_model = create_model(f"{model_dir}/config.yml")
    if ft_mode != "subject_emb":
        decoding_model.model.get_layer("decoder").trainable = True
        decoding_model.compile()  # recompile the model
    
    # Print out model architecture
    decoding_model.summary()

    # ---------- Load Data ---------- #
    # Determine train/val split
    session_id = np.tile(np.arange(1, n_sessions + 1), n_subjects)
    subject_id = np.repeat(np.arange(1, n_subjects + 1), n_sessions)

    if ft_mode in ["subject_emb"]:
        train_mask = [False] * n_sessions * n_subjects
        for i in range(len(train_mask)):
            if session_id[i] == 6:
                continue
            train_mask[i] = True
        # sessions 1-5 of all subjects are used for training
        # session 6 is used for validation

        # Load training data
        train_data = load_tfrecord_dataset(
            tokenized_data_tf_dir,
            batch_size=decoding_model.config.training_config.batch_size,
            shuffle=True,
            concatenate=True,
            drop_last_batch=True,
            buffer_size=2000,
            keep=list(np.where(train_mask)[0]),
        )

        # Load validation data
        val_data = load_tfrecord_dataset(
            tokenized_data_tf_dir,
            batch_size=decoding_model.config.training_config.batch_size,
            shuffle=False,
            concatenate=True,
            drop_last_batch=True,
            buffer_size=2000,
            keep=list(np.where(np.logical_not(train_mask))[0]),
        )

    else:
        # Get data files
        data_files = sorted(glob(f"{tokenized_data_fif_dir}/*.fif"))
        n_total_sessions = len(data_files)

        # Extract task event data and labels
        task_data, task_labels = ud.get_event_trials_and_labels(
            data_files, sequence_length=80,
        )
        # task_data.shape: (n_sessions, n_trials, n_samples, n_channels)
        # task_labels.shape: (n_sessions, n_trials)

        # Get subject labels
        subject_labels = [
            np.full(task_data[n].shape[:2], subject_id[n] - 1)
            for n in range(n_total_sessions)
        ]
        # subject_labels.shape: (n_sessions, n_trials, n_samples)

        # Convert task labels from string to integers
        task_dictionary = {"famous": 0, "unfamiliar": 1, "scrambled": 2, "button": 3}
        str_to_integer = lambda x, d: np.array([d[key] for key in x])
        task_labels = [
            str_to_integer(task_labels[n], task_dictionary)
            for n in range(n_total_sessions)
        ]

        # Split into train and val sets
        train_data, train_labels, train_subjects = [], [], []
        val_data, val_labels, val_subjects = [], [], []
        for n in range(n_total_sessions):
            if ft_mode == "within_subject":
                if session_id[n] == 6:
                    val_data.append(task_data[n])
                    val_labels.append(task_labels[n])
                    val_subjects.append(subject_labels[n])
                else:
                    train_data.append(task_data[n])
                    train_labels.append(task_labels[n])
                    train_subjects.append(subject_labels[n])
            # sessions 1-5 of all subjects are used for training
            # session 6 is used for validation
            elif ft_mode == "new_subject":
                if subject_id[n] == 19:
                    val_data.append(task_data[n])
                    val_labels.append(task_labels[n])
                    val_subjects.append(subject_labels[n])
                else:
                    train_data.append(task_data[n])
                    train_labels.append(task_labels[n])
                    train_subjects.append(subject_labels[n])
            # all sessions of subjects 1-18 are used for training
            # all sessions of subject 19 are used for validation

        # Concatenate data over sessions and event trials into batches
        train_data = np.concatenate(train_data, dtype=np.int32)
        train_labels = np.concatenate(train_labels, dtype=np.int32)
        train_subjects = np.concatenate(train_subjects, dtype=np.int32)

        val_data = np.concatenate(val_data, dtype=np.int32)
        val_labels = np.concatenate(val_labels, dtype=np.int32)
        val_subjects = np.concatenate(val_subjects, dtype=np.int32)

        # Create TensorFlow datasets
        train_data = {
            "data": train_data,
            "session_id": train_subjects,
            "task_label": train_labels,
        }
        val_data = {
            "data": val_data,
            "session_id": val_subjects,
            "task_label": val_labels,
        }

        train_data = (
            tf.data.Dataset
            .from_tensor_slices(train_data)
            .shuffle(buffer_size=10_000)
            .batch(16, drop_remainder=False)
            .prefetch(tf.data.AUTOTUNE)
        )
        val_data = (
            tf.data.Dataset
            .from_tensor_slices(val_data)
            .batch(16, drop_remainder=False)
            .prefetch(tf.data.AUTOTUNE)
        )

    # ---------- Fine-tune pre-trained generator ---------- #
    decoding_model.fit(
        train_data,
        validation_data=val_data,
        tokenize=False,
    )

    print("Generator fine-tuning complete.")
