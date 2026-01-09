"""Test script for validating dataset tokenization."""

# Import packages
import os
import numpy as np
from glob import glob
from osl_dynamics.data import load_tfrecord_dataset


if __name__ == "__main__":
    # ---------- Directories ---------- #
    BASE_DIR = "/well/woolrich/users/olt015/Cho2026_Tokenizer"
    DATA_DIR = os.path.join(BASE_DIR, "tokenized_data")
    TFR_DIR = os.path.join(BASE_DIR, "tokenized_data_tfrecords")

    # ---------- Model names and IDs ---------- #
    model_names = [
        "causal", "noncausal",
        "mu_transform", "mu_transform_big",
        "mu_transform_small", "mu_transform_tiny",
        "standard_quantile",
    ]
    model_ids = [25, 27] + [0] * 5

    # ---------- Test on numpy token dataset ---------- #
    print("[INFO] Testing tokenized numpy dataset...")

    token_files = []
    subject_nums = []
    for name, run_id in zip(model_names, model_ids):
        files = sorted(glob(f"{DATA_DIR}/{name}/{run_id}/x*.npy"))
        token_files.append(files)
        subject_nums.append(len(files))

    # Check the number of subjects
    assert np.all(np.array(subject_nums) == subject_nums[0]), "Number of subjects must be the same."

    # Check the shape of tokenized data across subjects
    data_shapes = []
    for files in token_files:
        shapes = []
        for file in files:
            data = np.load(file)
            shapes.append(data.shape)
        data_shapes.append(np.array(shapes))

    assert all(np.array_equal(ds, data_shapes[0]) for ds in data_shapes), "Data shapes must be the same."

    print("[INFO] Passed.")

    # ---------- Test on TFRecord token dataset ---------- #
    print("[INFO] Testing tokenized TFRecord dataset...")

    val_tokens = []
    val_labels = []
    val_raw_data = []

    for name, run_id in zip(model_names, model_ids):
        _, vd = load_tfrecord_dataset(
            f"{TFR_DIR}/{name}/{run_id}",
            batch_size=32,
            buffer_size=2000,
            shuffle=False,
            drop_last_batch=True,
            concatenate=True,
        )
        tokens, labels, raw_data = [], [], []
        for batch in vd:
            tokens.append(batch["data"].numpy())
            labels.append(batch["session_id"].numpy())
            raw_data.append(batch["raw_data"].numpy())
        val_tokens.append(np.concatenate(tokens, axis=0)) # shape: (n_batch, sequence_length, n_channels)
        val_labels.append(np.concatenate(labels, axis=0)) # shape: (n_batch, sequence_length)
        val_raw_data.append(np.concatenate(raw_data, axis=0)) # shape: (n_batch, sequence_length, n_channels)

    # Check the shape of tokenized data across subjects
    token_shapes = [vt.shape for vt in val_tokens]
    assert all(np.array_equal(vs, token_shapes[0]) for vs in token_shapes), "Tokenized data shapes must be the same."

    # Check the shape of session labels across subjects
    label_shapes = [vl.shape for vl in val_labels]
    assert all(np.array_equal(ls, label_shapes[0]) for ls in label_shapes), "Label shapes must be the same."

    # Check the shape of raw data across subjects
    raw_shapes = [vr.shape for vr in val_raw_data]
    assert all(np.array_equal(rs, raw_shapes[0]) for rs in raw_shapes), "Raw data shapes must be the same."

    # Verify the consistency of labels across models
    for i in range(1, len(val_labels)):
        assert np.array_equal(val_labels[i], val_labels[0]), ...
        f"Label mismatch between {model_names[i]} and {model_names[0]}"

    # Verify the consistency of raw data windows across models
    for i in range(1, len(val_raw_data)):
        assert np.array_equal(val_raw_data[i], val_raw_data[0]), ...
        f"Raw data mismatch between {model_names[i]} and {model_names[0]}"

    print("[INFO] Passed.")
    print("Testing complete.")
