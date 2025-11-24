"""Functions for data analysis."""

import os
import numpy as np
import tensorflow as tf
import yaml
from numpy.linalg import norm
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import accuracy_score
from osl_dynamics.analysis import static
from osl_dynamics.data import processing
from osl_dynamics.inference import modes
from osl_dynamics.utils import set_random_seed
from utils import data as ud
from models import create_model, load_model


def compute_l2_distance(a, b, axis=-1):
    """Computes the Euclidean distance along the given axis.

    Parameters
    ----------
    a : np.ndarray
        First array to compare.
    b : np.ndarray
        Second array to compare.
    axis : int, optional
        Axis along which to compute the distance. Default is -1.

    Returns
    -------
    l2_distance : np.ndarray
        Euclidean distance between a and b along the specified axis.
    """
    return norm(a - b, axis=axis)


def compute_cosine_similarity(a, b, eps=1e-8, axis=-1):
    """Computes the cosine similarity along the given axis.

    Parameters
    ----------
    a : np.ndarray
        First array to compare.
    b : np.ndarray
        Second array to compare.
    eps : float, optional
        Small value to avoid division by zero. Default is 1e-8.
    axis : int, optional
        Axis along which to compute the similarity. Default is -1.

    Returns
    -------
    cosine_similarity : np.ndarray
        Cosine similarity between a and b along the specified axis.
    """
    dot = np.sum(a * b, axis=axis)
    norm_a = norm(a, axis=axis)
    norm_b = norm(b, axis=axis)
    return dot / (norm_a * norm_b + eps)


def calculate_summary_stats(stc, sampling_frequency):
    """Calculates summary statistics of state time courses.

    Parameters
    ----------
    stc : list of np.ndarray
        List of state time courses for each subject.
        Shape is (n_subjects, n_samples, n_states).
    sampling_frequency : int
        Sampling frequency of the data.

    Returns
    -------
    fo : np.ndarray
        Fractional occupancies of the states. Shape is (n_subjects, n_states).
    lt : np.ndarray
        Mean lifetimes of the states. Shape is (n_subjects, n_states).
    intv : np.ndarray
        Mean intervals of the states. Shape is (n_subjects, n_states).
    sr : np.ndarray
        Switching rates of the states. Shape is (n_subjects, n_states).
    """
    # Compute summary statistics
    fo = modes.fractional_occupancies(stc)
    lt = modes.mean_lifetimes(
        stc, sampling_frequency=sampling_frequency
    )
    intv = modes.mean_intervals(
        stc, sampling_frequency=sampling_frequency
    )
    sr = modes.switching_rates(
        stc, sampling_frequency=sampling_frequency
    )
    return fo, lt, intv, sr


def get_fingerprint_features(
    feature_type,
    real_data_path,
    generated_data_path,
    save_dir,
    Fs=250,
    load=False,
):
    """Extracts fingerprint features from real and generated data.

    Parameters
    ----------
    feature_type : str
        Type of features to extract. Options are:
        - "tde": Time-delay embedding covariance features.
        - "spectral": Spectral features.
        - "spatial": Spatial features.
        - "spatial_spectral": Combined spatial-spectral features.
    real_data_path : str
        Path to the real data file.
    generated_data_path : str
        Path to the generated data file.
    save_dir : str
        Directory to save/load the extracted features.
    Fs : int, optional
        Sampling frequency of the data. Default is 250 Hz.
    load : bool, optional
        Whether to load the features from the saved files.
        If False, the features will be computed and saved.
        Default is False.
    
    Returns
    -------
    real_features : np.ndarray
        Features extracted from the real data.
        Shape is (n_sessions, n_features).
    generated_features : np.ndarray
        Features extracted from the generated data.
        Shape is (n_sessions, n_features).
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Define helper functions for feature extraction
    def _tde_features(x):
        """Extracts TDE covariance features for fingerprinting."""
        tde_dimension = 21
        n_channels = x[0].shape[1]
        tde_x = [processing.time_embed(d, tde_dimension) for d in x]
        tde_cov = static.functional_connectivity(tde_x, conn_type="cov")
        m, n = np.tril_indices(tde_dimension, k=-1)

        features = []
        for i in range(len(tde_cov)):
            feature = []
            for j in range(n_channels):
                block = tde_cov[i][j * tde_dimension : (j + 1) * tde_dimension][
                    :, j * tde_dimension : (j + 1) * tde_dimension
                ]
                feature.extend(block[m, n])
            features.append(np.array(feature))
        return np.array(features)
    
    def _spectral_features(x):
        """Extracts spectral features for fingerprinting."""
        _, psd = static.welch_spectra(
            data=x,
            sampling_frequency=Fs,
            frequency_range=[1, 45],
            n_jobs=12,
        )
        return psd.mean(axis=1)  # average over channels
    
    def _spatial_features(x):
        """Extracts spatial features for fingerprinting."""
        _, psd = static.welch_spectra(
            data=x,
            sampling_frequency=Fs,
            frequency_range=[1, 45],
            n_jobs=12,
        )
        return psd.mean(axis=2)  # average over frequencies

    def _spatial_spectral_features(x):
        """Extracts spatial-spectral features for fingerprinting."""
        _, psd = static.welch_spectra(
            data=x,
            sampling_frequency=Fs,
            frequency_range=[1, 45],
            n_jobs=12,
        )
        return psd.reshape((psd.shape[0], -1))  # flatten channels and frequencies
    
    FEATURES = {
        "tde": _tde_features,
        "spectral": _spectral_features,
        "spatial": _spatial_features,
        "spatial_spectral": _spatial_spectral_features,
    }

    if not load:
        # Load real and generated data
        real_data = ud.load(real_data_path)
        generated_data = ud.load(generated_data_path)

        # Trim the real data to match the length of the generated data
        for i, (d1, d2) in enumerate(zip(real_data, generated_data)):
            real_data[i] = d1[:d2.shape[0], :]

        # Extract features
        real_features = FEATURES[feature_type](real_data)
        generated_features = FEATURES[feature_type](generated_data)

        # Save features
        np.save(f"{save_dir}/real_features.npy", real_features)
        np.save(f"{save_dir}/generated_features.npy", generated_features)
    else:
        real_features = np.load(f"{save_dir}/real_features.npy")
        generated_features = np.load(f"{save_dir}/generated_features.npy")

    return real_features, generated_features


def get_fingerprint_pairwise_distance(
    real_features,
    generated_features,
    save_dir,
    metric_types="correlation",
    load=False,
):
    """Calculates the pairwise distance between real and generated features.
    
    Parameters
    ----------
    real_features : np.ndarray
        Features extracted from the real data.
        Shape is (n_sessions, n_features).
    generated_features : np.ndarray
        Features extracted from the generated data.
        Shape is (n_sessions, n_features).
    save_dir : str
        Directory to save/load the distance matrix.
    metric_types : str or list of str, optional
        Distance metric(s) to use.
    load : bool, optional
        Whether to load the distance matrix from the saved file.
        If False, the distance matrix will be computed and saved.
        Default is False.

    Returns
    -------
    pdist_list : list of np.ndarray
        List of pairwise distance matrices for each metric.
        Each matrix has shape (n_sessions * 2, n_sessions * 2).
    """
    # Validate inputs
    if not isinstance(metric_types, list):
        metric_types = [metric_types]

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Get pairwise distance matrices
    pdist_list = []
    if not load:
        # Concatenate the features
        concat_features = np.concatenate(
            (real_features, generated_features), axis=0
        )

        # Calculate pairwise distance for each metric
        for metric in metric_types:
            pairwise_dist = squareform(pdist(concat_features, metric=metric))
            np.save(f"{save_dir}/{metric}_pdist.npy", pairwise_dist)
            pdist_list.append(pairwise_dist)
    else:
        # Load the pairwise distance matrices
        for metric in metric_types:
            pairwise_dist = np.load(f"{save_dir}/{metric}_pdist.npy")
            pdist_list.append(pairwise_dist)
    return pdist_list


def get_fingerprint_accuracy(mat, top_k=1):
    """Calculates the accuracy of the subject fingerprints.

    The accuracy is defined as the proportion of subjects whose fingerprint
    is within the top k most similar fingerprints.
    
    Parameters
    ----------
    mat : np.ndarray
        Pairwise distance matrix of shape (n_subjects * 2, n_subjects * 2).
    top_k : int, optional
        The number of top k similar fingerprints to consider for accuracy.
        Default is 1.

    Returns
    -------
    accuracy : float
        The fingerprint accuracy (of the subject classification).
    """
    # Get number of subjects
    n_subjects = mat.shape[0] // 2

    # Only keep the top right quadrant
    mat = mat[n_subjects:, :n_subjects]

    # Calculate accuracy
    count = 0
    for n in range(n_subjects):
        sorted_column = np.sort(mat[:, n])
        if mat[n, n] <= sorted_column[top_k - 1]:
            count += 1
    return count / n_subjects 


def get_fingerprint_consistency_score(mat):
    """Calculates the consistency score between two pairwise distance matrices.
    
    Parameters
    ----------
    mat : np.ndarray
        Pairwise distance matrix of shape (n_subjects * 2, n_subjects * 2).

    Returns
    -------
    consistency : float
        The fingerprint consistency score.
    """
    # Get number of subjects
    n_subjects = mat.shape[0] // 2

    # Split the matrix into original and generated parts
    org_pdist = mat[:n_subjects, :n_subjects]
    gen_pdist = mat[n_subjects:, n_subjects:]

    # Calculate consistency score
    m, n = np.tril_indices(n_subjects, k=-1)
    org_pdist_flat = org_pdist[m, n]
    gen_pdist_flat = gen_pdist[m, n]
    consistency = np.corrcoef(org_pdist_flat, gen_pdist_flat)[0, 1]
    return consistency


def standardize_features(X_train, X_test):
    """Standardizes features using training data statistics.

    Parameters
    ----------
    X_train : list of np.ndarray
        List of training feature arrays for each session.
    X_test : list of np.ndarray
        List of testing feature arrays for each session.

    Returns
    -------
    X_train : list of np.ndarray
        List of standardized training feature arrays.
    X_test : list of np.ndarray
        List of standardized testing feature arrays.
    """
    X_mean = np.mean(
        np.concatenate(X_train), axis=0, keepdims=True
    )
    X_std = np.std(
        np.concatenate(X_train), axis=0, keepdims=True
    )
    X_train = [(x - X_mean) / X_std for x in X_train]
    X_test = [(x - X_mean) / X_std for x in X_test]
    return X_train, X_test


def compute_task_decoding_accuracy(
    data_dict,
    config_path,
    test_session=None,
    test_subject=None,
    seed=None,
    use_tfrecord=False,
    save_dir="tmp",
):
    """Computes task decoding accuracy using logistic regression classifier.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary containing the feature data. Keys are session IDs and
        values are a tuple of task features and labels.
    config_path : str
        Path to the model configuration file.
    test_session : str, optional
        Session ID to be used as the test set. If None, all sessions
        are used for training.
    test_subject : str, optional
        Subject ID to be used as the test set. If None, all subjects
        are used for training.
    seed : int, optional
        Random seed for reproducibility. If None, no seed is set.
        Default is None.
    use_tfrecord : bool, optional
        Whether to use TFRecord datasets for training and evaluation.
        Default is False.
    save_dir : str, optional
        Directory to save TFRecord datasets. Default is "tmp".

    Returns
    -------
    acc : np.ndarray
        Array of decoding accuracies for each test session.
    """
    # Validate inputs
    if test_session is None and test_subject is None:
        raise ValueError("Either test_session or test_subject must be provided.")

    # Set random seed for Python random, NumPy, and TensorFlow
    if seed is not None:
        set_random_seed(seed, op_determinism=False)

    # Get batch size
    with open(config_path, "rb") as f:
        config = yaml.safe_load(f)
    batch_size = config["training_config"]["batch_size"]

    # Load feature data
    feature_data = ud.load_features(data_dict)

    # Split data into training and test sets
    train_Xs, train_ys, test_Xs, test_ys = ud.split_feature_data(
        feature_data,
        test_session=test_session,
        test_subject=test_subject,
    )

    # Create TFRecord datasets
    if use_tfrecord:
        ud.write_tfrecord_shards(train_Xs, train_ys, save_dir=f"{save_dir}/train")
        ud.write_tfrecord_shards(test_Xs, test_ys, save_dir=f"{save_dir}/val")

        train_data = ud.load_tfrecord_shards(
            f"{save_dir}/train",
            batch_size=batch_size,
            buffer=2000,
            seed=seed,
            drop_remainder=False,
        )
        val_data = ud.load_tfrecord_shards(
            f"{save_dir}/val",
            batch_size=batch_size,
            buffer=2000,
            seed=seed,
            shuffle=False,
            drop_remainder=False,
        )
    else:
        # Concatenate data over sessions and event trials into batches
        train_data = np.concatenate(train_Xs)
        train_labels = np.concatenate(train_ys)
        test_data = np.concatenate(test_Xs)
        test_labels = np.concatenate(test_ys)

        # Create TensorFlow datasets
        train_data = {"data": train_data, "task_label": train_labels}
        test_data = {"data": test_data, "task_label": test_labels}

        train_data = (
            tf.data.Dataset
            .from_tensor_slices(train_data)
            .shuffle(buffer_size=2000, seed=seed)
            .batch(batch_size, drop_remainder=False)
            .prefetch(tf.data.AUTOTUNE)
        )
        val_data = (
            tf.data.Dataset
            .from_tensor_slices(test_data)
            .batch(batch_size, drop_remainder=False)
            .prefetch(tf.data.AUTOTUNE)
        )

    # Build logistic regression classifier
    classifier = create_model(config_path)
    classifier.summary()

    # Train the classifier
    classifier.fit(
        train_data,
        validation_data=val_data,
    )

    # Evaluate the classifier
    y_trues, y_preds = [], []
    for test_X, test_y in zip(test_Xs, test_ys):
        _, y_pred = classifier.model({
            "data": test_X,
            "task_label": test_y,
        })
        y_trues.append(test_y)
        y_preds.append(np.argmax(y_pred.numpy(), axis=-1))

    # Compute accuracy for each test session
    acc = np.array([
        accuracy_score(y_true, y_pred)
        for y_true, y_pred in zip(y_trues, y_preds)
    ]) # shape: (n_test_sessions,)
    return acc


def evaluate_fine_tuned_model(
    data_dict,
    config_dir,
    test_session=None,
    test_subject=None,
    batch_size=10,
):
    """Evaluates a fine-tuned model on the test data.

    Parameters
    ----------
    data_dict : dict
        Dictionary containing the feature data. Keys are session IDs and
        values are a tuple of task data and labels.
    config_dir : str
        Directory containing the fine-tuned model configuration and weights.
    test_session : str, optional
        Session ID to be used as the test set. If None, all sessions
        are used for training.
    test_subject : str, optional
        Subject ID to be used as the test set. If None, all subjects
        are used for training.
    batch_size : int, optional
        Batch size for evaluation. Default is 10.
    
    Returns
    -------
    acc : np.ndarray
        Array of decoding accuracies for each test session.
    """
    # Validate inputs
    if test_session is None and test_subject is None:
        raise ValueError("Either test_session or test_subject must be provided.")

    # Unpack data
    task_data, task_labels = [], []
    for data, labels in data_dict.values():
        task_data.append(data)
        task_labels.append(labels)

    # Convert task labels from string to integers
    task_dictionary = {"famous": 0, "unfamiliar": 1, "scrambled": 2, "button": 3}
    str_to_integer = lambda x, d: np.array([d[key] for key in x])
    task_labels = [
        str_to_integer(task_labels[n], task_dictionary)
        for n in range(len(task_labels))
    ]

    # Get the test sets
    test_Xs, test_ys, test_subjects = [], [], []
    for i, session_id in enumerate(list(data_dict.keys())):
        run_id = session_id.split("_")[1]
        subject_id = session_id.split("_")[0]
        if (run_id == test_session) or (subject_id == test_subject):
            test_Xs.append(task_data[i].astype(np.int32))
            test_ys.append(task_labels[i].astype(np.int32))
            test_subjects.append(
                np.full(task_data[i].shape[:2], int(subject_id[3:]) - 1).astype(np.int32)
            )

    # Load fine-tuned model
    ft_model = load_model(config_dir, checkpoint="latest")
    ft_model.summary()

    # Evaluate the model
    y_trues, y_preds = [], []
    for test_X, test_y, test_subject in zip(test_Xs, test_ys, test_subjects):
        trial_idx = np.array_split(
            np.arange(test_X.shape[0]), batch_size
        )
        y_pred = np.zeros_like(test_y)
        for idx in trial_idx:
            _, pred = ft_model.model({
                "data": test_X[idx],
                "session_id": test_subject[idx],
                "task_label": test_y[idx],
            }, training=False)
            pred = np.argmax(pred.numpy(), axis=-1)
            y_pred[idx] = pred

        # Collect session-level true and predicted labels
        y_trues.append(test_y)
        y_preds.append(y_pred)

    # Compute accuracy for each test session
    acc = np.array([
        accuracy_score(y_true, y_pred)
        for y_true, y_pred in zip(y_trues, y_preds)
    ]) # shape: (n_test_sessions,)
    return acc
