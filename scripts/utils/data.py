"""Functions for data loading and management."""

import os
import pickle
import math
import mne
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm.auto import tqdm
from utils import analysis as ua


def save(data, save_path):
    """Saves data to a specified path using pickle.

    This is a wrapper function for the pickle module
    to save data in a binary format.

    Parameters
    ----------
    data : object
        Data to be saved. Can be any Python object.
    save_path : str
        Path where the data will be saved. The file extension should be .pkl.
    """
    if not save_path.endswith(".pkl"):
        raise ValueError("The file extension should be .pkl.")

    with open(save_path, "wb") as output_handle:
        pickle.dump(data, output_handle)
    output_handle.close()

    return None


def load(save_path):
    """Loads data from a specified path using pickle.

    Parameters
    ----------
    save_path : str
        Path from where the data will be loaded.
        The file extension should be .pkl.

    Returns
    -------
    data : object
        Data loaded from the file. Can be any Python object.
    """
    if not save_path.endswith(".pkl"):
        raise ValueError("The file extension should be .pkl.")

    with open(save_path, "rb") as input_handle:
        data = pickle.load(input_handle)
    input_handle.close()

    return data


def get_tokenizer_history(model_dir):
    """Reads training history of a trained tokenizer model.

    Parameters
    ----------
    model_dir : str
        Directory where the model is stored.
    
    Returns
    -------
    loss : np.ndarray
        Training loss over epochs.
    temperature : np.ndarray
        Annealed temperature over epochs.
    """
    data_file = os.path.join(model_dir, "history.pkl")
    history = load(data_file)
    loss = np.array(history["loss"]) # shape: (n_epochs,)
    temperature = np.array(history["temperature"]) # shape: (n_epochs,)
    return loss, temperature


def get_generator_history(model_dir):
    """Reads training history of a trained generator model.

    Parameters
    ----------
    model_dir : str
        Directory where the model is stored.
    
    Returns
    -------
    train_loss : np.ndarray
        Training loss per epoch.
    val_loss : np.ndarray
        Validation loss per epoch.
    train_top1_acc : np.ndarray
        Training top-1 accuracy per epoch.
    val_top1_acc : np.ndarray
        Validation top-1 accuracy per epoch.
    """
    # Load history object
    data_file = os.path.join(model_dir, "history.pkl")
    history = load(data_file)

    train_loss = np.array(history["loss"])
    val_loss = np.array(history["val_loss"])
    train_top1_acc = np.array(history["top_1"])
    val_top1_acc = np.array(history["val_top_1"])
    # shape: (n_epochs,)

    return train_loss, val_loss, train_top1_acc, val_top1_acc


def remove_outliers(data, threshold=3):
    """Removes outliers from a 1D numpy array.
    
    Parameters
    ----------
    data : np.ndarray or list
        1D array or list from which outliers are to be removed.
    threshold : float, optional
        Threshold in terms of standard deviations to identify outliers.
        Default is 3.

    Returns
    -------
    cleaned_data : np.ndarray
        1D array with outliers removed.
    """
    # Convert to numpy array and validate input data
    if isinstance(data, list):
        data = np.array(data)
    assert data.ndim == 1, "Data must be a 1D array."
    
    # Calculate outlier threshold
    threshold *= np.std(data)
    
    # Remove outliers
    mean = np.mean(data)
    cleaned_data = data[np.abs(data - mean) <= threshold]
    return cleaned_data


def get_outliers(data, method="iqr", threshold=None):
    """Identifies outliers in a 1D numpy array.
    
    Parameters
    ----------
    data : np.ndarray or list
        1D array or list from which outliers are to be identified.
    method : str, optional
        Method to identify outliers. Options are "iqr" (interquartile range)
        and "std" (standard deviation). Default is "iqr".
    threshold : float, optional
        Threshold in terms of standard deviations to identify outliers when 
        method is "std". Default is None. Must be provided if method is "std".
    """
    # Get outliers based on specified method
    if method == "iqr":
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1  # interquartile range
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
    elif method == "std":
        if threshold is None:
            raise ValueError("Threshold must be provided for 'std' method.")
        mean = np.mean(data)
        std = np.std(data)
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    
    return outliers


def static_metric_dict_to_long(metric_dict):
    """Converts a nested dictionary of static spectral metrics to
       a long-format DataFrame.

    Parameters
    ----------
    metric_dict : dict
        Nested dictionary where keys are model names and values are dictionaries
        with dataset names as keys and 2D numpy arrays as values.
        Shape is dict[model][dataset] -> np.ndarray of shape (n_subjects, n_channels).

    Returns
    -------
    df : pd.DataFrame
        Long-format DataFrame with columns (subject, model, dataset, channel, metric).
    """
    # Create pandas dataframe
    df = []
    for mod, ds in metric_dict.items():
        for d, arr in ds.items():
            n_subjects, n_channels = arr.shape
            subjects, channels = np.meshgrid(
                np.arange(n_subjects), np.arange(n_channels), indexing='ij'
            )
            df.append(pd.DataFrame({
                "subject": subjects.ravel(),
                "model": mod,
                "dataset": d,
                "channel": channels.ravel(),
                "metric": arr.ravel().astype(float),
            }))
    
    return pd.concat(df, ignore_index=True)


def dynamic_metric_dict_to_long(metric_dict):
    """Converts a nested dictionary of dynamic spectral metrics to
       a long-format DataFrame.

    Parameters
    ----------
    metric_dict : dict
        Nested dictionary with the shape dict[model][channel][dataset] ->
        np.ndarray of shape (n_subjects,).

    Returns
    -------
    df : pd.DataFrame
        Long-format DataFrame with columns (subject, model, dataset, channel, metric).
    """
    # Create pandas dataframe
    df = []
    for mod, ch_data in metric_dict.items():
        for ch_key, gen_data in ch_data.items():
            ch_idx = int(ch_key[2:])  # extract channel index from keys
            for gen_id, arr in gen_data.items():
                n_subjects = arr.shape[0]
                df.append(pd.DataFrame({
                    "subject": np.arange(n_subjects, dtype=int),
                    "model": mod,
                    "dataset": gen_id,
                    "channel": ch_idx,
                    "metric": arr.astype(float),
                }))
    return pd.concat(df, ignore_index=True)


def find_task_events(raw):
    """Finds task-related events in the raw MNE data.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw MNE data object.

    Returns
    -------
    events : np.ndarray
        Array of events found in the raw data. Shape is (n_events, 3).
    new_event_ids : dict
        Dictionary mapping event names to new event codes.
    """
    # Define event mappings
    new_event_ids = {"famous": 1, "unfamiliar": 2, "scrambled": 3, "button": 4}
    old_event_ids = {
        "famous": [5, 6, 7],
        "unfamiliar": [13, 14, 15],
        "scrambled": [17, 18, 19],
        "buttonp": [
            256, 261, 262, 263, 269,
            270, 271, 273, 274, 275,
            4096, 4101, 4102, 4103, 4109,
            4110, 4111, 4114, 4115, 4352,
            4357, 4359, 4365, 4369,
        ],
    }

    # Find events and remap event codes
    events = mne.find_events(raw, min_duration=0.005, verbose=False)
    for old_event_codes, new_event_codes in zip(
        old_event_ids.values(), new_event_ids.values()
    ):
        events = mne.merge_events(events, old_event_codes, new_event_codes)
    return events, new_event_ids


def get_event_trials_and_labels(fif_files, sequence_length):
    """Extracts event-related trials and labels from a list of FIF files.

    Parameters
    ----------
    fif_files : list of str
        List of paths to FIF files.
    sequence_length : int
        Sequence length in samples for epoching.

    Returns
    -------
    trials : list of np.ndarray
        List of 3D numpy arrays containing trials for each session.
        Each array has shape (n_trials, n_samples, n_channels).
    trial_labels : list of np.ndarray
        List of 1D numpy arrays containing task labels for each session.
        Each array has shape (n_trials,).
    """
    # Extract trials and labels for each session
    trials, trial_labels = [], []
    for file in fif_files:
        raw = mne.io.read_raw_fif(file, preload=True, verbose=False)
        events, event_ids = find_task_events(raw)
        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_ids,
            tmin=0.0,
            tmax=sequence_length / raw.info["sfreq"],
            baseline=None,
            preload=True,
            verbose=False,
            on_missing="ignore",  # proceed silently if no events found
        )

        trial, trial_label = [], []
        for key in ["famous", "unfamiliar", "scrambled", "button"]:
            epoch_data = np.transpose(epochs[key].get_data(picks="misc"), (0, 2, 1))
            trial.extend(epoch_data)
            trial_label.extend([key] * len(epoch_data))

        trial = np.array(trial)
        trial_label = np.array(trial_label)
        trials.append(trial)
        trial_labels.append(trial_label)
    return trials, trial_labels


def get_session_features(generator, trials, subject_id, batch_size):
    """Extracts features from trials within a single session 
       using a trained generator model.

    Parameters
    ----------
    generator : osl_foundation.models.EphysGPT
        Trained generator model.
    trials : np.ndarray
        3D numpy array containing trials within a session.
        Shape is (n_trials, n_samples, n_channels).
    subject_id : int or None
        Subject ID corresponding to the session.
        If None, no subject embedding is used.
    batch_size : int
        Batch size for feature extraction.

    Returns
    -------
    features : np.ndarray
        Array containing extracted features for all trials.
        Shape is (n_trials, latent_sequence_length, n_channels, model_dim).
    """
    # Get model layers
    shift_token_layer = generator.model.get_layer("shift_token")
    input_embedding_layer = generator.model.get_layer("input_embedding")
    decoder_layer = generator.model.get_layer("decoder")

    # Create batches
    n_trials = len(trials)
    indices = np.array_split(
        np.arange(n_trials), math.ceil(n_trials / batch_size)
    )  # for batching

    # Build subject labels
    if subject_id is None:
        subject_labels = None
    else:
        subject_labels = tf.constant(
            subject_id, shape=trials.shape[:2], dtype=tf.int32
        )

    # Define helper function
    @tf.function
    def _extract_features(data, subject_labels):
        x, _ = shift_token_layer(data, training=False)
        x = input_embedding_layer([x, subject_labels], training=False)
        x = decoder_layer(x, training=False)
        return x
    
    # Extract features in batches
    features = []
    for idx in indices:
        batch_trials = trials[idx]
        if subject_labels is None:
            batch_subject_labels = []
        else:
            batch_subject_labels = [tf.gather(subject_labels, idx)]
        x = _extract_features(batch_trials, batch_subject_labels)
        features.extend(list(x.numpy()))
    return np.array(features)


def get_features(generator, trials, subject_ids, batch_size):
    """Extracts features from trials across multiple sessions
       using a trained generator model.
    
    Parameters
    ----------
    generator : osl_foundation.models.EphysGPT
        Trained generator model.
    trials : list of np.ndarray
        List of 3D numpy arrays containing trials for each session.
        Each array has shape (n_trials, n_samples, n_channels).
    subject_ids : list or np.ndarray
        List or array of subject IDs corresponding to each session.
    batch_size : int
        Batch size for feature extraction.

    Returns
    -------
    features : list of np.ndarray
        List of arrays containing extracted features for each session.
        Each array has shape (n_trials, latent_sequence_length, n_channels, model_dim).
    """
    # Validate inputs
    if len(trials) != len(subject_ids):
        raise ValueError(
            "The length of trials must match the length of subject IDs."
        )

    # Extract features for each session
    features = []
    for i, trial in enumerate(tqdm(trials)):
        features.append(
            get_session_features(
                generator, trial, subject_ids[i], batch_size
            )
        )
    return features


def load_features(data_dict, baseline=False):
    """Loads saved feature data from a specified path.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary containing the feature data. Keys are session IDs and
        values are a tuple of task features and labels.
    baseline : bool, optional
        If True, indicates that the baseline features are used.
        Default is False.
    
    Returns
    -------
    X : list of np.ndarray
        List of feature arrays for each session.
        Shape of each array is (n_trials, n_features).
    y : list of np.ndarray
        List of label arrays for each session.
        Shape of each array is (n_trials,).
    session_ids : list of str
        List of session IDs corresponding to each session.
    """
    # Extract and pre-process features and labels
    if baseline:
        X = list(map(
            lambda x: x[0].reshape(x[0].shape[0], -1),
            # flatten channel and time dimensions
            data_dict.values(),
        ))
    else:
        X = list(map(
            lambda x: np.mean(x[0], axis=1).reshape(x[0].shape[0], -1),
            # average over time dimension and flatten channel and model dimensions
            data_dict.values(),
        ))
    y = list(map(lambda x: x[1], data_dict.values()))
    return X, y, list(data_dict.keys())


def split_feature_data(
    data_tuples,
    test_session=None,
    test_subject=None,
    standardize=False,
):
    """Splits feature data into training and test sets based on session ID.
    
    Parameters
    ----------
    data_tuples : tuple of lists
        Tuple containing (X, y, session_ids) where:
        - X is a list of feature arrays for each session.
        - y is a list of label arrays for each session.
        - session_ids is a list of session IDs corresponding to each session.
    test_session : str, optional
        Session ID to use for testing. If None, all sessions
        are used for training.
    test_subject : str, optional
        Subject ID to use for testing. If None, all subjects
        are used for training.
    standardize : bool, optional
        If True, standardizes features using training set statistics.
        Default is False.

    Returns
    -------
    train_Xs : list of np.ndarray
        List of feature arrays for training sessions.
    train_ys : list of np.ndarray
        List of label arrays for training sessions.
    test_Xs : list of np.ndarray
        List of feature arrays for the test session.
    test_ys : list of np.ndarray
        List of label arrays for the test session.
    """
    # Initialize lists
    train_Xs, train_ys = [], []
    test_Xs, test_ys = [], []

    # Split data into training and test sets
    for x, y, session_id in zip(*data_tuples):
        run_id = session_id.split("_")[1]
        subject_id = session_id.split("_")[0]
        if (run_id == test_session) or (subject_id == test_subject):
            test_Xs.append(x)
            test_ys.append(y)
        else:
            train_Xs.append(x)
            train_ys.append(y)

    # Standardize features
    if standardize:
        train_Xs, test_Xs = ua.standardize_features(train_Xs, test_Xs)

    return train_Xs, train_ys, test_Xs, test_ys
