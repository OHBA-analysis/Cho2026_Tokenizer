"""Functions for data loading and management."""

import os
import pickle
import numpy as np
import pandas as pd


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
