"""Functions for data loading and management."""

import os
import pickle
import numpy as np


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
