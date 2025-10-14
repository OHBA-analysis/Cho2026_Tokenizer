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
