"""Functions for data analysis."""

import numpy as np
from numpy.linalg import norm
from osl_dynamics.inference import modes


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
