"""Functions for data analysis."""

import numpy as np
from numpy.linalg import norm


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
