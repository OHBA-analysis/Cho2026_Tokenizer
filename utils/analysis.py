"""Functions for data analysis."""

import os
import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import pdist, squareform
from osl_dynamics.analysis import static
from osl_dynamics.data import processing
from osl_dynamics.inference import modes
from utils import data as ud


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
    mat = mat[:n_subjects, n_subjects:]

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
