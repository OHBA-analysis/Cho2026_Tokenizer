"""Functions for data visualization and plotting."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from osl_dynamics.analysis import power


def save(fig, filename, **kwargs):
    """Saves a matplotlib figure to a file.

    Parameters
    ----------
    fig : plt.figure
        Matplotlib figure object.
    filename : str
        Output filename.
    kwargs : dict, optional
        Additional arguments to pass to `fig.savefig <https://matplotlib.org\
        /stable/api/_as_gen/matplotlib.figure.Figure.savefig.html>`_.
    """
    if not filename.endswith(".png"):
        filename += ".png"
    fig.savefig(filename, dpi=300, bbox_inches="tight", **kwargs)
    plt.close(fig)


def plot_tokenizer_loss(
    losses,
    filepath,
    temperature=None,
    fontsize=14,
    transparent=True,
):
    """Plots training loss (and annealed temperature) of learnable tokenizers.

    Parameters
    ----------
    losses : tuple of np.ndarray
        Tuple containing training losses for causal and noncausal tokenizers.
        Each element is a numpy array with shape (n_epochs,).
    filepath : str
        Path where the plot will be saved.
    temperature : np.ndarray, optional
        Annealed temperature over epochs. Shape is (n_epochs,).
    fontsize : int, optional
        Font size for the plot labels and titles. Default is 14.
    transparent : bool, optional
        Whether to make the background of the plot transparent.
        Default is True.
    """
    # Unpack input data
    causal_loss, noncausal_loss = losses

    # Define x-axis (epochs)
    epochs = np.arange(1, len(causal_loss) + 1)

    # Plot training loss
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
    ax.plot(epochs, causal_loss, color="k", lw=1.5, label="Causal")
    ax.plot(epochs, noncausal_loss, color="tab:red", lw=1.5, label="Noncausal")
    ax.set_xlabel("Epochs", fontsize=14)
    ax.set_ylabel("MSE Loss", fontsize=14)
    ax_tw = None
    if temperature is not None:
        ax_tw = ax.twinx()
        ax_tw.plot(
            epochs, temperature, color="tab:orange",
            lw=1.5, ls="--", label="Temperature",
        )
        ax_tw.set_ylabel("Temperature", fontsize=14)    
    for axis in [ax, ax_tw]:
        if axis is not None:
            axis.spines[["top"]].set_visible(False)
            axis.spines[["bottom", "left", "right"]].set_linewidth(1.5)
            axis.tick_params(labelsize=14)
    plt.tight_layout()
    save(fig, filepath, transparent=transparent)
    return None


def plot_token_count_histogram(
    token_counts,
    tokenizer_name,
    filepath,
    color="skyblue",
    fontsize=12,
    transparent=True,
):
    """Plots histograms of token counts for training and test sets.

    Parameters
    ----------
    token_counts : np.ndarray
        Array of token counts for both training and test sets.
        Shape should be (n_tokens,).
    tokenizer_name : str
        Name of the tokenizer.
    filepath : str
        Path where the plot will be saved.
    color : str, optional
        Color for the histogram bars. Default is 'skyblue'.
    fontsize : int, optional
        Font size for the plot labels and titles. Default is 14.
    transparent : bool, optional
        Whether to make the background of the plot transparent.
        Default is True.
    """
    # Plot token histograms
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
    ax.bar(
        range(1, token_counts.shape[0] + 1),
        token_counts,
        width=1,
        color=color,
        edgecolor=None,
        linewidth=0,
        alpha=0.6,
    )
    ax.set_xlabel("Token Index", fontsize=fontsize)
    ax.set_ylabel("Number of Occurrences", fontsize=fontsize)
    ax.set_title(
        f"{tokenizer_name} (n={len(token_counts)})",
        fontsize=fontsize + 2,
        fontweight="bold",
    )
    ax.tick_params(labelsize=fontsize)
    if ax.yaxis.get_offset_text() is not None:
        ax.yaxis.get_offset_text().set_fontsize(fontsize)
    plt.tight_layout()
    save(fig, filepath, transparent=transparent)
    return None


def plot_reconstructed_signals(
    original_ts,
    recon_ts,
    filepath,
    sampling_frequency=1,
    channel_idx=0,
    start_idx=200,
    end_idx=500,
    titles=None,
    fontsize=14,
    transparent=True,
):
    """Plots original and reconstructed time series signals.

    Parameters
    ----------
    original_ts : np.ndarray
        Original time series data. Shape is (n_samples, n_channels).
    recon_ts : list of np.ndarray
        List of reconstructed time series data from different models.
        Each element has shape (n_samples, n_channels).
    filepath : str
        Path where the plot will be saved.
    sampling_frequency : float, optional
        Sampling frequency of the time series data. Default is 1 Hz.
    channel_idx : int, optional
        Index of the channel to be plotted. Default is 0.
    start_idx : int, optional
        Starting index of the time segment to be plotted. Default is 200.
    end_idx : int, optional
        Ending index of the time segment to be plotted. Default is 500.
    titles : list of str, optional
        List of titles for each subplot corresponding to each model.
        Default is None.
    fontsize : int, optional
        Font size for the plot labels and titles. Default is 14.
    transparent : bool, optional
        Whether to make the background of the plot transparent.
        Default is True.
    """
    # Get number of models
    n_models = len(recon_ts)

    # Create time axis
    x_time = np.arange(start_idx, end_idx) / sampling_frequency  # unit: seconds

    # Plot original and reconstructed signals
    fig, ax = plt.subplots(
        nrows=n_models, ncols=1, figsize=(12, 20), sharex=True
    )
    for i in range(n_models):
        ax[i].plot(
            x_time,
            original_ts[start_idx:end_idx, channel_idx],
            color="tab:blue",
            lw=1.5,
            label="Original",
        )
        ax[i].plot(
            x_time,
            recon_ts[i][start_idx:end_idx, channel_idx],
            color="tab:orange",
            lw=1.5, ls="--",
            label="Reconstructed",
        )
        ax[i].set_ylabel("Amplitude (a.u.)", fontsize=fontsize)
        if titles is not None:
            ax[i].set_title(titles[i], fontsize=fontsize, fontweight="bold")
        ax[i].tick_params(labelsize=fontsize)
    ax[i].set_xlabel("Time (s)", fontsize=fontsize)
    plt.tight_layout()
    save(fig, filepath, transparent=transparent)
    return None


def plot_pve(dataframe, palette, filepath, ylim=None, fontsize=14):
    """Plots PVE for different models on each dataset.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame containing PVE data with columns 'Dataset', 'Model', and 'PVE'.
    palette : dict
        Color palette for different models.
        Keys are model names, and values are color codes.
    filepath : str
        Path where the plot will be saved.
    ylim : list of float, optional
        y-axis limits for the plot. Default is None, which lets matplotlib
        choose the limits automatically.
    fontsize : int, optional
        Font size for the plot labels and titles. Default is 14.
    """
    # Validate inputs
    if ylim is None:
        ylim = [None, None]

    # Plot PVE boxplots
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    sns.boxplot(
        data=dataframe,
        x="Dataset",
        y="PVE",
        hue="Model",
        palette=palette,
        ax=ax,
    )
    ax.legend(
        loc="lower right",
        title="Model",
        fontsize=fontsize - 4,
        title_fontsize=fontsize - 4,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_linewidth(1.5)
    ax.set_ylim(ylim)
    ax.set_xlabel("Dataset", fontsize=fontsize)
    ax.set_ylabel("Percentage of Variance Explained (%)", fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    plt.tight_layout()
    save(fig, filepath, transparent=True)
    return None

def plot_channel_location(channel, plot_dir):
    """Plots the location of a specific channel on the brain.

    Parameters
    ----------
    channel : int or np.ndarray
        The index or indices of the channel to plot.
    plot_dir : str
        Directory to save the plot.
    """
    # Define power map for the channel
    power_map = np.zeros(52)
    power_map[channel] = 1
    
    # Plot channel location using osl-dynamics
    power.save(
        power_map,
        mask_file="MNI152_T1_8mm_brain.nii.gz",
        parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
        plot_kwargs={"views": ["lateral", "medial"], "symmetric_cbar": True},
        filename=f"{plot_dir}/channel_location_{channel}.png",
    )
    return None
