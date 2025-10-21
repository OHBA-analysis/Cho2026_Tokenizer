"""Functions for data visualization and plotting."""

import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import cm
from matplotlib.ticker import ScalarFormatter
from nilearn.plotting import plot_markers
from osl_dynamics.analysis import power
from osl_dynamics.utils.misc import override_dict_defaults
from osl_dynamics.utils.parcellation import Parcellation
from osl_dynamics.utils.plotting import create_figure
from utils.array_ops import round_nonzero_decimal, round_up_half


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


def _format_colorbar_ticks(ax):
    """Formats x-axis ticks in the colobar such that integer values are 
       plotted, instead of decimal values.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        A colobar axis to format.
    """

    if np.any(np.abs(ax.get_xlim()) < 1):
        hmin = round_nonzero_decimal(ax.get_xlim()[0], method="ceil") # ceiling for negative values
        hmax = round_nonzero_decimal(ax.get_xlim()[1], method="floor") # floor for positive values
        ax.set_xticks(np.array([hmin, 0, hmax]))
    else:
        ax.set_xticks(
            [round_up_half(val) for val in ax.get_xticks()[1:-1]]
        )
    
    return None


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


def plot_token_prediction_loss(
    dataframe,
    metric,
    palette,
    filepath,
    ylim=None,
    fontsize=14,
):
    """Plots token prediction loss (PVE and MSE) for different tokenizer types.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame containing token prediction loss data with columns
        "Model" and the specified metric ("PVE" or "MSE").
    metric : str
        Metric to plot. Should be either "pve" or "mse".
    palette : dict
        Color palette for different models.
        Keys are model names, and values are color codes.
    filepath : str
        Path where the plot will be saved.
    ylim : list of float, optional
        Y-axis limits for the plot. Default is None, which lets matplotlib
        choose the limits automatically.
    fontsize : int, optional
        Font size for the plot labels and titles. Default is 14.
    """
    # Validate inputs
    if metric not in ["pve", "mse"]:
        raise ValueError("Metric must be either 'pve' or 'mse'.")

    # Plot violin plots for the specified metric
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    sns.violinplot(
        data=dataframe,
        x="Model",
        y=metric.upper(),
        hue="Model",
        ax=ax,
        density_norm="count",
        inner="box",
        palette=palette,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_linewidth(1.5)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel("Tokenizer", fontsize=fontsize)
    if metric == "pve":
        ax.set_ylabel("Percentage of Variance Explained (%)", fontsize=fontsize)
    else:
        ax.set_ylabel("Mean Squared Error", fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    plt.tight_layout()
    save(fig, filepath, transparent=True)

    return None


def plot_psd(
    *psds,
    freq,
    plot_dir,
    titles,
    parcellation_file,
    show_topomap=True,
    topomap_pos=None,
    fontsize=20,
):
    """Plots the power spectral density (PSD) of original and generated data.

    Parameters
    ----------
    psd : list of np.ndarray
        Power spectral density of the real and generated data.
        Each element has shape (n_subjects, n_channels, n_frequencies).
    freq : np.ndarray
        Frequencies corresponding to the PSD.
        Shape is (n_frequencies,).
    plot_dir : str
        Directory to save the plots.
    titles : list of str
        Titles for the plots, corresponding to each dataset.
    parcellation_file : str
        Path to parcellation file.
    show_topomap : bool, optional
        Whether to show the topomap of the parcels in the inset of each subplot.
    topomap_pos : list, optional
        Positioning and size of the topomap ([x0, y0, width, height]).
        x0, y0, width, and height should be floats between 0 and 1.
        Defaults to [0.45, 0.47, 0.5, 0.55] to place the topomap on
        the top right.
    fontsize : int, optional
        Font size for the plot. Default is 20.
    """
    # Create directory if it doesn't exist
    os.makedirs(plot_dir, exist_ok=True)

    # Get data dimension
    n_psds = len(psds)
    n_channels = psds[0].shape[1]

    # Compute mean and standard deviation across subjects/sessions
    psd_mean = [np.mean(p, axis=0) for p in psds]
    psd_std = [np.std(p, axis=0) for p in psds]
    # shape: (n_psds, n_channels, n_frequencies)

    # Set visualization parameters
    n_cols = n_psds // 2 + (n_psds % 2 > 0)
    n_rows = n_psds // n_cols + (n_psds % n_cols > 0)

    # Set default topomap position
    if topomap_pos is None:
        topomap_pos = [0.45, 0.47, 0.5, 0.55]

    # Get the center of each parcel
    parcellation = Parcellation(parcellation_file)
    roi_centers = parcellation.roi_centers()

    # Reorder to use colour to indicate anterior -> posterior location
    order = np.argsort(roi_centers[:, 1])
    roi_centers = roi_centers[order]

    # Plot subject-averaged, channel-wise PSDs
    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols,
        figsize=(6.5 * n_cols, 6 * n_rows),
        sharey=True,
    )
    for i, ax in enumerate(axes.flat):
        # Reorder channels
        psd = psd_mean[i]
        psd = np.copy(psd)[order]

        # Plot PSDs for each channel
        colors = cm.viridis_r(np.linspace(0, 1, n_channels))
        for j in reversed(range(n_channels)):
            ax.plot(freq, psd[j], color=colors[j])

        # Add parcel topomap in the inset
        if show_topomap:
            inside_ax = ax.inset_axes(topomap_pos)
            plot_markers(
                np.arange(parcellation.n_parcels),
                roi_centers,
                node_size=12,
                colorbar=False,
                axes=inside_ax,
            )
        
        # Adjust axis settings
        ax.set_ylim([None, 0.13])
        ax.spines[["top", "bottom", "left", "right"]].set_linewidth(1.5)
        if i >= n_cols:
            ax.set_xlabel("Frequency (Hz)", fontsize=fontsize)
        if i % n_cols == 0:
            ax.set_ylabel("PSD (a.u.)", fontsize=fontsize)
        ax.set_title(titles[i], fontsize=fontsize, fontweight="bold")
        ax.tick_params(labelsize=fontsize)
    
    save(fig, f"{plot_dir}/psd_mean_channel.png")

    # Average over channels
    psd_mean = [np.mean(p, axis=(0, 1)) for p in psds]
    psd_std = [np.std(p, axis=(0, 1)) for p in psds]
    # shape: (n_psds, n_frequencies)

    # Plot overall mean PSDs (averaged over channels)
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(6 * n_cols, 6 * n_rows),
        sharey=True,
    )
    for i, ax in enumerate(axes.flat):
        ax.plot(freq, psd_mean[i], color="tab:blue")
        ax.fill_between(
            freq,
            psd_mean[i] - psd_std[i],
            psd_mean[i] + psd_std[i],
            color="tab:blue",
            alpha=0.3,
        )

        # Adjust axis settings
        ax.set_ylim([None, 0.12])
        ax.spines[["top", "bottom", "left", "right"]].set_linewidth(1.5)
        if i >= n_cols:
            ax.set_xlabel("Frequency (Hz)", fontsize=fontsize)
        if i % n_cols == 0:
            ax.set_ylabel("PSD (a.u.)", fontsize=fontsize)
        ax.set_title(titles[i], fontsize=fontsize, fontweight="bold")
        ax.tick_params(labelsize=fontsize)
    
    fig.tight_layout()
    save(fig, f"{plot_dir}/psd_mean_overall.png")

    return None


def plot_static_power_maps(
    *psds,
    freq,
    filename,
    fontsize=25,
):
    """Plots the static power maps using the given PSDs.

    Parameters
    ----------
    psds : list of np.ndarray
        Power spectral density of the generated data.
        Each element has shape (n_subjects, n_channels, n_frequencies).
    freq : np.ndarray
        Frequencies corresponding to the PSD.
        Shape is (n_frequencies,).
    filename : str
        File path to the saved plot.
    fontsize : int, optional
        Font size for the plot. Default is 18.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Get data dimension
    n_psds = len(psds)

    # Set frequency bands
    # (figures will be numbered according to this order)
    frequency_bands = {
        # "Delta (1-4 Hz)": [1, 4],
        # "Theta (4-8 Hz)": [4, 8],
        # "Alpha (8-12 Hz)": [8, 12],
        # "Beta (12-30 Hz)": [12, 30],
        # "Gamma (30-45 Hz)": [30, 45],
        "Wide (1-45 Hz)": [1, 45],
    }

    # Compute subject-level power maps in each frequency band
    p = [{} for _ in range(n_psds)]
    # shape: (n_psds, n_freq_bands, n_subjects, n_channels)
    for band, (low, high) in frequency_bands.items():
        for i, psd_i in enumerate(psds):
            p[i][band] = power.variance_from_spectra(
                freq, psd_i, frequency_range=[low, high]
            )  # shape: (n_subjects, n_channels)

    # Compute group-level power maps in each frequency band
    group_p = [{} for _ in range(n_psds)]
    # shape: (n_psds, n_freq_bands, n_channels)
    for band in frequency_bands.keys():
        for i in range(n_psds):
            group_p[i][band] = np.mean(p[i][band], axis=0)
            # shape: (n_channels,)

    # Concatenate power maps across frequency bands
    power_maps = []
    for i in range(n_psds):
        power_maps += list(group_p[i].values())
    power_maps = np.array(power_maps)
    # shape: (n_psds, n_freq_bands, n_channels); or (n_freq_bands, n_channels) if n_psds=1
    power_maps -= np.mean(power_maps, axis=1, keepdims=True)

    # Plot power maps
    figures, _ = power.save(
        power_maps,
        mask_file="MNI152_T1_8mm_brain.nii.gz",
        parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
        subtract_mean=False,
        mean_weights=None,
        plot_kwargs={
            "symmetric_cbar": True,
            "views": ["lateral", "medial"],
            "cmap": "RdBu_r",
        },
        titles=list(frequency_bands.keys()),
    )
    for i, fig in enumerate(figures):
        # Reset figure size
        fig.set_size_inches(5, 6)
        
        # Change colorbar position
        cb_ax = fig.axes[-1]
        pos = cb_ax.get_position()
        new_pos = [pos.x0 * 0.92, pos.y0 + 0.02, pos.width * 1.50, pos.height * 1.10]
        cb_ax.set_position(new_pos)
        
        # Set colorbar styles
        _format_colorbar_ticks(cb_ax)
        cb_ax.xaxis.set_major_formatter(ScalarFormatter())
        cb_ax.ticklabel_format(style="scientific", axis="x", scilimits=(-2, 6))
        cb_ax.tick_params(labelsize=fontsize)
        cb_ax.xaxis.offsetText.set_fontsize(fontsize)
        if len(figures) > 1:
            tmp_filename = filename.replace(
                filename.split('.')[0], filename.split('.')[0] + f"_{i}"
            )
            save(fig, tmp_filename, transparent=True)
        else:
            save(fig, filename, transparent=True)

    return None


def plot_metric_violin(
    df,
    x,
    y,
    hue,
    emm_df,
    metric_name,
    palette,
    filename,
    x_labels=None,
    plot_strip=False,
):
    """Plots violin plots for a specific metric across different models.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the metric data.
    x : str
        Column name in the dataframe to be used as x-axis.
    y : str
        Column name in the dataframe to be used as y-axis.
    hue : str
        Column name in the dataframe to be used as hue.
    emm_df : pd.DataFrame
        DataFrame containing estimated marginal means (EMMs) for the metric.
    metric_name : str
        Name of the metric to be displayed on the y-axis label.
    palette : dict
        Color palette for different models.
        Keys are model names, and values are color codes.
    filename : str
        Path where the plot will be saved.
    x_labels : list of str, optional
        Custom x-axis labels. If None, model names from the palette keys
        will be used. Default is None.
    plot_strip : bool, optional
        Whether to overlay a strip plot on the violin plot.
        Default is False.
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
    sns.violinplot(
        data=df, x=x, y=y, hue=hue,
        inner="quart", # cut=0,
        palette=palette,
        ax=ax,
    )
    if plot_strip:
        sns.stripplot(
            data=df, x=x, y=y,
            color="k", alpha=0.4,
            jitter=0.15,
            ax=ax,
        )
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_linewidth(1.5)
    ax.errorbar(
        x=emm_df["model"].cat.codes,
        y=emm_df["emmean"],
        yerr=[
            emm_df["emmean"] - emm_df["asymp.LCL"],
            emm_df["asymp.UCL"] - emm_df["emmean"],
        ],
        fmt="o", capsize=6, linewidth=2, markersize=6, color="white",
    )
    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
    if x_labels is None:
        ax.set_xticklabels(list(palette.keys()))
    else:
        ax.set_xticklabels(x_labels)
    ax.set_xlabel("Model")
    ax.set_ylabel(metric_name)
    plt.tight_layout()
    save(fig, filename, transparent=True)
    
    return None


def plot_model_dataset_interaction(
    emm_df,
    metric_name,
    palette,
    filename,
    x_labels=None,
):
    """Plots model-dataset (fixed terms) interaction using estimated marginal 
       means (EMMs).

    By looking at how consistent EMMs are across datasets for each model, 
    we can assess the generalizability of the metric across datasets.

    Parameters
    ----------
    emm_df : pd.DataFrame
        DataFrame containing estimated marginal means (EMMs) for the metric.
    metric_name : str
        Name of the metric to be displayed on the y-axis label.
    palette : dict
        Color palette for different models.
        Keys are model names, and values are color codes.
    filename : str
        Path where the plot will be saved.
    x_labels : list of str, optional
        Custom x-axis labels. If None, labels will be automatically generated.
        Default is None.
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 5))
    sns.lineplot(
        data=emm_df,
        x="model",
        y="emmean",
        hue="dataset",
        marker="o",
        palette=palette,
        ax=ax,
    )
    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
    if x_labels is not None:
        ax.set_xticklabels(x_labels)
    ax.set_xlabel("Model")
    ax.set_ylabel(f"Estimated Marginal Means ({metric_name})")
    ax.set_title("Model-Dataset interaction (EMMs)")
    ax.legend(title="Dataset", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    save(fig, filename, transparent=True)
    
    return None


def plot_alpha(
    *alpha,
    n_samples=None,
    cmap="tab10",
    sampling_frequency=None,
    y_labels=None,
    title=None,
    fontsize=15,
    plot_kwargs=None,
    fig_kwargs=None,
    filename=None,
    axes=None,
):
    """Plot alpha.

    Parameters
    ----------
    alpha : np.ndarray
        A collection of alphas passed as separate arguments.
    n_samples: int, optional
        Number of time points to be plotted.
    cmap : str or matplotlib.colors.ListedColormap, optional
        Matplotlib colormap.
    sampling_frequency : float, optional
        The sampling frequency of the data in Hz.
    y_labels : str, optional
        Labels for the y-axis of each alpha time series.
    title : str, optional
        Title for the plot.
    fontsize : int, optional
        Font size for axes and tick labels. Defaults to 15.
    plot_kwargs : dict, optional
        Any parameters to be passed to plt.stackplot.
    fig_kwargs : dict, optional
        Arguments to pass to :code:`plt.subplots()`.
    filename : str, optional
        Output filename.
    axes : list of plt.axes, optional
        A list of matplotlib axes to plot on. If None, a new
        figure is created.

    Returns
    -------
    fig : plt.figure
        Matplotlib figure object. Only returned if `ax=None` and
        `filename=None`.
    ax : plt.axes
        Matplotlib axis object(s). Only returned if `ax=None` and
        `filename=None`.
    """
    n_alphas = len(alpha)
    if isinstance(axes, plt.Axes):
        axes = [axes]
    if axes is not None and len(axes) != n_alphas:
        raise ValueError("Number of axes must match number of alphas.")

    n_modes = max(a.shape[1] for a in alpha)
    n_samples = min(n_samples or np.inf, alpha[0].shape[0])
    if isinstance(cmap, str):
        if cmap in [
            "Pastel1",
            "Pastel2",
            "Paired",
            "Accent",
            "Dark2",
            "Set1",
            "Set2",
            "Set3",
            "tab10",
            "tab20",
            "tab20b",
            "tab20c",
        ]:
            cmap = matplotlib.colormaps.get_cmap(cmap)
        else:
            cmap = matplotlib.colormaps.get_cmap(cmap, lut=n_modes)
    cmap = cmap.copy()
    colors = cmap.colors

    # Validation
    if fig_kwargs is None:
        fig_kwargs = {}
    default_fig_kwargs = dict(
        figsize=(12, 2.5 * n_alphas), sharex="all", facecolor="white"
    )
    fig_kwargs = override_dict_defaults(default_fig_kwargs, fig_kwargs)

    if plot_kwargs is None:
        plot_kwargs = {}
    default_plot_kwargs = dict(colors=colors)
    plot_kwargs = override_dict_defaults(default_plot_kwargs, plot_kwargs)

    if y_labels is None:
        y_labels = [None] * n_alphas
    elif isinstance(y_labels, str):
        y_labels = [y_labels] * n_alphas
    elif len(y_labels) != n_alphas:
        raise ValueError("Incorrect number of y_labels passed.")

    # Create figure if axes not passed
    if axes is None:
        fig, axes = create_figure(n_alphas, **fig_kwargs)
    else:
        fig = axes[0].get_figure()

    if isinstance(axes, plt.Axes):
        axes = [axes]

    # Plot data
    for a, ax, y_label in zip(alpha, axes, y_labels):
        time_vector = (
            np.arange(n_samples) / sampling_frequency
            if sampling_frequency
            else range(n_samples)
        )
        ax.stackplot(time_vector, a[:n_samples].T, **plot_kwargs)
        ax.autoscale(tight=True)
        ax.set_ylabel(y_label, fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)

    # Set axis label and title
    axes[-1].set_xlabel("Time (s)" if sampling_frequency else "Sample", fontsize=fontsize)
    axes[0].set_title(title)

    # Fix layout
    plt.tight_layout()

    # Add a colour bar
    norm = matplotlib.colors.BoundaryNorm(
        boundaries=range(n_modes + 1), ncolors=n_modes
    )
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.subplots_adjust(right=0.94)
    cb_ax = fig.add_axes([0.95, 0.15, 0.025, 0.7])
    cb = fig.colorbar(mappable, cax=cb_ax, ticks=np.arange(0.5, n_modes, 1))
    cb.ax.set_yticklabels(range(1, n_modes + 1))

    # Save to file if a filename has been passed
    if filename is not None:
        save(fig, filename, transparent=True)
    else:
        return fig, axes


def plot_hmm_loss(loss, filename, fontsize=12):
    """Plots HMM training loss curve.
    
    Parameters
    ----------
    loss : np.ndarray
        Array of training loss values over epochs.
        Shape is (n_epochs,).
    filename : str
        Path where the plot will be saved.
    fontsize : int, optional
        Font size for the plot. Default is 12.
    """
    # Plot HMM loss curve
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
    epochs = np.arange(1, len(loss) + 1)
    ax.plot(epochs, loss, label="Training Loss")
    ax.set_xlabel("Epochs", fontsize=fontsize)
    ax.set_ylabel("Loss", fontsize=fontsize)
    for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_linewidth(2)
    plt.tight_layout()
    save(fig, filename, transparent=True)
    
    return None


def plot_dynamic_psds(
    freq,
    psds,
    filename,
    colors,
    xlim=None,
    ylim=None,
    legend=True,
    fontsize=18,
):
    """Plots the state-wise power spectral densities (PSDs) from 
       univariate TDE-HMM.

    Parameters
    ----------
    freq : np.ndarray
        Frequencies corresponding to the PSD.
        Shape is (n_freqs,).
    psds : np.ndarray
        Power spectral densities of the HMM states.
        Shape is (n_subjects, n_states, n_freqs).
    filename : str
        Path where the plot will be saved.
    colors : list of str
        List of colors for each state.
    xlim : list of float, optional
        X-axis limits for the plot. Default is None, which lets matplotlib
        choose the limits automatically.
    ylim : list of float, optional
        Y-axis limits for the plot. Default is None, which lets matplotlib
        choose the limits automatically.
    legend : bool, optional
        Whether to display the legend. Default is True.
    fontsize : int, optional
        Font size for the plot. Default is 18.
    """
    # Compute mean and standard deviation across subjects/sessions
    group_psd = np.mean(psds, axis=0)
    group_std = np.std(psds, axis=0)
    # shape: (n_states, n_frequencies)

    # Get number of states
    n_states = group_psd.shape[0]

    # Plot state-wise PSDs
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
    for n in range(n_states):
        ax.plot(
            freq, group_psd[n], color=colors[n], label=f"State {n+1}"
        )
        ax.fill_between(
            freq,
            group_psd[n] - group_std[n],
            group_psd[n] + group_std[n],
            color=colors[n],
            alpha=0.3,
        )
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(1.5)
    if legend:
        ax.legend(loc="upper right", fontsize=fontsize - 2)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel("Frequency (Hz)", fontsize=fontsize)
    ax.set_ylabel("PSD (a.u.)", fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    plt.tight_layout()
    save(fig, filename, transparent=True)

    return None


def plot_summary_stats(
    summary_stats,
    metric_name,
    filename,
    palette=None,
    ylim=None,
    fontsize=18,
):
    """Plots selected summary statistics computed from HMM 
       state time courses.

    Parameters
    ----------
    summary_stats : pd.DataFrame
        DataFrame containing summary statistics with columns
        "Value", "Metric", "Subject", and "State".
    metric_name : str
        Name of the metric to plot. Should be one of
        "fo", "lt", "intv", or "sr".
    filename : str
        Path where the plot will be saved.
    palette : dict, optional
        Color palette for different states.
        Keys are state indices (0, 1, 2, ...), and values are color codes.
        Default is None, which uses seaborn's default palette.
    ylim : list of float, optional
        Y-axis limits for the plot. Default is None, which lets matplotlib
        choose the limits automatically.
    fontsize : int, optional
        Font size for the plot. Default is 18.
    """
    # Set metric labels
    metric_labels = {
        "fo": "Fractional Occupancy",
        "lt": "Mean Lifetime (s)",
        "intv": "Mean Interval (s)",
        "sr": "Burst Rate (Hz)",
    }

    # Extract metric values
    metric_values = summary_stats[
        summary_stats["Metric"] == metric_name
    ]

    # Visualize summary statistics
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.5, 5))
    sns.violinplot(
        data=metric_values,
        x="State",
        y="Value",
        hue="State",
        inner="box",
        palette=palette,
        saturation=0.5,
        legend=False,
        ax=ax,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_linewidth(1.5)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels([int(tick) + 1 for tick in ax.get_xticks()])
    ax.set_xlabel("State", fontsize=fontsize)
    ax.set_ylabel(metric_labels[metric_name], fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    plt.tight_layout()
    save(fig, filename, transparent=True)

    return None


def plot_top_k_accuracy(
    pred_accuracy,
    model_names,
    palette,
    filename,
    fontsize=14,
):
    """Plots top-k accuracy curves for different models.
    
    Parameters
    ----------
    pred_accuracy : dict
        Dictionary containing top-k accuracy data for each model.
        Keys are model names, and values are list. Each list contains
        a list of top-k accuracy floats from different data generations.
        Each top-k accuracy list has shape (n_subjects,).
    model_names : list of str
        List of model names to be plotted.
    palette : dict
        Color palette for different models.
        Keys are model names, and values are color codes.
    filename : str
        Path where the plot will be saved.
    fontsize : int, optional
        Font size for the plot. Default is 14.
    """
    # Get the number of subjects
    n_subjects = len(pred_accuracy[model_names[0]][0])

    # Create x-axis and random baseline
    x = range(1, n_subjects + 1)
    random_baseline = np.arange(1, n_subjects + 1) / n_subjects

    # Get color keys
    color_keys = list(palette.keys())

    # Plot top-k accuracy curve for each model
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
    ax.plot(x, random_baseline, lw=2, ls="--", color="k", label="Random")
    for n, mod in enumerate(model_names):
        mean_accuracy = np.mean(pred_accuracy[mod], axis=0)  # average over data generations
        std_accuracy = np.std(pred_accuracy[mod], axis=0)  # std over data generations
        ax.plot(
            x, mean_accuracy,
            lw=2, color=palette[color_keys[n]],
            label=color_keys[n].replace("\n", " "),
        )
        ax.fill_between(
            x, mean_accuracy - std_accuracy, mean_accuracy + std_accuracy,
            color=palette[color_keys[n]], alpha=0.3,
        )
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(1.5)
    ax.set_xlabel("Top-k", fontsize=fontsize)
    ax.set_ylabel("Mean Accuracy (±1 SD)", fontsize=fontsize)
    ax.tick_params(width=1.5, labelsize=fontsize)
    ax.legend(loc="lower right", fontsize=fontsize - 3)
    plt.tight_layout()
    save(fig, filename, transparent=True)

    return None


def plot_fingerprint_box(
    metrics,
    metric_name,
    model_names,
    palette,
    filename,
    ylim=None,
    strip=False,
    fontsize=14,
):
    """Plots boxplots for subject fingerprinting metrics across different models.
    
    Parameters
    ----------
    metrics : dict
        Dictionary containing fingerprinting metric data for each model.
        Keys are model names, and values are list. Each list contains
        a list of metric floats from different data generations.
    metric_name : str
        Name of the metric to use.
    model_names : list of str
        List of model names to be plotted.
    palette : dict
        Color palette for different models.
        Keys are model names, and values are color codes.
    filename : str
        Path where the plot will be saved.
    ylim : list of float, optional
        Y-axis limits for the plot. Default is None, which lets matplotlib
        choose the limits automatically.
    strip : bool, optional
        Whether to overlay a strip plot on the box plot.
        Default is False.
    fontsize : int, optional
        Font size for the plot. Default is 14.
    """
    # Change keys of palette to model names
    color_keys = list(palette.keys())
    palette = {
        model_names[i]: palette[color_keys[i]]
        for i in range(len(model_names))
    }

    # Plot boxplots for fingerprinting metrics
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
    df_metric = pd.DataFrame(metrics).melt(
        var_name="Model", value_name=metric_name,
    )
    sns.boxplot(
        data=df_metric,
        x="Model",
        y=metric_name,
        hue="Model",
        hue_order=model_names,
        palette=palette,
        saturation=0.6,
        legend=False,
        ax=ax,
    )
    if strip:
        sns.stripplot(
            data=df_metric,
            x="Model",
            y=metric_name,
            hue="Model",
            hue_order=model_names,
            palette=palette,
            linewidth=1.5,
            alpha=0.4,
            jitter=0.15,
            ax=ax,
        )
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(1.5)
    if ylim is not None:
        ax.set_ylim(ylim)
    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
    ax.set_xticklabels(color_keys)
    ax.set_xlabel("Model", fontsize=fontsize)
    ax.set_ylabel(metric_name, fontsize=fontsize)
    ax.tick_params(width=1.5, labelsize=fontsize)
    plt.tight_layout()
    save(fig, filename, transparent=True)

    return None
