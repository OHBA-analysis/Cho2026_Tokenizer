"""Functions for data visualization and plotting."""

import os
import matplotlib.pyplot as plt
import numpy as np
from utils import analysis as ua
from utils import data as ud


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


def plot_generator_history(
    model_type,
    run_id,
    model_dir,
    save_dir,
    fontsize=15,
    legend=False,
    transparent=True,
):
    """Plots training and validation loss and accuracy of learned generators.

    Parameters
    ----------
    model_type : str
        Type of generator model (e.g., "causal", "noncausal").
    run_id : int
        ID for the specific training run of the model.
    model_dir : str
        Directory where the trained model and its history are stored.
    save_dir : str
        Directory where the plot will be saved.
    fontsize : int, optional
        Font size for the plot labels and titles. Default is 15.
    legend : bool, optional
        Whether to display the legend on the plot. Default is False.
    transparent : bool, optional
        Whether to make the background of the plot transparent.
        Default is True.
    """
    # Unpack metrics
    train_loss, val_loss, train_top1_acc, val_top1_acc = (
        ud.get_generator_history(
            os.path.join(model_dir, f"{model_type}/{run_id}")
        )
    )

    # Generate x-axis values
    x_epochs = np.arange(len(train_loss)) + 1

    # Create figure and axes
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(6, 3.8))
    ax2 = ax1.twinx()

    # Plot loss
    ax1.plot(x_epochs, train_loss, "r", lw=1.5, label="Train Loss")
    ax1.plot(x_epochs, val_loss, "b", lw=1.5, label="Val Loss")

    # Plot top-1 accuracy
    ax2.plot(x_epochs, train_top1_acc, "r--", lw=1.5, label="Train Acc.")
    ax2.plot(x_epochs, val_top1_acc, "b--", lw=1.5, label="Val Acc.")

    # Combine and add legends
    if legend:
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(
            lines_1 + lines_2, labels_1 + labels_2,
            loc="center right", fontsize=fontsize,
        )

    # Axis settings
    ax1.set_xlim([0, len(x_epochs) + 1])
    ax1.set_xlabel("Epoch", fontsize=fontsize)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=fontsize)
    ax2.set_ylabel("Top-1 Accuracy", fontsize=fontsize)
    ax1.tick_params(axis="both", which="major", labelsize=fontsize)
    ax2.tick_params(axis="both", which="major", labelsize=fontsize)

    # Save figure
    save_dir = os.path.join(save_dir, f"{model_type}/{run_id}")
    os.makedirs(save_dir, exist_ok=True)
    
    plt.tight_layout()
    save(
        fig,
        filename=f"{save_dir}/training_history.png",
        transparent=transparent
    )

    return None


def plot_fitted_curve(
    x,
    y,
    params,
    method,
    filename,
    fontsize=12,
):
    """Plots the fitted curve along with the original loss curve.
    
    Parameters
    ----------
    x : np.ndarray
        Epoch array values. Shape is (n_epochs,).
    y : np.ndarray
        Loss array values. Shape is (n_epochs,).
    params : list
        Parameters of the fitted curve.
    method : str
        Method used for fitting the curve.
        Should be either "exp_fit" or "power_fit".
    filename : str
        Path where the figure will be saved.
    fontsize : int, optional
        Font size for the figure. Default is 12.
    """
    # Validate inputs
    if method not in ["exp_fit", "power_fit"]:
        raise ValueError("Method should be either 'exp_fit' or 'power_fit'.")
    
    # Select function and label based on fitting method
    if method == "exp_fit":
        func = ua._exp_decay
        lbl = "Exponential Fit"
    elif method == "power_fit":
        func = ua._power_law
        lbl = "Power-law Fit"

    # Plot the original and fitted curves
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
    ax.plot(x, y, "b", lw=1.5, label="Loss")
    ax.plot(x, func(x, *params), "r--", lw=1.5, label=lbl)
    ax.axhline(
        params[0],
        color="k", ls=":", lw=1.5,
        label="Estimated Asymptote",
    )
    ax.set_xlabel("Epoch", fontsize=fontsize)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=fontsize)
    ax.tick_params(axis="both", which="major", labelsize=fontsize)
    ax.legend(loc="upper right", fontsize=fontsize - 2)
    plt.tight_layout()
    save(fig, filename, transparent=True)
    
    return None


def plot_convergence_metrics(
    metrics,
    label,
    color_palette,
    filename,
):
    """Plots convergence metrics (e.g., log-relative loss, convergence rates).
    
    Parameters
    ----------
    metrics : np.ndarray
        Array of metrics across different model types.
        Shape must be (n_models, n_epochs).
    label : str
        Metric name. Used as y-axis label for the plot.
    color_palette : dict
        Dictionary mapping model types to colors.
    filename : str
        Path where the figure will be saved.
    """
    # Validate inputs
    if metrics.ndim != 2:
        raise ValueError("Metrics array should be 2-dimensional.")
    n_models = metrics.shape[0]

    # Plot the metrics
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
    x = np.arange(1, metrics.shape[1])
    for i in range(n_models):
        ax.plot(
            x, metrics[i][:-1],
            lw=1.5, marker="o", markersize=2,
            color=list(color_palette.values())[i],
            label=list(color_palette.keys())[i],
        )
    ax.set_xlabel("Epoch", fontsize=15)
    ax.set_ylabel(label, fontsize=15)
    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    plt.tight_layout()
    save(fig, filename, transparent=True)

    # NOTE: The last point is excluded for the input convergence metrics
    #       because it is meaningless to compute these metrics at the last epoch 
    #       (relative to itself).

    return None
