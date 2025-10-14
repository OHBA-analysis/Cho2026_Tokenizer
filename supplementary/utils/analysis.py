"""Functions for data analysis."""

import os
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from utils import data as ud
from utils import plotting as up


def _exp_decay(t, L_inf, A, k):
    """Exponential decay function."""
    return L_inf + A * np.exp(-k * t)


def _power_law(t, L_inf, A, p):
    """Power-law decay function."""
    return L_inf + A * (t ** -p)


def _estimate_asymptote(
    loss,
    method="auto",
    tail_frac=0.1,
    min_tail_epochs=10,
    plot_fit=False,
    plot_path=None,
):
    """Estimates the asymptotic loss value using selected methods.
    
    Parameters
    ----------
    loss : np.ndarray
        1D array of loss values over epochs. Shape is (n_epochs,).
    method : str, optional
        Method to estimate the asymptote. Options are:
        - "exp_fit": Fit an exponential decay curve.
        - "power_fit": Fit a power-law decay curve.
        - "tail_mean": Use the mean of the last few epochs.
        - "auto": Try "exp_fit" first, then "power_fit", and fallback to "tail_mean".
        Default is "auto".
    tail_frac : float, optional
        Fraction of the total epochs to consider as the tail region for fitting.
        Default is 0.1 (i.e., last 10% of epochs).
    min_tail_epochs : int, optional
        Minimum number of epochs to consider in the tail region. Default is 10.
    plot_fit : bool, optional
        Whether to plot the fitted curve along with the original loss curve.
        Default is False.
    plot_path : str, optional
        Path to save the plot if plot_fit is `True`. Default is None.

    Returns
    -------
    L_inf : float
        Estimated asymptotic loss value.
    """
    # Validate inputs
    if loss.ndim != 1:
        raise ValueError("Loss array should be 1-dimensional.")
    if method not in ["exp_fit", "power_fit", "tail_mean", "auto"]:
        raise ValueError(
            "Method should be 'exp_fit', 'power_fit', 'tail_mean', or 'auto'."
        )

    # Get epochs
    T = len(loss)
    t = np.arange(1, T + 1).astype(np.float32)

    # Define tail region for initial guesses and fallback
    tail_start = max(int(T * (1 - tail_frac)), T - min_tail_epochs)
    tail_start = min(tail_start, T - 2)  # ensure at least 2 points in tail
    tail_y = loss[tail_start:]

    # Define simple fallback
    def fallback():
        return np.mean(loss[-min_tail_epochs:])  # mean of last few epochs

    # Estimate asymptote based on the selected method
    if method == "tail_mean":
        return fallback()
    
    if method in ["exp_fit", "auto"]:
        # Initial guesses
        L_inf_init = np.min(tail_y)
        A_init = (
            loss[0] - L_inf_init if loss[0] > L_inf_init
            else (np.max(loss) - L_inf_init)
        )
        k_init = 0.1

        # Define bounds
        lower = [0.0, 0.0, 1e-8]
        upper = [np.max(loss) * 1.5, np.max(loss) * 5, 10.0]

        # Fit exponential decay
        try:
            popt, _ = curve_fit(
                _exp_decay, t, loss,
                p0=[L_inf_init, A_init, k_init],
                bounds=(lower, upper),
                maxfev=10_000,
            )
            L_inf = popt[0]
            
            # Sanity checks
            assert np.isfinite(L_inf) and (L_inf <= np.min(loss)), "Invalid asymptote."            
            print("Exponential fit succeeded.")
            if plot_fit:
                up.plot_fitted_curve(
                    t, loss, popt, method="exp_fit",
                    filename=plot_path,
                )
            return L_inf
        
        except Exception as e:
            print(f"Exponential fit failed: {e}.")
            pass

    if method in ["power_fit", "auto"]:
        # Initial guesses
        L_inf_init = np.min(tail_y)
        A_init = (
            loss[0] - L_inf_init if loss[0] > L_inf_init
            else (np.max(loss) - L_inf_init)
        )
        p_init = 0.5

        # Define bounds
        lower = [0.0, 0.0, 1e-6]
        upper = [np.max(loss) * 1.5, np.max(loss) * 5, 10.0]

        # Fit power law
        try:
            popt, _ = curve_fit(
                _power_law, t, loss,
                p0=[L_inf_init, A_init, p_init],
                bounds=(lower, upper),
                maxfev=10_000,
            )
            L_inf = popt[0]
            # Sanity checks
            assert np.isfinite(L_inf) and (L_inf <= np.min(loss)), "Invalid asymptote."
            print("Power-law fit succeeded.")
            if plot_fit:
                up.plot_fitted_curve(
                    t, loss, popt, method="power_fit",
                    filename=plot_path,
                )
            return L_inf
        
        except Exception as e:
            print(f"Power-law fit failed: {e}.")
            pass

    print("Returning fallback asymptote estimate.")
    return fallback()


def compute_log_relative_loss(
    model_type,
    run_id,
    model_dir,
    loss_name="val_loss",
    plot_fit=False,
    plot_dir=None,
):
    """Computes the log-relative loss and convergence rate for a given model.

    Parameters
    ----------
    model_type : str
        Type of the model (e.g., "causal", "noncausal").
    run_id : int
        Run ID of the model.
    model_dir : str
        Directory where the trained models are stored.
    loss_name : str, optional
        Name of the loss to analyze (e.g., "train_loss", "val_loss").
        Default is "val_loss".
    plot_fit : bool
        Whether to plot the fit results. Default is False.
    plot_dir : str
        Directory where the plots will be saved if plot_fit is `True`.

    Returns
    -------
    log_rel_loss : np.ndarray
        Log-relative loss values over epochs. Shape is (n_epochs,).
    convergence_rate : np.ndarray
        Convergence rate values over epochs. Shape is (n_epochs - 1,).
    """
    # Get the loss curve
    history = ud.get_generator_history(
        model_dir=os.path.join(model_dir, model_type, str(run_id))
    )
    if loss_name == "train_loss":
        loss = history[0]
    elif loss_name == "val_loss":
        loss = history[1]

    # Smooth the curve using Savitzky-Golay filter
    if loss_name == "train_loss":
        window_length = 3
        poly_order = 2
    if loss_name == "val_loss":
        window_length = 9
        poly_order = 3
    loss = savgol_filter(loss, window_length, poly_order)
    # NOTE: The monotonicity of the smoothed curve is not guaranteed.
    #       However, it can be enforced by using a naive minimum accumulation 
    #       or isotonic regression (PAVA algorithm).

    # Estimate asymptotic loss
    asymptotic_loss = _estimate_asymptote(
        loss,
        method="auto",
        plot_fit=plot_fit,
        plot_path=os.path.join(plot_dir, f"asymptote_fit_{model_type}.png"),
    )
    print(f"Final loss for {model_type}: {loss[-1]:.4f}")
    print(f"Estimated asymptotic loss for {model_type}: {asymptotic_loss:.4f}")

    # Compute log-relative loss
    rel_loss = (loss - asymptotic_loss) / (loss[0] - asymptotic_loss)
    log_rel_loss = np.log(rel_loss + 1e-8)

    # Get slope of log-relative loss
    convergence_rate = -np.diff(log_rel_loss)

    return log_rel_loss, convergence_rate
