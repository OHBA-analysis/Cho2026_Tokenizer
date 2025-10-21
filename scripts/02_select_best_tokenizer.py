"""Script for selecting the best tokenizer model run."""

# Import packages
import os
import pickle
import numpy as np
from sys import argv


if __name__ == "__main__":
    # ---------- User Inputs ---------- #
    # Set user arguments
    if len(argv) != 3:
        raise ValueError("Please provide the model type and run IDs as arguments.")
    model_type = argv[1]  # model type
    run_ids = list(map(int, argv[2].split("-")))  # range of runs to compare
    print(f"[INFO] Model type: {model_type}, Run IDs: run{run_ids[0]} - run{run_ids[1]}")

    # ---------- Directories ---------- #
    tokenizer_dir = f"models/tokenizer/{model_type}/{{0}}"
    history_path = os.path.join(tokenizer_dir, "history.pkl")

    # ---------- Compare Runs ---------- #
    # If the range of runs is larger than 10, group 10 model runs as one set
    if np.diff(run_ids) + 1 > 10:
        intervals = [
            [i, min(i + 9, run_ids[1])]
            for i in range(run_ids[0], run_ids[1] + 1, 10)
        ]
    else: intervals = [run_ids]

    # Get the best model run
    best_runs, best_losses = [], []
    for i, (start, end) in enumerate(intervals):
        print(f"Loading loss (run{start}-run{end}) ...")
        losses = []
        run_id_list = np.arange(start, end + 1)
        for id in run_id_list:
            with open(history_path.replace("{0}", str(id)), "rb") as f:
                history = pickle.load(f)
            losses.append(history["loss"][-1])
        best_losses.append(np.min(losses))
        best_runs.append(run_id_list[losses.index(best_losses[i])])
        print(f"\tFinal loss (n={len(run_id_list)}): {losses}")
        print(f"\tMean ± SD: {np.mean(losses):.8f} ± {np.std(losses):.8f}")
        print(f"\tBest run: run{best_runs[i]}")
        print(f"\tBest training loss: {best_losses[i]}")

    # Identify the optimal run from all the best runs
    opt_loss = np.min(best_losses)
    opt_run = best_runs[np.argmin(best_losses)]
    print(f"The lowest training loss is {opt_loss} from run{opt_run}.")

    print("Selection complete.")
