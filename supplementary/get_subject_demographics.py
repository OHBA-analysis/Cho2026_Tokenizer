"""Script for inspecting the subject demographics of the MEG Cam-CAN datasets."""

# Set up dependencies
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from glob import glob


if __name__ == "__main__":
    # ---------- Settings ---------- #
    # Set directory paths
    BASE_DIR = "/well/woolrich/users/olt015/Cho2026_Tokenizer"
    PROJ_DIR = "/well/win-camcan/shared"
    
    data_dir = os.path.join(PROJ_DIR, "spring23/src")
    meta_dir = os.path.join(PROJ_DIR, "participants.tsv")
    plot_dir = os.path.join(BASE_DIR, "plots/data")
    os.makedirs(plot_dir, exist_ok=True)

    # Read demographics data
    demographics = pd.read_csv(meta_dir, sep="\t")

    # Get subject IDs
    data_files = sorted(glob(f"{data_dir}/*/sflip_parc-raw.fif"))
    full_ids = [file.split("/")[-2] for file in data_files]
    print(f"Total Number of subjects: {len(full_ids)}")

    # Get subject IDs of the training subset for tokenizer
    rng = np.random.default_rng(seed=813)
    idx = rng.choice(len(full_ids), size=50, replace=False)  # select 50 random subjects
    print("Data Subset Subject Idx: ", idx)

    subset_ids = [full_ids[i] for i in idx]
    print(f"Number of subjects in subset: {len(subset_ids)}")

    subject_ids = [full_ids, subset_ids]
    labels = ["full", "subset"]

    for ids, label in zip(subject_ids, labels):
        print(f"Visualizing demographics for the '{label}' data...")

        # ---------- Get Demographic Features ---------- #
        ages = np.array([
            demographics.loc[demographics["participant_id"] == id]["age"].values[0]
            for id in ids
        ])

        sexes = np.array([
            "Female" if sex == "FEMALE" else "Male"
            for sex in [demographics.loc[demographics["participant_id"] == id]["sex"].values[0] for id in ids]
        ])

        handedness = np.array([
            "Right" if hand > 0 else "Left" if hand < 0 else "Other"
            for hand in [demographics.loc[demographics["participant_id"] == id]["hand"].values[0] for id in ids]
        ])
        
        if "Other" in handedness:
            print("Number of 'Other' handedness subjects:", np.sum(handedness == "Other"))

        # Sort ages by age ranges
        age_categories = np.empty(ages.shape, dtype=object)
        age_intervals = [[18, 28], [28, 38], [38, 48], [48, 58],
                        [58, 68], [68, 78], [78, 89]]
        for n, (start, end) in enumerate(age_intervals):
            if n == 0 or n == 3:
                mask = np.logical_and(ages >= start, ages <= end)
            else:
                mask = np.logical_and(ages > start, ages <= end)
            age_categories[mask] = f"{start}-{end}"
        ages = age_categories # reassign variable

        # ---------- Create Dataframes ---------- #
        df_age = pd.DataFrame({"Age": ages})
        df_age["Age"] = pd.Categorical(
            df_age["Age"],
            categories=["18-28", "28-38", "38-48", "48-58", "58-68", "68-78", "78-89"],
            ordered=True,
        )
        
        df_sex = pd.DataFrame({"Sex": sexes})
        df_sex["Sex"] = pd.Categorical(
            df_sex["Sex"],
            categories=["Female", "Male"],
            ordered=True,
        )

        df_handedness = pd.DataFrame({"Handedness": handedness})
        df_handedness["Handedness"] = pd.Categorical(
            df_handedness["Handedness"],
            categories=["Left", "Right", "Other"],
            ordered=True,
        )

        # ---------- Visualization ---------- #
        # Set visualization hyperparameters
        fontsize=13
        facecolor="#114B5F"
        edgecolor="none"

        # Visualize demographics
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3))

        sns.countplot(
            data=df_age, x="Age",
            color=facecolor, edgecolor=edgecolor,
            width=0.7, alpha=0.75, ax=ax[0],
        )
        ax[0].tick_params(axis="x", labelrotation=45)
        ax[0].set_xlabel("Age (years)", fontsize=fontsize)

        sns.countplot(
            data=df_sex, x="Sex",
            color=facecolor, edgecolor=edgecolor,
            width=0.5, alpha=0.75, ax=ax[1],
        )
        ax[1].set_xlabel("Sex", fontsize=fontsize)

        sns.countplot(
            data=df_handedness, x="Handedness",
            color=facecolor, edgecolor=edgecolor,
            width=0.5, alpha=0.75, ax=ax[2],
        )
        ax[2].set_xlabel("Handedness", fontsize=fontsize)

        for axis in ax:
            axis.set_ylabel("Count", fontsize=fontsize)
            axis.tick_params(
                axis="both", which="major", width=1.5, labelsize=fontsize
            )
            axis.spines[["top", "right"]].set_visible(False)
            axis.spines[["bottom", "left"]].set_linewidth(1.5)

        plt.tight_layout()
        fig.savefig(
            f"{plot_dir}/demographics_{label}.png",
            dpi=300, bbox_inches="tight", transparent=False,
        )
        plt.close(fig)

    print("Visualization complete.")
