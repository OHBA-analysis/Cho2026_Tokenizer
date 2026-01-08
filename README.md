# Cho2025_Tokenizer

💡 Please email SungJun Cho at sungjun.cho@ndcn.ox.ac.uk or simply open a GitHub issue if you have any questions or concerns.

## ⚡️ Getting Started

This repository contains all the scripts necessary to reproduce the analyses and figures presented in the manuscript. It is divided into three main directories.

| Directory       | Description                                                                                 |
| :-------------- | :------------------------------------------------------------------------------------------ |
| `scripts`       | Scripts for training the tokenizer and MEG-GPT models and conducting subsequent analyses.   |
| `supplementary` | Scripts for additional data inspection, post-hoc analysis, and visualization.               |
| `testing`       | Scripts for debugging and testing.                                                          |

For detailed descriptions of the scripts in each directory, please consult the README file located within each respective folder.

In addition, the `models` directory contains configuration files specifying the hyperparameters used for all models trained in this work. Corresponding tables summarizing these hyperparameters are provided within the same directory.

## 🎯 Requirements

This repository builds on the [`osl-foundation`](https://github.com/OHBA-analysis/osl-foundation) software package, which provides the `MEG-GPT` model and its associated tokenizers. To start, please install `osl-foundation` and set up its environment by following the installation guide [here](https://github.com/OHBA-analysis/osl-foundation/blob/main/README.md).

The scripts used in this paper rely on the following dependencies:

```
python==3.10.4
tensorflow==2.11.0
tensorflow-probability==0.19.0
osl-dynamics==2.1.8
osl-foundation==0.0.1
```

Once these steps are complete, you can clone or download this repository to your preferred directory, and you're ready to begin!

### Notes on hardware

All scripts in this repository were executed on the Oxford Biomedical Research Computing ([BMRC](https://www.medsci.ox.ac.uk/for-staff/resources/bmrc)) servers. Our experiments were run using two NVIDIA GPUs (V100 or A100) with CUDA 11.7.0.

## 🪪 License
Copyright (c) 2025 [SungJun Cho](https://github.com/scho97) and [OHBA Analysis Group](https://github.com/OHBA-analysis). `Cho2025_Tokenizer` is a free and open-source software licensed under the [MIT License](https://github.com/OHBA-analysis/Cho2025_Tokenizer/blob/main/LICENSE).
