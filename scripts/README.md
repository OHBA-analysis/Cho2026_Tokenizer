# Analysis Scripts

### 🔎 Overview

This directory contains scripts for model training and data analysis presented in the paper.
The workflows are built around three publicly available MEG datasets: Cam-CAN, Nottingham MEGUK, and Wakeman–Henson.
Please refer to the paper for more details on datasets, preprocessing, and experimental design.

The scripts are organized into two main categories:

1. Model Training
   * For training tokenizers and MEG-GPT models
   * For fine-tuning and downstream decoding
2. Data Analysis
   * For analyzing and evaluating trained models

Detailed descriptions of each script are provided below.

## ⚙️ Model Training

We have eight main scripts for tokenizer training, MEG-GPT pre-training with different tokenizer types, and downstream decoding.

For learnable tokenizers, we trained each model 10 times and selected the best-performing one. MEG-GPT pre-training and downstream decoding were each performed once, using a fixed random seed for a fair comparison.

| Scripts                       | Description                                                                               |
| :---------------------------- | :---------------------------------------------------------------------------------------- |
| `01_train_tokenizer.py`       | Trains different types of tokenizers on a randomly subsampled Cam-CAN dataset.            |
| `02_select_best_tokenizer.py` | Selects the best tokenizer across multiple training runs.                                 |
| `03_tokenize_data.py`         | Splits datasets into train and validation sets and tokenizes them as an input to MEG-GPT. |
| `05_train_generator.py`       | Pre-trains MEG-GPT models using different tokenizer types on the Cam-CAN dataset.         |
| `06_generate_data.py`         | Generates and saves synthetic MEG data using pre-trained MEG-GPT models.                  |
| `07_train_hmm.py`             | Trains univariate TDE-HMM models on MEG time series of selected channels.                 |
| `13_wh_tokenize_data.py`      | Tokenizes the Wakeman-Henson dataset for downstream decoding analyses.                    |
| `14_wh_fine_tune.py`          | Performs end-to-end fine-tuning of MEG-GPT on the Wakeman-Henson dataset.                 |

**NOTE:** For the μ-transform tokenizer, we used four variants, differing in the predefined vocabulary size:
* `mu_transform` (256 tokens)
* `mu_transform_big` (182 tokens)
* `mu_transform_small` (108 tokens)
* `mu_transform_tiny` (54 tokens)

### 🧑‍🔧 Training for downstream decoding task

The `config` and `models` subdirectories contain codes for building configuration objects and network architectures of logistic regression classifiers and fine-tuned MEG-GPT models used in downstream decoding tasks. These components are built on top of the `osl-foundaiton` software package and adapted from its own `config` and `models` modules.

## 🧐 Data Analysis

For the data analysis, we have nine scripts that produce the main results reported in the paper:

| Scripts                                | Description                                                                                                          | Figures |
| :------------------------------------- | :------------------------------------------------------------------------------------------------------------------- | :------ |
| `04_analyze_tokenizer.py`              | Analyzes token distributions and signal reconstruction accuracy across tokenizers.                                   | 4, 5    |
| `08_token_prediction_analysis.py`      | Computes token prediction accuracy of the pretrained MEG-GPT models against the ground-truth labels.                 | -       |
| `09_token_prediction_visualization.py` | Visualizes token prediciton accuracy across different tokenizer types.                                               | 6       |
| `10_static_spectral.py`                | Computes and visualizes static spectral features of real and generated MEG data.                                     | 7       |
| `10-1_static_spectral_lmm.r`           | Fits linear mixed-effects model and performs statistical comparisons of static spectral features across tokenizers.  | 7       |
| `11_dynamic_spectral.py`               | Computes and visualizes dynamic spectral features of real and generated MEG data.                                    | 8       |
| `11-1_dynamic_spectral_lmm.r`          | Fits linear mixed-effects model and performs statistical comparisons of dynamic spectral features across tokenizers. | 8       |
| `12_subject_fingerprinting.py`         | Performs subject fingerprintng analysis and quantifies inter-subject similarity for different tokenizers.            | 9       |
| `15_wh_decoding.py`                    | Performs downstream task classification analysis.                                                                    | 10      |

### 🙋‍♂️ FAQ: What about the `utils` subdirectory?
The `utils` subdirectory contains essential functions required to run the scripts summarized above. Each script in `utils` includes multiple 
functions. These functions are self-explanatory and include detailed annotations, so their descriptions are not repeated here.
