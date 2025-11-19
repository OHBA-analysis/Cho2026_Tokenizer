"""Module providing model creation and loading functionalities.

This module is adapted from the osl_foundation framework (see osl_foundation/models/__init__.py)
for fine-tuning on the Wakeman-Henson task dataset.
"""

# Import packages
import pickle
import tensorflow as tf
from typing import Union
from osl_foundation.config import Config
from osl_foundation.models.tokenizers import (
    OSLTokenizer,
    MuTransformTokenizer,
    StandardQuantileTokenizer,
)
from config import get_config
from models.logistic_regression import LogisticRegression
from models.meg_gpt_subject_emb import MEGGPT_SE
from models.meg_gpt_fine_tune import MEGGPT_FT


models = {
    "osl_tokenizer": OSLTokenizer,
    "mu_transform_tokenizer": MuTransformTokenizer,
    "standard_quantile_tokenizer": StandardQuantileTokenizer,
    "meg_gpt_subject_emb": MEGGPT_SE,
    "meg_gpt_fine_tune": MEGGPT_FT,
    "logistic_regression": LogisticRegression,
}


def create_model(
    config: Union[Config, dict, str],
    save_dir: str = None,
    strategy: tf.distribute.Strategy = None,
):
    """Create a model based on the configuration.

    Parameters
    ----------
    config : Union[Config, dict, str]
        String, path to a configuration file, dictionary, or a Config object.
    save_dir : str, optional
        Path to a directory where the configuration will be saved.
        Defaults to None, in which case the configuration will not be saved.
    strategy : tf.distribute.Strategy, optional
        TensorFlow distribution strategy for distributed training.
        Defaults to None, in which case the config strategy will be used.

    Returns
    -------
    model
        Model object.
    """
    if not isinstance(config, Config):
        config = get_config(config)

    if save_dir:
        config.save_config(save_dir)

    if config.model_config.name not in models:
        raise ValueError(
            f"Model {config.model_config.name} not implemented. "
            f"Options are {', '.join(models.keys())}"
        )

    return models[config.model_config.name](config, strategy=strategy)


def load_model(
    model_dir: str,
    checkpoint: str = None,
    strategy: tf.distribute.Strategy = None,
):
    """Load a saved model from a directory.

    Parameters
    ----------
    model_dir : str
        Directory containing the saved model.
    checkpoint : str, optional
        Path to the checkpoint file. If `latest`, the latest checkpoint will be used.
        Defaults to None, in which case the weights will be loaded from `weights.h5`.
    strategy : tf.distribute.Strategy, optional
        TensorFlow distribution strategy for distributed training.
        Defaults to None, in which case the config strategy will be used.

    Returns
    -------
    model
        Model object.
    """
    config = get_config(configuration=f"{model_dir}/config.yml")
    model = models[config.model_config.name](config, strategy=strategy)
    
    if checkpoint:
        cp = tf.train.Checkpoint(model=model.model, optimizer=model.model.optimizer)
        if checkpoint == "latest":
            checkpoint_path = tf.train.latest_checkpoint(f"{model_dir}/checkpoints")
        else:
            checkpoint_path = checkpoint
        with model.model.distribute_strategy.scope():
            cp.restore(checkpoint_path).expect_partial()
    else:
        model.load_weights(f"{model_dir}/weights.h5")

    try:
        with open(f"{model_dir}/history.pkl", "rb") as f:
            model.history = pickle.load(f)
    except FileNotFoundError:
        pass

    if model.config.model_config.name == "osl_tokenizer":
        with open(f"{model_dir}/vocab.pkl", "rb") as f:
            model.vocab = pickle.load(f)
    
    return model
