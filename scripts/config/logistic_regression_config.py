"""Module providing the LogisticRegression model configuration."""

# Import packages
from dataclasses import dataclass
from osl_foundation.config.base import BaseModelConfig


@dataclass
class LogisticRegressionModelConfig(BaseModelConfig):
    name: str = "logistic_regression"

    # ---------- Model parameters ---------- #
    feature_type: str = "baseline"
    model_dim: int = None
    latent_sequence_length = None
    norm_type: str = "layer"
    n_groups: int = None
    n_task_classes: int = 4

    # ---------- Loss parameters ---------- #
    top_k: list = None

    def validate(self) -> None:
        super().validate()
        self._validate_model_parameters()

    def _validate_model_parameters(self) -> None:
        assert (
            self.feature_type in ["baseline", "zero_shot", "fine_tune"],
            "feature_type must be either 'baseline', 'zero_shot', or 'fine_tune'"
        )
        assert self.model_dim is not None, "model_dim must be set"
        assert (
            self.latent_sequence_length is not None
        ), "latent_sequence_length must be set"
        assert (
            self.latent_sequence_length > 0
        ), "latent_sequence_length must be greater than 0"
        assert self.n_groups is None or self.n_groups > 0
        if self.n_groups is not None:
            assert (
                self.model_dim % self.n_groups == 0
            ), "model_dim must be divisible by n_groups"
        assert self.n_task_classes > 1, "n_task_classes must be greater than 1"

    def set_config(self, config: dict) -> None:
        self.name = config.get("name", "logistic_regression")
        self._set_model_parameters(config.get("model_parameters", {}))
        self._set_loss_parameters(config.get("loss_parameters", {}))

    def _set_model_parameters(self, config: dict) -> None:
        self.feature_type = config.get("feature_type", "baseline")
        self.model_dim = config.get("model_dim", 64)
        self.latent_sequence_length = config.get(
            "latent_sequence_length", self.sequence_length // 2
        )
        self.norm_type = config.get("norm_type", "layer")
        self.n_groups = config.get("n_groups", None)
        self.n_task_classes = config.get("n_task_classes", 4)

    def _set_loss_parameters(self, config: dict) -> None:
        self.top_k = config.get("top_k", None)
