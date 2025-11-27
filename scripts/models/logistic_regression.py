"""Basic logistic regression classifier model."""

# Import packages
import logging
import tensorflow as tf
from osl_dynamics.utils.misc import get_argument, replace_argument
from osl_foundation.inference.layers import NormalizationLayer
from osl_foundation.models.base import BaseModel


_logger = logging.getLogger("osl-foundation")


class CrossEntropyLossLayer(tf.keras.layers.Layer):
    """
    Layer for calculating the cross-entropy loss.

    Parameters
    ----------
    top_k : list, optional
        List of top k values to calculate the accuracy for.
        By default only top 1 accuracy is calculated.
    """

    def __init__(
        self,
        top_k: list = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.top_k = top_k or [1]

    def call(self, inputs, **kwargs):
        y_pred, y_true = inputs
        # y_true.shape = (batch_size,)
        # y_pred.shape = (batch_size, n_task_classes)

        loss = tf.keras.metrics.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=True
        )

        # Reduce per sample loss to a scalar value
        loss = tf.reduce_mean(loss)
        self.add_loss(loss)

        for k in self.top_k:
            accuracy = tf.keras.metrics.sparse_top_k_categorical_accuracy(
                y_true, y_pred, k=k
            )
            self.add_metric(accuracy, name=f"top_{k}")

        return tf.expand_dims(loss, -1), y_pred


class LogisticRegression(BaseModel):
    """
    Logistic regression classifier model.

    Parameters
    ----------
    config : Config
        Config object.
    """

    def build_model(self) -> None:
        self.model = self._build_model()

    def fit(
        self,
        *args,
        step_size: int = None,
        **kwargs,
    ) -> None:
        """
        First tokenizes the data and then fits the model.

        Parameters
        ----------
        *args : list
            Positional arguments to pass to the model's fit method.
        step_size : int, optional
            Step size when creating the dataset.
        **kwargs : dict
            Keyword arguments to pass to the model's fit method.
        """
        x = get_argument(self.model.fit, "x", args, kwargs)

        validation_split = get_argument(
            self.model.fit, "validation_split", args, kwargs
        )

        # If step_per_epoch is passed, repeat the dataset indefinitely
        steps_per_epoch = get_argument(self.model.fit, "steps_per_epoch", args, kwargs)

        dataset = self.make_dataset(
            x,
            shuffle=True,
            concatenate=True,
            step_size=step_size,
            drop_last_batch=True,
            validation_split=validation_split,
        )
        if validation_split is None:
            args, kwargs = replace_argument(
                self.model.fit,
                "x",
                dataset,
                args,
                kwargs,
            )
        else:
            args, kwargs = replace_argument(
                self.model.fit,
                "x",
                dataset[0],
                args,
                kwargs,
            )
            args, kwargs = replace_argument(
                self.model.fit,
                "validation_data",
                dataset[1],
                args,
                kwargs,
            )

        super().fit(*args, **kwargs)

    def _flatten_data(self, data):
        config = self.config.model_config
        if config.feature_type == "baseline":
            x = tf.reshape(
                data, shape=[-1, config.sequence_length * config.n_channels]
            )
            # x.shape = (batch_size, n_samples * n_channels)
        if config.feature_type in ["zero_shot", "fine_tune"]:
            x = tf.reshape(
                data, shape=[-1, config.n_channels * config.model_dim]
            )
            # x.shape = (batch_size, n_channels * model_dim)
        return x

    def _build_model(self) -> tf.keras.Model:
        config = self.config.model_config

        # ---------- Inputs ---------- #
        if config.feature_type == "baseline":
            input_shape = (config.sequence_length, config.n_channels)
        elif config.feature_type in ["zero_shot", "fine_tune"]:
            input_shape = (
                config.latent_sequence_length, config.n_channels, config.model_dim
            )

        data_input = tf.keras.layers.Input(
            shape=input_shape,
            dtype=tf.float32,
            name="data",
        )
        # data.shape = (batch_size, latent_sequence_length, n_channels, model_dim); or
        # data.shape = (batch_size, n_channels, model_dim)

        task_label = tf.keras.layers.Input(
            shape=(), dtype=tf.int32, name="task_label"
        )
        # task_label.shape = (batch_size,)

        # ---------- Initialize layers ---------- #
        if config.feature_type != "baseline":
            linear_projection_layer = tf.keras.layers.Dense(
                1, name="linear_projection"
            )
        norm_layer = NormalizationLayer(config.norm_type, config.n_groups)
        prediction_head_layer = tf.keras.layers.Dense(
            config.n_task_classes,
            name="prediction_head",
        )
        loss_layer = CrossEntropyLossLayer(config.top_k, name="loss")

        # ---------- Forward Pass ---------- #
        
        # Set input data
        data = data_input

        # Reduce the time dimension if using non-baseline features
        if config.feature_type != "baseline":
            data = tf.transpose(data, perm=[0, 2, 3, 1])
            data = linear_projection_layer(data)
            data = tf.squeeze(data, axis=-1)
            # data.shape = (batch_size, n_channels, model_dim)

        # Flatten the input data features
        x = self._flatten_data(data)
        # x.shape = (batch_size, n_features)

        # Apply layer normalization
        x = norm_layer(x)
        # x.shape = (batch_size, n_features)

        # Get the prediction of the task label
        y_pred = prediction_head_layer(x)
        # y_pred.shape = (batch_size, n_task_classes)

        # Calculate the loss
        loss, y_pred = loss_layer([y_pred, task_label])

        # ---------- Model ---------- #
        return tf.keras.Model(
            inputs=[data_input, task_label],
            outputs=[loss, y_pred],
            name="logistic_regression",
        )
