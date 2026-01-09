"""Script for the token prediction analysis."""

# Import packages
import os
import pickle
import numpy as np
import tensorflow as tf
from sys import argv
from tqdm.auto import trange
from osl_dynamics.data import load_tfrecord_dataset
from osl_dynamics.inference import tf_ops
from osl_foundation import load_model


if __name__ == "__main__":
    # ---------- User Inputs ---------- #
    # Set GPU memory growth
    tf_ops.gpu_growth()

    # Set user arguments
    if len(argv) != 4:
        raise ValueError(
            "Please provide the model type, best tokenizer run ID, and best generator run ID as arguments."
        )
    model_type = argv[1]
    best_tk_run_id = int(argv[2])
    best_gt_run_id = int(argv[3])

    # Set hyperparameters
    seq_len = 81  # sequence length used in training MEG-GPT models
    n_channels = 52  # number of channels in the data
    loss_sequence_length = 8  # the number of tokens to predict
    tk_sequence_length = 200  # sequence length used in training the tokenizer
    kernel_size = 10  # convolution kernel width used in the tokenizer
    sub_batch_size = 100  # sub-batch size for processing data

    # ---------- Directories ---------- #
    BASE_DIR = "/well/woolrich/users/olt015/Cho2026_Tokenizer"
    DATA_DIR = os.path.join(BASE_DIR, "tokenized_data_tfrecords")
    MODEL_DIR = os.path.join(BASE_DIR, "models/generator")

    tokenized_data_tf_dir = f"{DATA_DIR}/{model_type}/{best_tk_run_id}"
    generator_dir = f"{MODEL_DIR}/{model_type}/{best_gt_run_id}"

    # ---------- Load Generator ---------- #
    print("Loading generator ...")
    generator = load_model(generator_dir, checkpoint="latest")
    tokenizer_vocab = generator.tokenizer.vocab

    # Get tokenizer vocabulary details
    if model_type in ["causal", "noncausal"]:
        label_map = tokenizer_vocab["label_map"]
        token_order = tf.cast(tokenizer_vocab["token_order"], tf.int32)
        default_id = tf.cast(
            np.where(label_map == 0)[0][0],
            tf.int32,
        )  # use the first token ID with zero counts to replace unseen tokens (i.e., 0-th token)

    # ---------- Load TFRecord Data for Tokens (Validation) ---------- #
    print("Loading TFRecord data for tokens (validation) ...")

    _, val_data = load_tfrecord_dataset(
        tokenized_data_tf_dir,
        batch_size=100,
        shuffle=False,
        buffer_size=2000,
        drop_last_batch=False,
        concatenate=True,
    )
    
    input_tokens = []  # token ID integers
    input_labels = []  # subject labels
    input_raws = []  # raw data
    for batch in val_data:
        input_tokens.extend(batch["data"])
        input_labels.extend(batch["session_id"])
        input_raws.extend(batch["raw_data"])

    input_tokens = tf.stack(input_tokens, axis=0)
    # shape: (n_total_sequences, sequence_length, n_channels)
    input_labels = tf.stack(input_labels, axis=0)
    # shape: (n_total_sequences, sequence_length)
    input_raws = tf.stack(input_raws, axis=0)
    # shape: (n_total_sequences, sequence_length, n_channels)

    print("Shape of input tokens:", input_tokens.shape)
    print("Shape of input labels:", input_labels.shape)
    print("Shape of input raw data:", input_raws.shape)

    # Define batch size
    batch_size = int(tf.shape(input_tokens)[0].numpy())
    print("Total batch size:", batch_size)

    # ---------- Define Utility Functions ---------- #  
    @tf.function
    def remap_tokens(tokens, token_order, default_id):
        """Remaps refactored tokens to original tokens.
        
        Parameters
        ----------
        tokens : tf.Tensor
            The tokenized data.
            Shape should be (batch_size, n_samples, n_channels).
        token_order : tf.Tensor (tf.int32)
            The token order array mapping refactored tokens to original tokens.
        default_id : tf.int32
            The default token ID to use for unseen tokens.
        
        Returns
        -------
        remapped_tokens : tf.Tensor
            The remapped tokens. Shape is (batch_size, n_samples, n_channels).
        """
        # Ensure data types
        tokens = tf.cast(tokens, tf.int32)
        token_order = tf.cast(token_order, tf.int32)
        default_id = tf.cast(default_id, tf.int32)

        # Shift tokens down by 1 (i.e., 0 to n_tokens-2; -1 for unseen tokens)
        tokens_shifted = tokens - 1
        valid = tokens_shifted >= 0

        # Clip invalid indices to zero to avoid out-of-bounds errors in tf.gather
        t_clipped = tf.where(valid, tokens_shifted, tf.zeros_like(tokens_shifted))

        # Remap tokens back to original tokens
        mapped = tf.gather(token_order, t_clipped)

        # Replace invalid positions with given token ID
        remapped_tokens = tf.where(
            valid, mapped, tf.fill(tf.shape(tokens), tf.cast(default_id, tf.int32))
        )

        return remapped_tokens

    def detokenize(tokenizer, sequence):
        """Reconstruct data from tokens.

        Parameters
        ----------
        tokenizer : osl_foundation.OSLTokenizer
            The tokenizer object used for detokenization.
        sequence : np.ndarray
            The tokenized data to be detokenized.
            Shape of sequence should be (batch_size, sequence_length, n_channels).

        Returns
        -------
        reconstructed_data : np.ndarray
            The detokenized data.
            Shape of reconstructed_data should be (batch_size, sequence_length, n_channels).
        """

        # Get data dimensions
        n_tokens = tokenizer.config.model_config.n_tokens
        n_channels = tokenizer.config.model_config.n_channels
        sequence_length = sequence.shape[1]

        # Get the token basis (kernel) layer
        token_basis_layer = tokenizer.model.get_layer("decoder").token_basis_layer

        # Reconstruct data
        x = np.transpose(sequence, axes=(0, 2, 1))
        # x.shape = (batch_size, n_channels, sequence_length)

        x = np.reshape(x, (-1, x.shape[-1]))
        # x.shape = (batch_size * n_channels, sequence_length)

        x = tf.one_hot(x, n_tokens).numpy().astype(np.float32)
        # x.shape = (batch_size * n_channels, sequence_length, n_tokens)

        x = np.sum(token_basis_layer(x), axis=-1)
        # x.shape = (batch_size * n_channels, sequence_length)

        x = np.reshape(x, (-1, n_channels, sequence_length))
        # x.shape = (batch_size, n_channels, sequence_length)

        x = np.transpose(x, axes=(0, 2, 1))
        # x.shape = (batch_size, sequence_length, n_channels)

        return x
    
    # ---------- Perform Prediction Token Analysis ---------- #
    # Initialize arrays
    collection_length = loss_sequence_length - (kernel_size // 2)
    raw_inputs = np.zeros((batch_size, collection_length, n_channels))
    recon_preds = np.zeros((batch_size, collection_length, n_channels))

    # Set variables for accuracy calculation
    accuracy_sum = 0
    prediction_size = batch_size * loss_sequence_length * n_channels

    # Set save path
    save_path = os.path.join(generator_dir, "token_pred_result.pkl")

    if os.path.exists(save_path):
        print(f"Results already exist at {save_path}. Skipping computation.")
    else:
        print(f"Performing prediction token analysis ...")

        with generator.model.distribute_strategy.scope():
            for i in trange(0, batch_size, sub_batch_size):
                # Get the input data
                input_token = input_tokens[i:i + sub_batch_size]
                # shape: (sub_batch_size, tk_sequence_length, n_channels)
                input_label = input_labels[i:i + sub_batch_size]
                # shape: (sub_batch_size, tk_sequence_length)
                input_raw = input_raws[i:i + sub_batch_size]
                # shape: (sub_batch_size, tk_sequence_length, n_channels)

                # Cast to appropriate data types
                input_token = tf.cast(input_token, tf.int32)
                input_label = tf.cast(input_label, tf.int32)
                input_raw = tf.cast(input_raw, tf.float32)

                # Get the predicted token probabilities using MEG-GPT
                model_inputs = {
                    "data": input_token,
                    "session_id": input_label,
                }
                outputs = generator.model(model_inputs)
                predicted_token = outputs[1]
                # shape: (sub_batch_size, loss_sequence_length, n_channels, n_tokens)
                
                # Get the predicted token IDs
                predicted_token = tf.argmax(predicted_token, axis=-1, output_type=tf.int32)
                # shape: (sub_batch_size, loss_sequence_length, n_channels)

                # Compute accuracy for the sub-batch
                accuracy_mask = tf.equal(
                    input_token[:, -loss_sequence_length:, :],
                    predicted_token,
                )  # shape: (sub_batch_size, loss_sequence_length, n_channels)
                accuracy_sum += int(tf.reduce_sum(tf.cast(accuracy_mask, tf.int32)).numpy())

                loop_range = [
                    input_token.shape[1] - loss_sequence_length,
                    input_token.shape[1] - kernel_size // 2,
                ]
                for n, l in enumerate(range(loop_range[0], loop_range[1])):
                    token_sequence = tf.concat([
                            input_token[:, :l, :],
                            tf.expand_dims(predicted_token[:, n, :], axis=1),
                            input_token[:, l + 1:, :]
                        ], axis=1,
                    )  # shape: (sub_batch_size, tk_sequence_length, n_channels)

                    # Remap refactored tokens to original tokens
                    if model_type in ["causal", "noncausal"]:
                        token_sequence = remap_tokens(
                            token_sequence, token_order, default_id
                        )

                    # Convert to numpy array
                    token_sequence = token_sequence.numpy()

                    # Detokenize the predicted sequences
                    print("Detokenizing the predicted sequences ...")

                    if model_type in ["causal", "noncausal"]:
                        # Perform detokenization
                        recon_sequence = detokenize(generator.tokenizer, token_sequence)
                        # shape: (sub_batch_size, tk_sequence_length, n_channels)
                    else:
                        # Perform detokenization
                        recon_sequence = generator.tokenizer.reconstruct_data(token_sequence)
                        # shape: (sub_batch_size, tk_sequence_length, n_channels)

                    # Collect reconstructed inputs and corresponding raw inputs
                    raw_inputs[i:i + sub_batch_size, n, :] = input_raw[:, l, :].numpy()
                    recon_preds[i:i + sub_batch_size, n, :] = recon_sequence[:, l, :]

            # Compute accuracy
            accuracy_score = accuracy_sum / prediction_size
            print("Token prediction accuracy:", accuracy_score)

        # Save the results
        outputs = {
            "accuracy_score": accuracy_score,
            "raw_inputs": raw_inputs,
            "recon_preds": recon_preds,
        }
        with open(save_path, "wb") as f:
            pickle.dump(outputs, f)
        print(f"Results saved to {save_path}.")

    print("Analysis complete.")
