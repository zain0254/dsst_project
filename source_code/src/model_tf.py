from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf

# Configure GPU memory growth before TensorFlow initializes GPU devices.
gpus = tf.config.list_physical_devices("GPU")

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

print(f"TensorFlow version: {tf.__version__}")
print(f"Available GPUs: {gpus}")
@dataclass
class TrainingResult:
    model: tf.keras.Model
    best_validation_loss: float
    epochs_run: int
    device: str


class LSTMAutoencoder(tf.keras.Model):
    """Sequence-to-sequence univariate LSTM autoencoder implemented in Keras."""

    def __init__(self, input_size: int = 1, hidden_size: int = 64, latent_size: int = 32,
                 window_size: int = 30):
        super().__init__(name="lstm_autoencoder")
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.latent_size = int(latent_size)
        self.window_size = int(window_size)
        self.encoder = tf.keras.layers.LSTM(hidden_size, return_sequences=False, name="encoder_lstm")
        self.to_latent = tf.keras.layers.Dense(latent_size, activation="tanh", name="latent")
        self.latent_to_hidden = tf.keras.layers.Dense(hidden_size, activation="tanh", name="decoder_initial_hidden")
        self.latent_to_cell = tf.keras.layers.Dense(hidden_size, activation="tanh", name="decoder_initial_cell")
        self.repeat_latent = tf.keras.layers.RepeatVector(window_size, name="repeat_latent")
        self.decoder = tf.keras.layers.LSTM(hidden_size, return_sequences=True, name="decoder_lstm")
        self.output_layer = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(input_size), name="reconstruction"
        )

    def call(self, inputs, training: bool = False):
        encoded = self.encoder(inputs, training=training)
        latent = self.to_latent(encoded)
        initial_hidden = self.latent_to_hidden(latent)
        initial_cell = self.latent_to_cell(latent)
        repeated = self.repeat_latent(latent)
        decoded = self.decoder(
            repeated,
            initial_state=[initial_hidden, initial_cell],
            training=training,
        )
        return self.output_layer(decoded)


def _set_seed(seed: int, numpy_seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(numpy_seed)
    tf.keras.utils.set_random_seed(seed)
    # Keep NumPy's preprocessing seed fixed independently of each model seed.
    np.random.seed(numpy_seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


def _resolve_device(device: str) -> tuple[str, str]:
    gpu_available = bool(tf.config.list_physical_devices("GPU"))
    if device == "auto":
        device = "cuda" if gpu_available else "cpu"
    if device == "cuda" and not gpu_available:
        raise RuntimeError(
            "GPU/CUDA was requested but TensorFlow 2.15 cannot see a GPU. "
            "Install a supported TensorFlow GPU environment or use --device cpu."
        )
    return ("/GPU:0", "GPU:0") if device == "cuda" else ("/CPU:0", "CPU:0")


def train_autoencoder(
    x_train: np.ndarray,
    x_validation: np.ndarray,
    *,
    hidden_size: int = 64,
    latent_size: int = 32,
    epochs: int = 20,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 4,
    seed: int = 15,
    numpy_seed: int = 42,
    device: str = "auto",
) -> TrainingResult:
    if len(x_train) == 0 or len(x_validation) == 0:
        raise ValueError("Training and validation sequence arrays must both be non-empty.")
    if epochs < 1 or batch_size < 1 or patience < 1:
        raise ValueError("epochs, batch_size, and patience must all be >= 1.")

    _set_seed(seed, numpy_seed=numpy_seed)
    device_scope, device_label = _resolve_device(device)
    x_train = np.asarray(x_train, dtype=np.float32)
    x_validation = np.asarray(x_validation, dtype=np.float32)
    window_size = int(x_train.shape[1])
    input_size = int(x_train.shape[2])

    with tf.device(device_scope):
        model = LSTMAutoencoder(
            input_size=input_size,
            hidden_size=hidden_size,
            latent_size=latent_size,
            window_size=window_size,
        )
        # Build layer weights before the early-stopping snapshot.
        _ = model(tf.convert_to_tensor(x_train[:1]), training=False)
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
        train_dataset = train_dataset.shuffle(
            buffer_size=len(x_train), seed=seed, reshuffle_each_iteration=True
        ).batch(batch_size)
        val_dataset = tf.data.Dataset.from_tensor_slices(x_validation).batch(batch_size)

        best_loss = float("inf")
        best_weights = [weight.copy() for weight in model.get_weights()]
        epochs_without_improvement = 0
        epochs_run = 0

        for epoch in range(1, epochs + 1):
            for batch in train_dataset:
                with tf.GradientTape() as tape:
                    reconstruction = model(batch, training=True)
                    loss = tf.reduce_mean(tf.square(reconstruction - batch))
                gradients = tape.gradient(loss, model.trainable_variables)
                pairs = [
                    (gradient, variable)
                    for gradient, variable in zip(gradients, model.trainable_variables)
                    if gradient is not None
                ]
                if pairs:
                    clipped, _ = tf.clip_by_global_norm([pair[0] for pair in pairs], 1.0)
                    optimizer.apply_gradients(
                        zip(clipped, [pair[1] for pair in pairs])
                    )

            validation_losses = []
            for batch in val_dataset:
                reconstruction = model(batch, training=False)
                validation_losses.append(float(tf.reduce_mean(
                    tf.square(reconstruction - batch)
                ).numpy()))
            validation_loss = float(np.mean(validation_losses))
            epochs_run = epoch

            if validation_loss < best_loss - 1e-8:
                best_loss = validation_loss
                best_weights = [weight.copy() for weight in model.get_weights()]
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    break

        model.set_weights(best_weights)
        return TrainingResult(
            model=model,
            best_validation_loss=best_loss,
            epochs_run=epochs_run,
            device=device_label,
        )


def reconstruction_errors(
    model: tf.keras.Model,
    sequences: np.ndarray,
    device: str,
    batch_size: int = 512,
) -> np.ndarray:
    sequences = np.asarray(sequences, dtype=np.float32)
    if len(sequences) == 0:
        return np.empty(0, dtype=np.float64)
    device_scope = "/GPU:0" if str(device).upper().startswith("GPU") else "/CPU:0"
    errors: list[float] = []
    with tf.device(device_scope):
        for start in range(0, len(sequences), batch_size):
            batch = tf.convert_to_tensor(sequences[start : start + batch_size])
            reconstruction = model(batch, training=False)
            batch_errors = tf.reduce_mean(tf.square(reconstruction - batch), axis=(1, 2))
            errors.extend(batch_errors.numpy().astype(float).tolist())
    return np.asarray(errors, dtype=np.float64)


def save_checkpoint(model: tf.keras.Model, output_base: Path, metadata: dict) -> str:
    """Save Keras weights and a JSON sidecar with model/scaler metadata."""
    output_base = Path(output_base)
    weights_path = Path(str(output_base) + ".weights.h5")
    metadata_path = Path(str(output_base) + ".json")
    model.save_weights(str(weights_path))
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    return str(weights_path)


def get_runtime_info() -> dict:
    """Return TensorFlow framework and accelerator details."""
    devices = tf.config.list_physical_devices("GPU")
    device_names = []
    for device in devices:
        try:
            details = tf.config.experimental.get_device_details(device)
            device_names.append(details.get("device_name", device.name))
        except Exception:
            device_names.append(device.name)
    return {
        "framework": "TensorFlow",
        "framework_version": tf.__version__,
        "cuda_available": bool(devices),
        "accelerator_name": "; ".join(device_names) if device_names else None,
    }
