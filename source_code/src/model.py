"""Framework selector: TensorFlow is primary; PyTorch is a fallback.

The original reference environment used TensorFlow 2.15.1 on Python 3.10/3.11.
The recorded WSL GPU benchmark used TensorFlow 2.21.0 on Python 3.12.3. When
TensorFlow is installed, this module uses its Keras implementation. The PyTorch fallback exists for compatibility with
other environments and for running the software tests where TensorFlow is not
available; it is not the proposal's reference environment.
"""

try:
    import tensorflow  # noqa: F401
except ImportError:  # pragma: no cover - selected only when TensorFlow is absent
    from .model_torch import (  # noqa: F401
        LSTMAutoencoder,
        TrainingResult,
        get_runtime_info,
        reconstruction_errors,
        save_checkpoint,
        set_seed,
        train_autoencoder,
    )
    ACTIVE_BACKEND = "pytorch_fallback"
else:
    from .model_tf import (  # noqa: F401
        LSTMAutoencoder,
        TrainingResult,
        get_runtime_info,
        reconstruction_errors,
        save_checkpoint,
        train_autoencoder,
    )
    ACTIVE_BACKEND = "tensorflow"
