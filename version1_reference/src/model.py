"""Framework selector: TensorFlow 2.15 is primary; PyTorch is a fallback.

The proposal's reproducibility environment specifies TensorFlow 2.15. When
TensorFlow is installed (as it is through requirements.txt), this module uses
its Keras implementation. The PyTorch fallback exists for compatibility with
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
