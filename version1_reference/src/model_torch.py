from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainingResult:
    model: nn.Module
    best_validation_loss: float
    epochs_run: int
    device: str


class LSTMAutoencoder(nn.Module):
    """Univariate sequence-to-sequence LSTM autoencoder."""

    def __init__(self, input_size: int = 1, hidden_size: int = 64, latent_size: int = 32):
        super().__init__()
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.to_latent = nn.Linear(hidden_size, latent_size)
        self.latent_to_hidden = nn.Linear(latent_size, hidden_size)
        self.latent_to_cell = nn.Linear(latent_size, hidden_size)
        self.decoder = nn.LSTM(input_size=latent_size, hidden_size=hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.encoder(x)
        latent = torch.tanh(self.to_latent(hidden[-1]))
        repeated = latent.unsqueeze(1).expand(-1, x.size(1), -1)
        h0 = torch.tanh(self.latent_to_hidden(latent)).unsqueeze(0)
        c0 = torch.tanh(self.latent_to_cell(latent)).unsqueeze(0)
        decoded, _ = self.decoder(repeated, (h0, c0))
        return self.output_layer(decoded)


def set_seed(seed: int, numpy_seed: int = 42) -> None:
    """Seed Python/PyTorch per run while keeping NumPy preprocessing fixed."""
    random.seed(seed)
    np.random.seed(numpy_seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Prefer repeatable kernels over maximum throughput.
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # Some accelerator operations may not have deterministic implementations;
    # warn rather than abort so the project can still run on supported devices.
    torch.use_deterministic_algorithms(True, warn_only=True)


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

    set_seed(seed, numpy_seed=numpy_seed)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this PyTorch installation.")

    model = LSTMAutoencoder(
        input_size=x_train.shape[-1], hidden_size=hidden_size, latent_size=latent_size
    ).to(device)
    train_tensor = torch.from_numpy(np.asarray(x_train, dtype=np.float32))
    validation_tensor = torch.from_numpy(np.asarray(x_validation, dtype=np.float32))
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=loader_generator,
    )
    validation_loader = DataLoader(
        TensorDataset(validation_tensor), batch_size=batch_size, shuffle=False, drop_last=False
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_function = nn.MSELoss()
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    epochs_run = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(batch)
            loss = loss_function(reconstruction, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        validation_losses = []
        with torch.no_grad():
            for (batch,) in validation_loader:
                batch = batch.to(device)
                reconstruction = model(batch)
                validation_losses.append(loss_function(reconstruction, batch).item())
        validation_loss = float(np.mean(validation_losses))
        epochs_run = epoch

        if validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return TrainingResult(
        model=model,
        best_validation_loss=best_loss,
        epochs_run=epochs_run,
        device=device,
    )


def reconstruction_errors(
    model: nn.Module, sequences: np.ndarray, device: str, batch_size: int = 512
) -> np.ndarray:
    if len(sequences) == 0:
        return np.empty(0, dtype=np.float64)
    tensor = torch.from_numpy(np.asarray(sequences, dtype=np.float32))
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=False)
    result = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            reconstruction = model(batch)
            per_sequence_mse = torch.mean((reconstruction - batch) ** 2, dim=(1, 2))
            result.extend(per_sequence_mse.detach().cpu().numpy().astype(float).tolist())
    return np.asarray(result, dtype=np.float64)


def save_checkpoint(model, output_base, metadata: dict) -> str:
    """Save the optional PyTorch fallback checkpoint and metadata."""
    from pathlib import Path
    import json

    output_base = Path(output_base)
    checkpoint_path = output_base.with_suffix(".pt")
    torch.save({"state_dict": model.state_dict(), **metadata}, checkpoint_path)
    metadata_path = output_base.with_suffix(".json")
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    return str(checkpoint_path)


def get_runtime_info() -> dict:
    """Return framework and accelerator details for reproducibility logs."""
    return {
        "framework": "PyTorch fallback",
        "framework_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "accelerator_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
