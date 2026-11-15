from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def exploratory_summary(frame: pd.DataFrame, dataset: str, domain: str) -> dict:
    values = frame["value"].astype(float)
    median = float(values.median())
    mad = float(np.median(np.abs(values.to_numpy() - median)))
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    q25 = float(values.quantile(0.25))
    q75 = float(values.quantile(0.75))
    return {
        "dataset": dataset,
        "domain": domain,
        "observations": int(values.size),
        "start_timestamp": frame["timestamp"].iloc[0],
        "end_timestamp": frame["timestamp"].iloc[-1],
        "mean": mean,
        "std_population": std,
        "median": median,
        "mad_raw": mad,
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "q25": q25,
        "q75": q75,
        "iqr": q75 - q25,
        "coefficient_of_variation": float(std / abs(mean)) if mean != 0 else np.nan,
        "skewness": float(values.skew()) if values.size > 2 else np.nan,
        "kurtosis": float(values.kurt()) if values.size > 3 else np.nan,
    }


def write_rolling_eda(
    frame: pd.DataFrame,
    dataset: str,
    domain: str,
    output_dir: Path,
    rolling_window: int = 50,
) -> dict:
    """Write rolling mean/std/median/MAD tables and a diagnostic plot."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = output_dir / "rolling_csv"
    plot_dir = output_dir / "plots"
    csv_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    values = frame["value"].astype(float)
    rolling = values.rolling(window=rolling_window, min_periods=2)
    roll_median = rolling.median()
    roll_mad = values.rolling(window=rolling_window, min_periods=2).apply(
        lambda x: float(np.median(np.abs(x - np.median(x)))), raw=True
    )
    table = pd.DataFrame({
        "timestamp": frame["timestamp"],
        "value": values,
        "rolling_mean": rolling.mean(),
        "rolling_std": rolling.std(ddof=0),
        "rolling_median": roll_median,
        "rolling_mad_raw": roll_mad,
    })
    stem = Path(dataset).stem
    table.to_csv(csv_dir / f"{stem}_rolling_stats.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(frame["timestamp"], values, linewidth=0.8, label="Observed value")
    axes[0].plot(frame["timestamp"], table["rolling_mean"], linewidth=1.0, label=f"Rolling mean (W={rolling_window})")
    axes[0].set_ylabel("Value")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(alpha=0.2)
    axes[1].plot(frame["timestamp"], table["rolling_std"], linewidth=0.9, label="Rolling standard deviation")
    axes[1].plot(frame["timestamp"], table["rolling_mad_raw"], linewidth=0.9, label="Rolling raw MAD")
    axes[1].set_ylabel("Variability")
    axes[1].set_xlabel("Timestamp")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(alpha=0.2)
    fig.suptitle(f"Exploratory rolling statistics: {stem} ({domain})")
    fig.tight_layout()
    fig.savefig(plot_dir / f"{stem}_rolling_eda.png", dpi=160)
    plt.close(fig)

    summary = exploratory_summary(frame, dataset, domain)
    summary["rolling_window_for_eda"] = rolling_window
    return summary
