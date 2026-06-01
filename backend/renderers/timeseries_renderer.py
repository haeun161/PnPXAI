import os
import numpy as np
import matplotlib.pyplot as plt


def _normalize_attr(attr, signals):
    """Normalize attribution to match signal dimensions."""
    if attr.ndim == 1:
        attr = attr.reshape(1, -1)
    num_channels = signals.shape[0]
    if attr.shape[0] == 1 and num_channels > 1:
        attr = np.tile(attr, (num_channels, 1))
    if attr.shape[-1] != signals.shape[-1]:
        new_attr = np.zeros_like(signals)
        for c in range(min(attr.shape[0], num_channels)):
            new_attr[c] = np.interp(
                np.linspace(0, 1, signals.shape[-1]),
                np.linspace(0, 1, attr.shape[-1]),
                attr[c],
            )
        attr = new_attr
    attr = np.abs(attr)
    attr_max = attr.max()
    if attr_max > 0:
        attr = attr / attr_max
    return attr


def render_timeseries_attribution(
    signals: np.ndarray,
    attribution: np.ndarray,
    output_path: str,
    col_names: list[str] | None = None,
) -> str:
    """Render the most important variable's line chart + attribution overlay."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if signals.ndim == 1:
        signals = signals.reshape(1, -1)
    num_channels = signals.shape[0]
    if col_names is None:
        col_names = [f"var_{i+1}" for i in range(num_channels)]

    attr = _normalize_attr(attribution, signals)

    # Find most important channel
    channel_importance = attr.mean(axis=-1)
    top_idx = int(np.argmax(channel_importance))

    # Single plot: top variable line + attribution overlay
    fig, ax1 = plt.subplots(figsize=(6, 3), dpi=100)
    x = np.arange(signals.shape[-1])
    ax1.plot(x, signals[top_idx], color="steelblue", linewidth=1.5)
    ax1.set_ylabel(col_names[top_idx], color="steelblue", fontsize=10)
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax2 = ax1.twinx()
    ax2.fill_between(x, 0, attr[top_idx], alpha=0.3, color="orangered")
    ax2.set_ylabel("Attribution", color="orangered", fontsize=10)
    ax2.tick_params(axis="y", labelcolor="orangered")
    ax2.set_ylim(0, 1.2)
    ax1.set_xlabel("Time Step", fontsize=10)
    title = col_names[top_idx] if num_channels > 1 else "Time-Series Attribution"
    ax1.set_title(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    # If multi-variate, also render expanded view with all variables
    if num_channels > 1:
        expanded_path = output_path.replace(".png", "_expanded.png")
        _render_expanded(signals, attr, col_names, expanded_path)

    return output_path


def _render_expanded(
    signals: np.ndarray,
    attr: np.ndarray,
    col_names: list[str],
    output_path: str,
):
    """Render all variables as subplots with attribution overlay."""
    num_channels = signals.shape[0]
    fig, axes = plt.subplots(num_channels, 1, figsize=(7, 2 * num_channels), dpi=100, sharex=True)
    if num_channels == 1:
        axes = [axes]

    x = np.arange(signals.shape[-1])
    for i, ax in enumerate(axes):
        ax.plot(x, signals[i], color="steelblue", linewidth=1.2)
        ax.set_ylabel(col_names[i], fontsize=8, color="steelblue")
        ax.tick_params(axis="y", labelcolor="steelblue", labelsize=7)

        ax2 = ax.twinx()
        ax2.fill_between(x, 0, attr[i], alpha=0.25, color="orangered")
        ax2.set_ylim(0, 1.2)
        ax2.tick_params(axis="y", labelcolor="orangered", labelsize=7)
        if i == 0:
            ax.set_title("All Variables — Attribution", fontsize=10, fontweight="bold")

    axes[-1].set_xlabel("Time Step", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
