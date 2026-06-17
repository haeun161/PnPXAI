import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase

# Maximum number of variables to visualize in expanded view.
# Beyond this, users can download the full attribution as Excel.
MAX_VIZ_VARIABLES = 15  # K — change this to adjust the limit


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


_ATTR_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "cyan_magenta", ["#00FFFF", "#FF00FF"]
)


def _plot_single(ax, signal, attr, col_name, x, rank=None, show_xlabel=True):
    """Plot one variable with attribution as background color strips (cyan→magenta)."""
    for i in range(len(x)):
        color = _ATTR_CMAP(attr[i])
        ax.axvspan(x[i] - 0.5, x[i] + 0.5, color=color, alpha=0.3 + 0.7 * attr[i], linewidth=0)

    ax.plot(x, signal, color="black", linewidth=1.3, zorder=5)
    label = f"#{rank}  {col_name}" if rank is not None else col_name
    ax.set_ylabel(label, fontsize=11, fontweight="bold", color="black")
    ax.tick_params(axis="y", labelsize=8, colors="black")
    ax.tick_params(axis="x", labelsize=8, colors="black")
    ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)

    # Always show timestep ticks
    n_ticks = min(10, len(x))
    tick_positions = np.linspace(0, len(x) - 1, n_ticks, dtype=int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(int(t)) for t in tick_positions], fontsize=7, color="black")

    if show_xlabel:
        ax.set_xlabel("Time Step", fontsize=9, color="black")


def _add_colorbar(fig, bottom_margin=0.06):
    """Add a horizontal colorbar well below the plots."""
    cbar_ax = fig.add_axes([0.2, bottom_margin - 0.04, 0.6, 0.012])
    norm = mcolors.Normalize(vmin=0, vmax=1)
    cb = ColorbarBase(cbar_ax, cmap=_ATTR_CMAP, norm=norm, orientation="horizontal")
    cb.set_label("Attribution  (low → high)", fontsize=8, color="black", labelpad=4)
    cb.ax.tick_params(labelsize=7, colors="black")


def render_timeseries_attribution(
    signals: np.ndarray,
    attribution: np.ndarray,
    output_path: str,
    col_names: list[str] | None = None,
) -> str:
    """Render time-series attribution with background color strips (cyan→magenta).

    - Single variate: one plot.
    - Multi-variate: top 3 in main view, top K in expanded (5×3 grid), Excel for all if > K.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if signals.ndim == 1:
        signals = signals.reshape(1, -1)
    num_channels = signals.shape[0]
    if col_names is None:
        col_names = [f"var_{i+1}" for i in range(num_channels)]

    attr = _normalize_attr(attribution, signals)
    channel_importance = attr.mean(axis=-1)
    sorted_idx = np.argsort(channel_importance)[::-1]
    x = np.arange(signals.shape[-1])

    if num_channels == 1:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
        _plot_single(ax, signals[0], attr[0], col_names[0], x)
        ax.set_title("Time-Series Attribution", fontsize=12, fontweight="bold", color="black")
        fig.subplots_adjust(bottom=0.15)
        _add_colorbar(fig, bottom_margin=0.10)
    else:
        show_n = min(num_channels, 3)
        fig, axes = plt.subplots(show_n, 1, figsize=(6, 2.5 * show_n + 1), dpi=100, sharex=True)
        if show_n == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            ch = sorted_idx[i]
            is_last = (i == show_n - 1)
            _plot_single(ax, signals[ch], attr[ch], col_names[ch], x, rank=i + 1, show_xlabel=is_last)
            if i == 0:
                extra = f" (top {show_n} of {num_channels})" if num_channels > 3 else ""
                ax.set_title(f"Attribution by Variable{extra}", fontsize=12, fontweight="bold", color="black")

        fig.subplots_adjust(bottom=0.10)
        _add_colorbar(fig, bottom_margin=0.07)

        # Expanded view
        expanded_path = output_path.replace(".png", "_expanded.png")
        _render_expanded(signals, attr, col_names, sorted_idx, expanded_path)

        # ZIP bundle: individual variable PNGs + Excel data
        zip_path = output_path.replace(".png", "_bundle.zip")
        _create_bundle_zip(signals, attr, col_names, sorted_idx, channel_importance, x, zip_path)

    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return output_path


def _render_expanded(signals, attr, col_names, sorted_idx, output_path):
    """Render top K variables in a 5-row × 3-col grid, sorted by importance."""
    num_channels = signals.shape[0]
    show_n = min(num_channels, MAX_VIZ_VARIABLES)
    ncols = 3
    nrows = int(np.ceil(show_n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 2.5 * nrows + 1), dpi=100)
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    x = np.arange(signals.shape[-1])
    for i in range(show_n):
        ax = axes_flat[i]
        ch = sorted_idx[i]
        is_bottom = (i >= (nrows - 1) * ncols)
        _plot_single(ax, signals[ch], attr[ch], col_names[ch], x, rank=i + 1, show_xlabel=is_bottom)
        if i == 0:
            extra = f" (top {show_n} of {num_channels})" if num_channels > show_n else ""
            ax.set_title(f"All Variables — Ranked by Importance{extra}", fontsize=11, fontweight="bold", color="black")

    for i in range(show_n, len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.subplots_adjust(bottom=0.07, hspace=0.45, wspace=0.35)
    _add_colorbar(fig, bottom_margin=0.04)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def _create_bundle_zip(signals, attr, col_names, sorted_idx, channel_importance, x, zip_path):
    """Create a ZIP bundle with individual variable PNGs + Excel data."""
    import zipfile
    import tempfile
    import pandas as pd

    with tempfile.TemporaryDirectory() as tmp_dir:
        img_dir = os.path.join(tmp_dir, "images")
        os.makedirs(img_dir)

        # Render individual variable PNGs (one per variable, ranked)
        for rank_i, ch in enumerate(sorted_idx):
            rank = rank_i + 1
            safe_name = col_names[ch].replace("/", "_").replace("\\", "_").replace(" ", "_")
            fname = f"#Rank{rank}_{safe_name}.png"

            fig, ax = plt.subplots(figsize=(8, 4.5), dpi=100)
            _plot_single(ax, signals[ch], attr[ch], col_names[ch], x, rank=rank)
            ax.set_title(f"#{rank} {col_names[ch]} — Attribution", fontsize=12, fontweight="bold", color="black", pad=10)
            fig.subplots_adjust(bottom=0.22, top=0.88, left=0.12, right=0.92)
            _add_colorbar(fig, bottom_margin=0.08)
            fig.savefig(os.path.join(img_dir, fname), bbox_inches="tight", pad_inches=0.15)
            plt.close(fig)

        # Excel 1: Variable Ranking (mean attribution per variable)
        seq_len = signals.shape[-1]
        ranking_path = os.path.join(tmp_dir, "variable_ranking.xlsx")
        ranking_data = [
            {"Rank": i + 1, "Variable": col_names[ch], "Mean Attribution": float(channel_importance[ch])}
            for i, ch in enumerate(sorted_idx)
        ]
        pd.DataFrame(ranking_data).to_excel(ranking_path, index=False, engine="openpyxl")

        # Excel 2: Original data + per-timestep per-variable attribution side by side
        data_attr_path = os.path.join(tmp_dir, "data_attribution.xlsx")
        data_attr = {"timestep": list(range(seq_len))}
        for ch in range(signals.shape[0]):
            name = col_names[ch]
            data_attr[name] = signals[ch].tolist()
            data_attr[f"{name}_attribution"] = attr[ch].tolist()
        pd.DataFrame(data_attr).to_excel(data_attr_path, index=False, engine="openpyxl")

        # Create ZIP
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(ranking_path, "variable_ranking.xlsx")
            zf.write(data_attr_path, "data_attribution.xlsx")
            for fname in sorted(os.listdir(img_dir)):
                zf.write(os.path.join(img_dir, fname), f"images/{fname}")
