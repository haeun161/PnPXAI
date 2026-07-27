import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase

# Maximum number of variables to visualize in expanded view.
# Beyond this, users can download the full attribution as Excel.
MAX_VIZ_VARIABLES = 15  # K — change this to adjust the limit


def _normalize_attr(attr, signals):
    """Reshape attribution to the signal grid and scale it so ±1 is a strong contribution.

    The divisor is the 99th percentile of |attr| across *all* channels and timesteps —
    global, so a value keeps its sign (negative = pushed the prediction down) and stays
    comparable between channels rather than being stretched to fill its own range.

    Percentile rather than max: one cell (usually the most recent timestep) routinely
    carries 30-60x the average attribution, and dividing by that flattens everything else
    to near zero — measured at 92% of cells below 0.1. Scaling by p99 lifts the bulk into
    a visible range and lets the top 1% saturate at the ends of the colour ramp, which is
    where the strongest contributions belong anyway.
    """
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
    magnitude = np.abs(attr)
    scale = np.percentile(magnitude, 99)
    if scale <= 0:
        # Attribution so sparse that under 1% of cells are non-zero, making p99 zero.
        scale = magnitude.max()
    if scale > 0:
        attr = attr / scale
    return attr


# Diverging, centred on white: -1 blue (pushed the prediction down), 0 white (no
# influence), +1 red (pushed it up). Hue carries the sign and lightness the magnitude,
# so the ramp alone encodes both — no opacity trick needed on top of it.
_ATTR_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "blue_white_red", ["#0000FF", "#FFFFFF", "#FF0000"]
)
# clip=True so the top 1% that _normalize_attr leaves beyond ±1 saturates at the ends of
# the ramp instead of being mapped past it.
_ATTR_NORM = mcolors.Normalize(vmin=-1, vmax=1, clip=True)


def _plot_single(ax, signal, attr, col_name, x, rank=None, show_xlabel=True, time_labels=None):
    """Plot one variable over a background shaded by its attribution."""
    # Long series: one strip per block of samples instead of per sample, or the number of
    # patches makes rendering crawl.
    if len(x) > 1000:
        idx = np.arange(0, len(x), max(1, len(x) // 500))
        if time_labels is not None and len(time_labels) == len(x):
            time_labels = [time_labels[i] for i in idx]
        # Average within each block so a strip stands for everything it covers.
        attr = np.array([b.mean() for b in np.split(attr, idx[1:])])
        x, signal = x[idx], signal[idx]

    half = (x[1] - x[0]) / 2 if len(x) > 1 else 0.5
    for i in range(len(x)):
        ax.axvspan(x[i] - half, x[i] + half,
                   color=_ATTR_CMAP(_ATTR_NORM(attr[i])), linewidth=0)
    ax.plot(x, signal, color="black", linewidth=1.3, zorder=5)
    ax.set_xlim(x[0] - half, x[-1] + half)

    label = f"#{rank}  {col_name}" if rank is not None else col_name
    # The label runs vertically, so a long sensor name is taller than the ~1.7in row it
    # belongs to and runs into the neighbouring plots. Clip it rather than let rows
    # overlap; the untruncated names are in the downloadable bundle.
    if len(label) > 18:
        label = label[:17] + "…"
    ax.set_ylabel(label, fontsize=8, fontweight="bold", color="black")
    ax.tick_params(axis="y", labelsize=8, colors="black")
    ax.tick_params(axis="x", labelsize=8, colors="black")

    # Ticks are placed in data coordinates, which are the original timestep numbers —
    # they no longer match positions in the array once the series has been thinned.
    n_ticks = min(10, len(x))
    tick_idx = np.linspace(0, len(x) - 1, n_ticks, dtype=int)
    ax.set_xticks([x[i] for i in tick_idx])
    if time_labels is not None and len(time_labels) == len(x):
        ax.set_xticklabels([time_labels[i] for i in tick_idx], fontsize=6, color="black", rotation=30, ha="right")
    else:
        ax.set_xticklabels([str(int(x[i])) for i in tick_idx], fontsize=7, color="black")

    if show_xlabel:
        ax.set_xlabel("Time" if time_labels else "Time Steps", fontsize=9, color="black")


def _add_colorbar(fig, bottom_margin=0.06):
    """Add a horizontal colorbar well below the plots."""
    cbar_ax = fig.add_axes([0.2, bottom_margin - 0.04, 0.6, 0.012])
    cb = ColorbarBase(cbar_ax, cmap=_ATTR_CMAP, norm=_ATTR_NORM, orientation="horizontal")
    cb.set_label("Attribution  (− pushes down / + pushes up)", fontsize=8, color="black", labelpad=4)
    cb.ax.tick_params(labelsize=7, colors="black")


def render_timeseries_attribution(
    signals: np.ndarray,
    attribution: np.ndarray,
    output_path: str,
    col_names: list[str] | None = None,
    time_labels: list[str] | None = None,
) -> str:
    """Render time-series attribution as background colour strips (cyan→magenta).

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
    # Rank on magnitude: attribution is signed, so a plain mean would let a channel's
    # positive and negative contributions cancel into "unimportant".
    channel_importance = np.abs(attr).mean(axis=-1)
    sorted_idx = np.argsort(channel_importance)[::-1]
    x = np.arange(signals.shape[-1])

    # The result cards are landscape, and object-contain scales by whichever dimension
    # binds first — so a portrait figure gets shrunk to fit the height and leaves most of
    # the card's width empty. Keeping the figure wider than any card makes width the
    # binding dimension instead, so it fills the card without cropping.
    seq_len = signals.shape[-1]
    # Requested rather than final: savefig(bbox_inches="tight") trims side margins, so the
    # saved image lands near 2.6 — comfortably past any card, with room for wider windows.
    CARD_ASPECT = 3.0

    def _fig_width(fig_height: float) -> float:
        return max(CARD_ASPECT * fig_height, min(20, seq_len / 100))

    # Rotated timestamps are far taller than bare indices, and they sit in the gap between
    # the plots and the colorbar — without extra room the two overlap.
    has_time_labels = time_labels is not None and len(time_labels) == seq_len

    if num_channels == 1:
        fig_height = 3.2
        fig, ax = plt.subplots(figsize=(_fig_width(fig_height), fig_height), dpi=100)
        _plot_single(ax, signals[0], attr[0], col_names[0], x, time_labels=time_labels)
        ax.set_title("Time-Series Attribution", fontsize=12, fontweight="bold", color="black")
        bottom = 0.34 if has_time_labels else 0.18
        fig.subplots_adjust(bottom=bottom)
        _add_colorbar(fig, bottom_margin=bottom * 0.55)
    else:
        show_n = min(num_channels, 3)
        fig_height = 1.7 * show_n + 0.9
        fig, axes = plt.subplots(show_n, 1, figsize=(_fig_width(fig_height), fig_height), dpi=100, sharex=True)
        if show_n == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            ch = sorted_idx[i]
            is_last = (i == show_n - 1)
            _plot_single(ax, signals[ch], attr[ch], col_names[ch], x, rank=i + 1, show_xlabel=is_last, time_labels=time_labels)
            if i == 0:
                extra = f" (top {show_n} of {num_channels})" if num_channels > 3 else ""
                ax.set_title(f"Attribution by Variable{extra}", fontsize=12, fontweight="bold", color="black")

        bottom = 0.24 if has_time_labels else 0.10
        fig.subplots_adjust(bottom=bottom)
        _add_colorbar(fig, bottom_margin=bottom * 0.5)

        # Expanded view
        expanded_path = output_path.replace(".png", "_expanded.png")
        _render_expanded(signals, attr, col_names, sorted_idx, expanded_path, time_labels=time_labels)

        # ZIP bundle: individual variable PNGs + Excel data
        zip_path = output_path.replace(".png", "_bundle.zip")
        _create_bundle_zip(signals, attr, col_names, sorted_idx, channel_importance, x, zip_path, time_labels=time_labels)

    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return output_path


def _render_expanded(signals, attr, col_names, sorted_idx, output_path, time_labels=None):
    """Render top K variables in a 5-row × 3-col grid, sorted by importance."""
    num_channels = signals.shape[0]
    show_n = min(num_channels, MAX_VIZ_VARIABLES)
    # One variable per row: every channel gets the full width, and they share a time axis
    # so the same instant lines up vertically across all of them.
    ROW_HEIGHT = 1.7
    fig_height = ROW_HEIGHT * show_n + 0.9
    fig, axes = plt.subplots(show_n, 1, figsize=(14, fig_height), dpi=100, sharex=True)
    axes_flat = np.atleast_1d(axes)

    x = np.arange(signals.shape[-1])
    for i in range(show_n):
        ax = axes_flat[i]
        ch = sorted_idx[i]
        _plot_single(ax, signals[ch], attr[ch], col_names[ch], x, rank=i + 1,
                     show_xlabel=(i == show_n - 1), time_labels=time_labels)
        if i == 0:
            extra = f" (top {show_n} of {num_channels})" if num_channels > show_n else ""
            ax.set_title(f"All Variables — Ranked by Importance{extra}", fontsize=11, fontweight="bold", color="black")

    # The figure grows with the channel count, so reserve the space below the last plot in
    # inches — a fixed fraction would balloon into a huge gap once there are many rows.
    has_time_labels = time_labels is not None and len(time_labels) == signals.shape[-1]
    bottom = (1.5 if has_time_labels else 0.7) / fig_height
    fig.subplots_adjust(bottom=bottom, hspace=0.3)
    _add_colorbar(fig, bottom_margin=bottom * 0.5)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def _create_bundle_zip(signals, attr, col_names, sorted_idx, channel_importance, x, zip_path, time_labels=None):
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
            _plot_single(ax, signals[ch], attr[ch], col_names[ch], x, rank=rank, time_labels=time_labels)
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

        # Data + attribution: use CSV for large data (>10K rows), Excel otherwise
        data_attr = {"timestep": list(range(seq_len))}
        if time_labels and len(time_labels) == seq_len:
            data_attr["time"] = time_labels
        for ch in range(signals.shape[0]):
            name = col_names[ch]
            data_attr[name] = signals[ch].tolist()
            data_attr[f"{name}_attribution"] = attr[ch].tolist()
        df_attr = pd.DataFrame(data_attr)

        if seq_len > 10000:
            data_attr_path = os.path.join(tmp_dir, "data_attribution.csv")
            data_attr_zip_name = "data_attribution.csv"
            df_attr.to_csv(data_attr_path, index=False)
        else:
            data_attr_path = os.path.join(tmp_dir, "data_attribution.xlsx")
            data_attr_zip_name = "data_attribution.xlsx"
            df_attr.to_excel(data_attr_path, index=False, engine="openpyxl")

        # Create ZIP
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(ranking_path, "variable_ranking.xlsx")
            zf.write(data_attr_path, data_attr_zip_name)
            for fname in sorted(os.listdir(img_dir)):
                zf.write(os.path.join(img_dir, fname), f"images/{fname}")
