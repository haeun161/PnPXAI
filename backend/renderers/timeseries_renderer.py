import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase

# Maximum number of variables to visualize in expanded view.
# Beyond this, users can download the full attribution as Excel.
MAX_VIZ_VARIABLES = 15  # K — change this to adjust the limit

def _normalize_attr(attr, signals):
    """Reshape attribution to the signal grid, take its magnitude, and scale so 1 is strong.

    Only the magnitude is shown: the sign of an attribution says which way a cell pushed
    the prediction, which is not what a reader of these plots is asking — they want to
    know *where* the model looked, and a strongly negative cell mattered just as much as
    a strongly positive one. Interpolation to the signal grid happens on the signed values
    first, so neighbouring cells of opposite sign do not inflate each other.

    The divisor is the 99th percentile across *all* channels and timesteps — global, so a
    value stays comparable between channels rather than being stretched to fill its own
    range.

    Percentile rather than max: one cell (usually the most recent timestep) routinely
    carries 30-60x the average attribution, and dividing by that flattens everything else
    to near zero — measured at 92% of cells below 0.1. Scaling by p99 lifts the bulk into
    a visible range and lets the top 1% saturate at the end of the colour ramp, which is
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
    attr = np.abs(attr)
    scale = np.percentile(attr, 99)
    if scale <= 0:
        # Attribution so sparse that under 1% of cells are non-zero, making p99 zero.
        scale = attr.max()
    if scale > 0:
        attr = attr / scale
    return attr


# Sequential, since _normalize_attr hands over magnitudes: 0 white (no influence) through
# 1 red (strong contribution, either direction). One ramp end, not two — with the sign
# dropped there is nothing for a second hue to carry.
_ATTR_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "white_red", ["#FFFFFF", "#FF0000"]
)
# clip=True so the top 1% that _normalize_attr leaves beyond 1 saturates at the end of the
# ramp instead of being mapped past it.
_ATTR_NORM = mcolors.Normalize(vmin=0, vmax=1, clip=True)

def _plot_single(ax, signal, attr, col_name, x, rank=None, show_xlabel=True, time_labels=None,
                 pred_value=None, pred_label=None, pred_region=False):
    """Plot one variable over a background shaded by its attribution.

    `pred_region` draws the solid divider marking where the input window ends, and is set
    on every row so they share one x-axis. `pred_value` additionally plots the forecast
    itself, and belongs *only* to the attributed channel: the explainers attribute a
    single scalar, so every row's colours describe that same one value. A row carrying
    its own next step would read as though its colours explained it.

    The predicted point carries no attribution strip -- it is the output being explained,
    not part of the input the explanation is over -- so the blank cell past the divider is
    meaningful rather than missing data.
    """
    # One step past the last input sample. Taken before the thinning below, which drops
    # elements and would otherwise leave this landing short of the window's real end.
    pred_x = float(x[-1]) + 1.0

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

    if pred_region:
        # Solid, unlike the dashed guides elsewhere: this is a hard boundary between what
        # the model was shown and what it produced, not a subdivision of either.
        ax.axvline(pred_x - 0.5, color="black", linewidth=1.4, zorder=7)
        # One step out of ~96 is a sliver, and against the frame the marker reads as
        # clipped rather than plotted. Padding the right margin gives it somewhere to sit
        # without moving it -- the point stays at its true +1 position, so the axis is
        # still linear in time. Scaled to the window, or the same absolute pad that suits
        # a 96-step window becomes a sixth of a 12-step one.
        ax.set_xlim(x[0] - half, pred_x + max(0.6, 0.02 * (pred_x - float(x[0]))))

    if pred_value is not None:
        ax.plot([x[-1], pred_x], [signal[-1], pred_value], color="#FF0000", linewidth=1.6,
                zorder=6)
        ax.plot([pred_x], [pred_value], marker="o", markersize=5, color="#FF0000",
                markeredgecolor="white", markeredgewidth=0.7, zorder=8)

    label = f"#{rank}  {col_name}" if rank is not None else col_name
    # The label runs vertically, so a long sensor name is taller than the ~1.7in row it
    # belongs to and runs into the neighbouring plots. Clip it rather than let rows
    # overlap; the untruncated names are in the downloadable bundle. The cutoff scales
    # with the label's own font size -- a bigger font needs fewer characters to fill
    # the same 1.7in.
    max_label_chars = 12
    if len(label) > max_label_chars:
        label = label[:max_label_chars - 1] + "…"
    ax.set_ylabel(label, fontsize=13, color="black")
    ax.tick_params(axis="y", labelsize=11, colors="black")
    ax.tick_params(axis="x", labelsize=11, colors="black")

    # Ticks are placed in data coordinates, which are the original timestep numbers —
    # they no longer match positions in the array once the series has been thinned.
    #
    # A short window gets a tick on every timestep. One strip is drawn per timestep, so
    # labelling only a subset leaves strips the reader can't name — and because the
    # subset is picked by rounding evenly-spaced floats, the survivors end up at uneven
    # spacing (a 12-step window kept steps 0-4, 6-9, 11), which reads as a plotting bug.
    n_ticks = len(x) if len(x) <= 24 else 10
    tick_idx = list(np.linspace(0, len(x) - 1, n_ticks, dtype=int))
    ticks = [x[i] for i in tick_idx]
    if time_labels is not None and len(time_labels) == len(x):
        # Few enough ticks to have room: draw them large enough to actually read.
        size = 10 if len(tick_idx) <= 16 else 8
        tick_labels = [time_labels[i] for i in tick_idx]
        rotation, ha = 30, "right"
    else:
        size = 9
        tick_labels = [str(int(x[i])) for i in tick_idx]
        rotation, ha = 0, "center"

    if pred_region:
        # The predicted step gets its own tick -- without one the point past the divider
        # has no position on the axis, and its timestamp is the one a reader most wants.
        # It sits one step from the end of the window, so the regular tick nearest the
        # end would print on top of it; drop whatever falls inside that gap first.
        min_gap = 0.06 * (pred_x - float(x[0]))
        while ticks and pred_x - float(ticks[-1]) < min_gap:
            ticks.pop()
            tick_labels.pop()
        ticks.append(pred_x)
        tick_labels.append(pred_label if pred_label else str(int(pred_x)))

    ax.set_xticks(ticks)
    # Use the Text objects set_xticklabels hands back rather than re-reading them off the
    # axis: on the shared-x subplots the getter comes back empty.
    drawn = ax.set_xticklabels(tick_labels, fontsize=size, color="black", rotation=rotation, ha=ha)
    if pred_region and drawn:
        # Matched to the point it names, so it reads as the forecast rather than as one
        # more observation.
        drawn[-1].set_color("#FF0000")

    if show_xlabel:
        ax.set_xlabel("Time" if time_labels else "Time Steps", fontsize=11, color="black")


def _add_colorbar(fig, bottom_margin=0.06, target_name=None):
    """Add a horizontal colorbar well below the plots.

    `target_name` names what was attributed. Every row's colours are gradients of one
    scalar -- the forecast for that single channel -- so without it a reader has no way
    to tell that a row's colours are not about that row's own variable.
    """
    cbar_ax = fig.add_axes([0.2, bottom_margin - 0.04, 0.6, 0.012])
    cb = ColorbarBase(cbar_ax, cmap=_ATTR_CMAP, norm=_ATTR_NORM, orientation="horizontal")
    what = f"predicted {target_name}" if target_name else "the prediction"
    cb.set_label(f"Attribution to {what}  (magnitude: white = none / red = strong)",
                 fontsize=15, color="black", labelpad=4)
    cb.ax.tick_params(labelsize=9, colors="black")


def render_timeseries_attribution(
    signals: np.ndarray,
    attribution: np.ndarray,
    output_path: str,
    col_names: list[str] | None = None,
    time_labels: list[str] | None = None,
    display_name: str | None = None,
    next_pred: np.ndarray | None = None,
    next_pred_label: str | None = None,
    attributed_channel: int | None = None,
) -> str:
    """Render time-series attribution magnitude as background colour strips (white→red).

    - Single variate: one plot.
    - Multi-variate: top 3 in main view, top K in expanded (5×3 grid), Excel for all if > K.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if signals.ndim == 1:
        signals = signals.reshape(1, -1)
    num_channels = signals.shape[0]
    if col_names is None:
        col_names = [f"var_{i+1}" for i in range(num_channels)]

    # Already magnitudes, so the mean cannot cancel a channel's contributions into
    # "unimportant" the way a mean of signed values would.
    attr = _normalize_attr(attribution, signals)
    channel_importance = attr.mean(axis=-1)
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

    title = display_name or "Attribution"

    # The forecaster predicts every channel, but only one of them was attributed, and
    # only that one's forecast may be drawn -- see `_plot_single`. Every row still gets
    # the divider so they share an x-axis and the input window's end is unambiguous.
    has_pred = next_pred is not None
    target_ch = attributed_channel if attributed_channel is not None else num_channels - 1
    target_name = col_names[target_ch] if has_pred and 0 <= target_ch < len(col_names) else None

    def pred_for(c: int):
        if not has_pred or c != target_ch or c >= len(next_pred):
            return None
        return float(next_pred[c])

    if num_channels == 1:
        fig_height = 3.2
        fig, ax = plt.subplots(figsize=(_fig_width(fig_height), fig_height), dpi=100)
        _plot_single(ax, signals[0], attr[0], col_names[0], x, time_labels=time_labels,
                     pred_value=pred_for(0), pred_label=next_pred_label, pred_region=has_pred)
        ax.set_title(title, fontsize=20, color="black")
        bottom = (1.5 if has_time_labels else 0.7) / fig_height
        fig.subplots_adjust(bottom=bottom)
        _add_colorbar(fig, bottom_margin=0.34 / fig_height, target_name=target_name)
    else:
        show_n = min(num_channels, 3)
        fig_height = 1.7 * show_n + 0.9
        fig, axes = plt.subplots(show_n, 1, figsize=(_fig_width(fig_height), fig_height), dpi=100, sharex=True)
        if show_n == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            ch = sorted_idx[i]
            is_last = (i == show_n - 1)
            _plot_single(ax, signals[ch], attr[ch], col_names[ch], x, rank=i + 1,
                         show_xlabel=is_last, time_labels=time_labels,
                         pred_value=pred_for(ch), pred_label=next_pred_label,
                         pred_region=has_pred)
            if i == 0:
                ax.set_title(title, fontsize=20, color="black")

        # Reserved in inches rather than as a fraction: the strip below the plots holds
        # rotated timestamps plus the axis title, whose height is set by the font, not by
        # how tall the figure happens to be. As a fraction they collided with the colorbar
        # as soon as the labels were full timestamps.
        bottom = (1.7 if has_time_labels else 0.8) / fig_height
        fig.subplots_adjust(bottom=bottom)
        _add_colorbar(fig, bottom_margin=0.34 / fig_height, target_name=target_name)

        # Expanded view
        expanded_path = output_path.replace(".png", "_expanded.png")
        _render_expanded(signals, attr, col_names, sorted_idx, expanded_path,
                         time_labels=time_labels, display_name=display_name,
                         next_pred=next_pred, next_pred_label=next_pred_label,
                         target_ch=target_ch, target_name=target_name)

        # ZIP bundle: individual variable PNGs + Excel data
        zip_path = output_path.replace(".png", "_bundle.zip")
        _create_bundle_zip(signals, attr, col_names, sorted_idx, channel_importance, x, zip_path,
                           time_labels=time_labels, next_pred=next_pred,
                           next_pred_label=next_pred_label, target_ch=target_ch,
                           target_name=target_name)

    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return output_path


def _render_expanded(signals, attr, col_names, sorted_idx, output_path, time_labels=None,
                     display_name=None, next_pred=None, next_pred_label=None,
                     target_ch=None, target_name=None):
    """Render top K variables (all of them, up to MAX_VIZ_VARIABLES) in a single column,
    sorted by importance."""
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
        pred_value = (float(next_pred[ch]) if next_pred is not None and ch == target_ch
                      and ch < len(next_pred) else None)
        _plot_single(ax, signals[ch], attr[ch], col_names[ch], x, rank=i + 1,
                     show_xlabel=(i == show_n - 1), time_labels=time_labels,
                     pred_value=pred_value, pred_label=next_pred_label,
                     pred_region=next_pred is not None)
        if i == 0:
            ax.set_title(display_name or "Attribution", fontsize=20, color="black")

    # The figure grows with the channel count, so reserve the space below the last plot in
    # inches — a fixed fraction would balloon into a huge gap once there are many rows.
    has_time_labels = time_labels is not None and len(time_labels) == signals.shape[-1]
    bottom = (1.7 if has_time_labels else 0.8) / fig_height
    fig.subplots_adjust(bottom=bottom, hspace=0.3)
    _add_colorbar(fig, bottom_margin=0.34 / fig_height, target_name=target_name)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def _create_bundle_zip(signals, attr, col_names, sorted_idx, channel_importance, x, zip_path,
                       time_labels=None, next_pred=None, next_pred_label=None,
                       target_ch=None, target_name=None):
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
            pred_value = (float(next_pred[ch]) if next_pred is not None and ch == target_ch
                          and ch < len(next_pred) else None)
            _plot_single(ax, signals[ch], attr[ch], col_names[ch], x, rank=rank,
                         time_labels=time_labels, pred_value=pred_value,
                         pred_label=next_pred_label, pred_region=next_pred is not None)
            ax.set_title(f"#{rank} {col_names[ch]} — Attribution", fontsize=20, color="black") # , pad=10
            fig.subplots_adjust(bottom=0.22, top=0.88, left=0.12, right=0.92)
            _add_colorbar(fig, bottom_margin=0.08, target_name=target_name)
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
