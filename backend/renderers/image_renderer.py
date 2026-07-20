import os
import numpy as np
import matplotlib.pyplot as plt


def render_heatmap(
    attribution_map: np.ndarray,
    output_path: str,
    colormap: str = "Reds",
) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    is_unsigned = attribution_map.min() >= 0

    limit = np.percentile(attribution_map if is_unsigned else np.abs(attribution_map), 99)
    if limit == 0:
        limit = attribution_map.max() if is_unsigned else np.abs(attribution_map).max()
    if limit == 0:
        limit = 1.0

    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=100)
    if is_unsigned:
        ax.imshow(attribution_map, cmap=colormap, vmin=0, vmax=limit)
    else:
        ax.imshow(attribution_map, cmap=colormap, vmin=-limit, vmax=limit)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    return output_path
