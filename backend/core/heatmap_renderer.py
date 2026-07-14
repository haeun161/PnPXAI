import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def render_heatmap(
    original_image: np.ndarray,
    attribution_map: np.ndarray,
    output_path: str,
    colormap: str = "RdBu_r",
    alpha: float = 0.5,
) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=100)
    ax.imshow(attribution_map, cmap=colormap, vmin=0, vmax=1)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    return output_path
