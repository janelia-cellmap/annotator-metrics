import os
from typing import List
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
import seaborn
import warnings


def plot_all_to_all(
    group: str,
    crop: str,
    organelle_name: str,
    metric_value: str,
    metric_name: str,
    segmentation_types: List[str],
    all_to_all: np.ndarray,
    score_range: dict,
    output_path: str,
) -> None:
    """Plot the output all-to-all matrix for a given set of data.

    Args:
        group (str): Group to plot.
        crop (str): Crop to plot.
        organelle_name (str): Organelle to plot.
        metric_value (str): Metric value (contains underscores).
        metric_name (str): Metric name.
        segmentation_types (List[str]): List of all segmentations (annotations, predictions, refinements, ariadne, ...)
        all_to_all (np.ndarray): All-to-all matrix containing scores.
        score_range (dict): Dict containing information relevant to score plotting.
        output_path (str): Path to output images.
    """
    matplotlib.rcParams["figure.dpi"] = 300
    matplotlib.rcParams["axes.facecolor"] = "white"
    matplotlib.rcParams["savefig.facecolor"] = "white"

    # I expect to see RuntimeWarnings in this block for mean of empty slice
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", message="Mean of empty slice")
        if score_range["sorting"] == 1:  # low to high
            sort_order = np.argsort(np.nanmean(all_to_all, axis=0))
            cmap = "rocket_r"
        else:
            cmap = "rocket"
            sort_order = np.argsort(np.nanmean(all_to_all, axis=0))[::-1]

    all_to_all = all_to_all[:, sort_order]
    all_to_all = all_to_all[sort_order, :]
    segmentation_types = [segmentation_types[s] for s in sort_order]

    # write out csvs of data:
    output_directory = f"{output_path}/csvs/{group}/{crop}/{metric_value}"
    os.makedirs(output_directory, exist_ok=True)
    try:
        os.remove(f"{output_directory}/{organelle_name}.csv")
    except OSError:
        pass

    ignored = ["refinements", "ariadne"]
    with open(f"{output_directory}/{organelle_name}.csv", "w") as f:
        f.write(f"annotator 1,annotator 2,{metric_name}\n")
        for i in range(len(segmentation_types)):
            for j in range(i):
                if (
                    segmentation_types[i] not in ignored
                    and segmentation_types[j] not in ignored
                ):
                    f.write(
                        f"{segmentation_types[i]},{segmentation_types[j]},{all_to_all[i][j]}\n"
                    )

    for color_range in ["standard", "combined"]:
        _, ax = plt.subplots(1, 1, figsize=(8, 6),)
        if color_range == "standard":
            # I expect to see RuntimeWarnings in this block for all-nan slice
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    action="ignore", message="All-NaN slice encountere"
                )
                mask = ~np.eye(all_to_all.shape[0], dtype=bool)
                score_range_min = np.nanmin(all_to_all[mask])
                score_range_max = np.nanmax(all_to_all[mask])
                output_directory = (
                    f"{output_path}/plots_standard_color/{group}/{crop}/{metric_value}"
                )
        else:
            score_range_min = score_range["min"]
            score_range_max = score_range["max"]
            output_directory = f"{output_path}/plots/{group}/{crop}/{metric_value}"

        seaborn.heatmap(
            all_to_all,
            annot=True,
            square=True,
            vmin=score_range_min,
            vmax=score_range_max,
            cmap=cmap,
        )
        if score_range["sorting"] == 1:  # low to high
            plt.gcf().axes[1].invert_yaxis()

        ax.set_title(organelle_name)
        ax.collections[0].colorbar.set_label(metric_name)
        ax.xaxis.tick_top()
        plt.xticks(
            [i + 0.5 for i in range(len(segmentation_types))],
            segmentation_types,
            rotation=45,
            ha="left",
        )
        plt.yticks(
            [i + 0.5 for i in range(len(segmentation_types))],
            segmentation_types,
            rotation=0,
        )

        os.makedirs(output_directory, exist_ok=True)
        try:
            os.remove(f"{output_directory}/{organelle_name}.png")
        except OSError:
            pass
        plt.savefig(f"{output_directory}/{organelle_name}.png", bbox_inches="tight")
        plt.close()
