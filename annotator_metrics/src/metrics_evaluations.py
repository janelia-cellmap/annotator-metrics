import os
import matplotlib
import numpy as np
import tifffile
from ..util.doc_io import MaskInformation
from matplotlib import pyplot as plt
import seaborn
from CNNectome.validation.organelles.segmentation_metrics import (
    Evaluator,
    EvaluationMetrics,
    display_name,
    sorting,
)


def calculate_all_to_all(group_id: str, input_base_path: str):
    mi = MaskInformation()
    matplotlib.rcParams["figure.dpi"] = 300
    all_to_all_by_crop = {}
    names_by_crop = {}
    group_rows = [row for row in mi.rows if row.group == group_id and row.crop == "04"]
    for row in group_rows:
        all_segmentations = os.listdir(f"{input_base_path}/{row.group}/{row.crop}")
        names_by_crop[row.crop] = [n.split(".")[0] for n in all_segmentations]
        all_to_all_by_crop[row.crop] = {}
        for organelle_name in row.organelle_info.keys():
            all_to_all_by_crop[row.crop][organelle_name] = {}
            for metric in EvaluationMetrics:
                metric_name = display_name(metric)
                all_to_all_by_crop[row.crop][organelle_name][metric_name] = np.zeros(
                    (len(all_segmentations), len(all_segmentations))
                )

        images_list = [
            tifffile.imread(f"{input_base_path}/{row.group}/{row.crop}/{f}")
            for f in all_segmentations
        ]

        metric_params = {
            "tol_distance": 40,
            "clip_distance": 200,
            "threshold": 127,
        }

        for gt_idx, gt_image in enumerate(images_list):
            for test_idx, test_image in enumerate(images_list):
                for organelle_name, organelle_label in row.organelle_info.items():
                    if organelle_label == 0:
                        gt_image_binary = (gt_image >= 3) & (gt_image <= 5)
                        test_image_binary = (test_image >= 3) & (test_image <= 5)
                    else:
                        gt_image_binary = gt_image == organelle_label
                        test_image_binary = test_image == organelle_label
                    evaluator = Evaluator(
                        gt_image_binary,
                        test_image_binary,
                        not gt_image_binary.any(),
                        not test_image_binary.any(),
                        metric_params,
                        resolution=[row.correct_resolution] * 3,
                    )
                    for metric in EvaluationMetrics:
                        metric_name = display_name(metric)
                        if metric_name == "F1 Score":
                            try:
                                score = evaluator.compute_score(metric)
                            except:
                                score = float("NaN")
                            if score == np.nan_to_num(np.inf):
                                score = float("NaN")  # Necessary for plotting

                        all_to_all_by_crop[row.crop][organelle_name][metric_name][
                            gt_idx
                        ][test_idx] = score

    score_ranges = {}
    for row in group_rows:
        for organelle_name in row.organelle_info.keys():
            metric_name = display_name(metric)
            if organelle_name not in score_ranges:
                score_ranges[organelle_name] = {}
            for metric in EvaluationMetrics:
                metric_name = display_name(metric)
                if metric_name not in score_ranges[organelle_name]:
                    score_ranges[organelle_name][metric_name] = {
                        "min": np.nan_to_num(np.inf),
                        "max": np.nan_to_num(-np.inf),
                        "sorting": sorting(metric),
                    }

                current_scores_matrix = all_to_all_by_crop[row.crop][organelle_name][
                    metric_name
                ]

                # exclude diagonal
                mask = np.ones(current_scores_matrix.shape, dtype=bool)
                np.fill_diagonal(mask, 0)

                current_min = np.nanmin(current_scores_matrix[mask])
                current_max = np.nanmax(current_scores_matrix[mask])

                current_range = score_ranges[organelle_name][metric_name]

                current_range["min"] = np.nanmin([current_range["min"], current_min])
                current_range["max"] = np.nanmax([current_range["max"], current_max])

    for crop_id, organelle_dict in all_to_all_by_crop.items():
        for organelle_name, metric_dict in organelle_dict.items():
            for metric in EvaluationMetrics:
                current_names_to_use = names_by_crop[crop_id]
                metric_name = display_name(metric)
                if metric_name == "F1 Score":
                    _, ax = plt.subplots(1, 1, figsize=(8, 6),)
                    current_all_to_all = metric_dict[metric_name].copy()
                    score_range = score_ranges[organelle_name][metric_name]
                    if sorting(metric) == 1:  # low to high
                        sort_order = np.argsort(np.nanmean(current_all_to_all, axis=1))
                        cmap = "rocket_r"
                    else:
                        cmap = "rocket"
                        sort_order = np.argsort(np.nanmean(current_all_to_all, axis=1))[
                            ::-1
                        ]

                    current_all_to_all = current_all_to_all[:, sort_order]
                    current_all_to_all = current_all_to_all[sort_order, :]
                    current_names_to_use = [current_names_to_use[s] for s in sort_order]
                    seaborn.heatmap(
                        current_all_to_all,
                        annot=True,
                        square=True,
                        vmin=score_range["min"],
                        vmax=score_range["max"],
                        cmap=cmap,
                    )
                    if score_range["sorting"] == 1:  # low to high
                        plt.gcf().axes[1].invert_yaxis()

                    ax.set_title(organelle_name)
                    ax.collections[0].colorbar.set_label(display_name(metric))
                    ax.xaxis.tick_top()
                    plt.xticks(
                        [i + 0.5 for i in range(len(current_names_to_use))],
                        current_names_to_use,
                        rotation=45,
                        ha="left",
                    )
                    plt.yticks(
                        [i + 0.5 for i in range(len(current_names_to_use))],
                        current_names_to_use,
                        rotation=0,
                    )
                    output_directory = (
                        f"{input_base_path}/plots/{group_id}/{crop_id}/{metric.value}"
                    )
                    os.system(f"mkdir -p {output_directory}")
                    os.system(f"rm {output_directory}/{organelle_name}.png")
                    plt.savefig(
                        f"{output_directory}/{organelle_name}.png", bbox_inches="tight"
                    )
                    plt.close()

