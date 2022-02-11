import os
from typing import List
import dask
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
from dask import delayed, compute
import dask.distributed
from dask.distributed import Client
import pandas
from dask.diagnostics import ProgressBar
import collections

Result = collections.namedtuple(
    "Result",
    "crop organelle_name metric_value metric_name annotator_names gt_idx test_idx sorting score",
)


def compute_row_score(r) -> List[list]:
    gt_image = tifffile.imread(r.gt_path)
    test_image = tifffile.imread(r.test_path)
    if r.organelle_label == 0:
        gt_image_binary = (gt_image >= 3) & (gt_image <= 5)
        test_image_binary = (test_image >= 3) & (test_image <= 5)
    else:
        gt_image_binary = gt_image == r.organelle_label
        test_image_binary = test_image == r.organelle_label

    metric_params = {"tol_distance": 40, "clip_distance": 200, "threshold": 127}
    evaluator = Evaluator(
        gt_image_binary,
        test_image_binary,
        not gt_image_binary.any(),
        not test_image_binary.any(),
        metric_params,
        resolution=[r.resolution] * 3,
    )
    output_rows = []
    for metric in EvaluationMetrics:
        try:
            score = evaluator.compute_score(metric)
        except:
            score = float("NaN")
        if score == np.nan_to_num(np.inf):
            score = float("NaN")  # Necessary for plotting
        output_rows.append(
            Result(
                r.crop,
                r.organelle_name,
                metric.value,
                display_name(metric),
                r.annotator_names,
                r.gt_idx,
                r.test_idx,
                sorting(metric),
                score,
            )
        )
    return output_rows


def create_dataframe(group_id: str, input_base_path: str) -> pandas.DataFrame:
    mi = MaskInformation()
    group_rows = [row for row in mi.rows if row.group == group_id]
    df_row_values = []
    for row in group_rows:
        all_segmentations = os.listdir(f"{input_base_path}/{row.group}/{row.crop}")
        original_annotator_names = [n.split(".")[0] for n in all_segmentations]
        original_image_paths = [
            f"{input_base_path}/{row.group}/{row.crop}/{f}" for f in all_segmentations
        ]
        for organelle_name, organelle_label in row.organelle_info.items():
            image_paths = []
            annotator_names = []
            for idx, p in enumerate(original_image_paths):
                if not ("ariadne" in p and organelle_name != "mito"):
                    image_paths.append(p)
                    annotator_names.append(original_annotator_names[idx])

            for gt_idx, gt_path in enumerate(image_paths):
                for test_idx, test_path in enumerate(image_paths):
                    df_row_values.append(
                        [
                            row.crop,
                            organelle_name,
                            organelle_label,
                            annotator_names,
                            gt_idx,
                            test_idx,
                            gt_path,
                            test_path,
                            row.correct_resolution,
                        ]
                    )
    df = pandas.DataFrame(
        df_row_values,
        columns=[
            "crop",
            "organelle_name",
            "organelle_label",
            "annotator_names",
            "gt_idx",
            "test_idx",
            "gt_path",
            "test_path",
            "resolution",
        ],
    )
    return df


def compile_results_for_plotting(results):
    all_to_all = {}
    score_ranges = {}
    for result in results:
        score_tuple = (result.crop, result.organelle_name, result.metric_value)
        if score_tuple not in all_to_all:
            all_to_all[score_tuple] = {
                "scores_matrix": np.zeros(
                    (len(result.annotator_names), len(result.annotator_names))
                ),
                "annotator_names": result.annotator_names,
                "metric_name": result.metric_name,
            }
        all_to_all[score_tuple]["scores_matrix"][result.gt_idx][
            result.test_idx
        ] = result.score

        score_ranges_tuple = (result.organelle_name, result.metric_value)
        if score_ranges_tuple not in score_ranges:
            score_ranges[score_ranges_tuple] = {
                "min": np.nan_to_num(np.inf),
                "max": np.nan_to_num(-np.inf),
                "sorting": result.sorting,
            }
        if result.gt_idx != result.test_idx:
            current_range = score_ranges[score_ranges_tuple]
            current_range["min"] = np.nanmin([current_range["min"], result.score])
            current_range["max"] = np.nanmax([current_range["max"], result.score])
    return all_to_all, score_ranges


def plot_figure(
    group,
    crop,
    organelle_name,
    metric_value,
    metric_name,
    annotator_names,
    all_to_all,
    score_range,
    output_path,
):
    matplotlib.rcParams["figure.dpi"] = 300
    matplotlib.rcParams["axes.facecolor"] = "white"
    matplotlib.rcParams["savefig.facecolor"] = "white"

    _, ax = plt.subplots(1, 1, figsize=(8, 6),)
    if score_range["sorting"] == 1:  # low to high
        sort_order = np.argsort(np.nanmean(all_to_all, axis=1))
        cmap = "rocket_r"
    else:
        cmap = "rocket"
        sort_order = np.argsort(np.nanmean(all_to_all, axis=1))[::-1]

    all_to_all = all_to_all[:, sort_order]
    all_to_all = all_to_all[sort_order, :]
    annotator_names = [annotator_names[s] for s in sort_order]
    seaborn.heatmap(
        all_to_all,
        annot=True,
        square=True,
        vmin=score_range["min"],
        vmax=score_range["max"],
        cmap=cmap,
    )
    if score_range["sorting"] == 1:  # low to high
        plt.gcf().axes[1].invert_yaxis()

    ax.set_title(organelle_name)
    ax.collections[0].colorbar.set_label(metric_name)
    ax.xaxis.tick_top()
    plt.xticks(
        [i + 0.5 for i in range(len(annotator_names))],
        annotator_names,
        rotation=45,
        ha="left",
    )
    plt.yticks(
        [i + 0.5 for i in range(len(annotator_names))], annotator_names, rotation=0,
    )
    output_directory = f"{output_path}/plots/{group}/{crop}/{metric_value}"

    os.makedirs(output_directory, exist_ok=True)
    try:
        os.remove(f"{output_directory}/{organelle_name}.png")
    except OSError:
        pass
    plt.savefig(f"{output_directory}/{organelle_name}.png", bbox_inches="tight")
    plt.close()


def calculate_all_to_all(group_id: str, input_base_path: str, num_workers: int = 10):
    df = create_dataframe(group_id, input_base_path)

    # Setup dask client
    dask.config.set({"distributed.comm.timeouts.connect": 100})
    client = Client(n_workers=num_workers, threads_per_worker=1)
    print(client.dashboard_link)

    lazy_results = []
    for _, row in df.iterrows():
        lazy_results.append(delayed(compute_row_score)(row))

    results = dask.compute(*lazy_results)
    results = [metric_row for result in results for metric_row in result]
    all_to_all, score_ranges = compile_results_for_plotting(results)

    lazy_results = []
    for score_tuple, score_entry in all_to_all.items():
        crop = score_tuple[0]
        organelle_name = score_tuple[1]
        metric_value = score_tuple[2]
        metric_name = score_entry["metric_name"]
        annotator_names = score_entry["annotator_names"]
        current_all_to_all = score_entry["scores_matrix"]
        score_range = score_ranges[(organelle_name, metric_value)]
        lazy_results.append(
            delayed(plot_figure)(
                group_id,
                crop,
                organelle_name,
                metric_value,
                metric_name,
                annotator_names,
                current_all_to_all,
                score_range,
                output_path="/groups/cosem/cosem/ackermand/annotation_and_analytics/",
            )
        )

    dask.compute(*lazy_results)
