import os
from typing import List, Tuple, Union
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
from dask import delayed
import dask.distributed
from dask.distributed import Client
import pandas
import collections
import socket
import shutil
import warnings
from IPython.core.display import display, HTML

Result = collections.namedtuple(
    "Result",
    "crop organelle_name metric_value metric_name segmentation_types gt_idx test_idx sorting score",
)


def compare_two_images(
    gt_path: str,
    test_path: str,
    organelle_label: Union[int, list],
    resolution: Union[int, float],
    metrics_to_calculate: Union[str, list] = None,
) -> List[list]:
    """
    Compare two images according to the specified metric(s).

    Args:
        gt_path (str): Path to ground truth image
        test_path (str): Path to test image
        organelle_label (Union[int, list]): Organelle label(s) to compare
        resolution (Union[int, float]): Voxel resolution
        metrics_to_calculate (Union[str, list], optional):  Metric(s) used for image comparisons. If None or "all" provided, will compare images across all metrics. Defaults to None.

    Returns:
        List[list]: List of lists of metrics and associated scores
    """

    gt_image = tifffile.imread(gt_path)
    test_image = tifffile.imread(test_path)
    if type(organelle_label) == list:
        gt_image_binary = np.zeros(gt_image.shape, dtype=bool)
        test_image_binary = np.zeros(gt_image.shape, dtype=bool)
        for current_organelle_label in organelle_label:
            gt_image_binary |= gt_image == current_organelle_label
            test_image_binary |= test_image == current_organelle_label
    else:
        gt_image_binary = gt_image == organelle_label
        test_image_binary = test_image == organelle_label

    metric_params = {"tol_distance": 40, "clip_distance": 200, "threshold": 127}
    evaluator = Evaluator(
        gt_image_binary,
        test_image_binary,
        not gt_image_binary.any(),
        not test_image_binary.any(),
        metric_params,
        resolution=[resolution] * 3,
    )

    if not metrics_to_calculate or metrics_to_calculate == "all":
        metrics_to_calculate = [display_name(metric) for metric in EvaluationMetrics]

    scores = []
    for metric in EvaluationMetrics:
        if display_name(metric) in metrics_to_calculate:
            try:
                score = evaluator.compute_score(metric)
            except:
                score = float("NaN")
            if score == np.nan_to_num(np.inf):
                score = float("NaN")  # Necessary for plotting
            scores.append([metric, score],)
    return scores


def calculate_metric_scores(
    r: pandas.Series, metrics_to_calculate: Union[str, list] = "all",
) -> List[Result]:
    """Calculates the metric score(s) between two images and returns a list of the results.

    Args:
        r (pandas.Series): Row containing relevant paths, labels and resolution for getting metric score(s) of two images.
        metrics_to_calculate (Union[str, list], optional): Metric(s) to calculate when comparing two images. Defaults to "all".

    Returns:
        List[Result]: List of score results.
    """
    scores = compare_two_images(
        r.gt_path, r.test_path, r.organelle_label, r.resolution, metrics_to_calculate,
    )

    output_formatted = []
    for current_score_entry in scores:
        metric = current_score_entry[0]
        metric_score = current_score_entry[1]
        output_formatted.append(
            Result(
                r.crop,
                r.organelle_name,
                metric.value,
                display_name(metric),
                r.segmentation_types,
                r.gt_idx,
                r.test_idx,
                sorting(metric),
                metric_score,
            )
        )
    return output_formatted


def create_dataframe(
    group: str, crop: Union[list, str], input_base_path: str,
) -> pandas.DataFrame:
    """Generates dataframe from mask information of images to compare.

    Args:
        group (str): Group to use.
        crop (Union[list, str]): Crop(s) to use.
        input_base_path (str): Path to data.

    Returns:
        pandas.DataFrame: Data frame with rows containing information about images to compare.
    """
    mi = MaskInformation(group, crop)
    df_row_values = []
    for row in mi.rows:
        all_segmentations = os.listdir(f"{input_base_path}/{row.group}/{row.crop}")
        original_segmentation_types = [n.split(".")[0] for n in all_segmentations]
        original_image_paths = [
            f"{input_base_path}/{row.group}/{row.crop}/{f}" for f in all_segmentations
        ]
        for organelle_name, organelle_label in row.organelle_info.items():
            image_paths = []
            segmentation_types = []
            for idx, p in enumerate(original_image_paths):
                if not ("ariadne" in p and organelle_name != "mito"):
                    image_paths.append(p)
                    segmentation_types.append(original_segmentation_types[idx])

            if len(image_paths) > 1:
                for gt_idx, gt_path in enumerate(image_paths):
                    for test_idx, test_path in enumerate(image_paths):
                        df_row_values.append(
                            [
                                row.crop,
                                organelle_name,
                                organelle_label,
                                segmentation_types,
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
            "segmentation_types",
            "gt_idx",
            "test_idx",
            "gt_path",
            "test_path",
            "resolution",
        ],
    )
    return df


def compile_results_for_plotting(results: List[Result]) -> Tuple[dict, dict]:
    """Combine all the metric results into a format that is more easy to deal with for plotting.

    Args:
        results (List[Result]): List of metric score results.

    Returns:
        Tuple[dict, dict]: Dict containing all-to-all scores and dict containing corresponding information about scores.
    """
    all_to_all = {}
    score_ranges = {}
    for result in results:
        score_tuple = (result.crop, result.organelle_name, result.metric_value)
        if score_tuple not in all_to_all:
            all_to_all[score_tuple] = {
                "scores_matrix": np.zeros(
                    (len(result.segmentation_types), len(result.segmentation_types))
                ),
                "segmentation_types": result.segmentation_types,
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

    _, ax = plt.subplots(1, 1, figsize=(8, 6),)

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
        if color_range == "standard":
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


def calculate_all_to_all(
    group: str,
    input_path: str,
    output_path: str,
    metrics_to_calculate: Union[list, str] = "all",
    num_workers: int = 10,
    crop: Union[list, str] = None,
) -> Tuple[dict, dict]:
    """Calculates all-to-all matrices for specified metric(s).

    Args:
        group (str): Group to use.
        input_base_path (str): Path to data.
        output_path (str): Path to write out data.
        metrics_to_calculate (Union[list, str], optional): Metric(s) to calculate scores for. Defaults to "all".
        num_workers (int, optional): Number of dask workers. Defaults to 10.
        crop (Union[list, str], optional): Crop(s) to calculate scores for. Defaults to None.

    Returns:
        Tuple[dict, dict]:  Dict containing all-to-all scores and dict containing corresponding information about scores.
    """
    df = create_dataframe(group, crop, input_path)

    # Setup dask client
    dask.config.set({"distributed.comm.timeouts.connect": 100})

    with Client(n_workers=num_workers, threads_per_worker=1) as client:
        # client = Client(n_workers=num_workers, threads_per_worker=1)
        local_ip = socket.gethostbyname(socket.gethostname())
        url = client.dashboard_link.replace("127.0.0.1", local_ip)

        display(
            HTML(
                f"""<a href="{url}">Click here to montior all-to-all calculation progress.</a>"""
            ),
        )

        lazy_results = []
        for _, row in df.iterrows():
            lazy_results.append(
                delayed(calculate_metric_scores)(row, metrics_to_calculate)
            )

        results = dask.compute(*lazy_results)
        results = [metric_row for result in results for metric_row in result]
        all_to_all, score_ranges = compile_results_for_plotting(results)

        lazy_results = []
        for score_tuple, score_entry in all_to_all.items():
            crop = score_tuple[0]
            organelle_name = score_tuple[1]
            metric_value = score_tuple[2]
            metric_name = score_entry["metric_name"]
            segmentation_types = score_entry["segmentation_types"]
            current_all_to_all = score_entry["scores_matrix"]
            score_range = score_ranges[(organelle_name, metric_value)]

            # clean up plot dirs before creating images
            for plot_type in ["plots_standard_color", "plots"]:
                output_directory = (
                    f"{output_path}/{plot_type}/{group}/{crop}/{metric_value}"
                )
                if os.path.isdir(output_directory):
                    shutil.rmtree(output_directory)
            output_directory = f"{output_path}/csvs/{group}/{crop}/{metric_value}"
            if os.path.isdir(output_directory):
                shutil.rmtree(output_directory)

            lazy_results.append(
                delayed(plot_figure)(
                    group,
                    crop,
                    organelle_name,
                    metric_value,
                    metric_name,
                    segmentation_types,
                    current_all_to_all,
                    score_range,
                    output_path=output_path,
                )
            )

        dask.compute(*lazy_results)

    return all_to_all, score_ranges
