from contextlib import ExitStack
import os
from typing import List, Tuple, Union
import dask
import numpy as np
import tifffile

from annotator_metrics.util.io_util import print_with_datetime
from annotator_metrics.util.plot_util import plot_all_to_all
from annotator_metrics.src.preprocessing import follow_symlinks

from ..util.url_util import display_url
from ..util.doc_util import MaskInformation
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
import traceback
import logging

logger = logging.getLogger(__name__)


Result = collections.namedtuple(
    "Result",
    "crop organelle_name metric_value metric_name segmentation_types gt_idx test_idx sorting score",
)


def compare_two_images(
    gt_path: str,
    test_path: str,
    organelle_label: Union[int, list],
    resolution: Union[int, float],
    metrics_to_calculate: Union[str, list] = "all",
) -> List[list]:
    """
    Compare two images according to the specified metric(s).

    Args:
        gt_path (str): Path to ground truth image
        test_path (str): Path to test image
        organelle_label (Union[int, list]): Organelle label(s) to compare
        resolution (Union[int, float]): Voxel resolution
        metrics_to_calculate (Union[str, list], optional):  Metric(s) used for image comparisons. If None or "all" provided, will compare images across all metrics. Defaults to "all".

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
        mask=None,
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
            scores.append(
                [metric, score],
            )
    return scores


def calculate_metric_scores(
    r: pandas.Series,
    metrics_to_calculate: Union[str, list] = "all",
) -> List[Result]:
    """Calculates the metric score(s) between two images and returns a list of the results.

    Args:
        r (pandas.Series): Row containing relevant paths, labels and resolution for getting metric score(s) of two images.
        metrics_to_calculate (Union[str, list], optional): Metric(s) to calculate when comparing two images. Defaults to "all".

    Returns:
        List[Result]: List of score results.
    """
    scores = compare_two_images(
        r.gt_path,
        r.test_path,
        r.organelle_label,
        r.resolution,
        metrics_to_calculate,
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
    group: Union[list, str],
    crop: Union[list, str],
    input_base_path: str,
) -> pandas.DataFrame:
    """Generates dataframe from mask information of images to compare.

    Args:
        group (Union[list,str]): Group(s) to use.
        crop (Union[list, str]): Crop(s) to use.
        input_base_path (str): Path to data.

    Returns:
        pandas.DataFrame: Data frame with rows containing information about images to compare.
    """
    mi = MaskInformation(group, crop, input_base_path)
    pred_seg_path = "/groups/cellmap/cellmap/ackermand/forDavis/renumbered"
    df_row_values = []
    for row in mi.rows:
        all_segmentations = os.listdir(f"{input_base_path}/{row.group}/{row.crop}")
        original_segmentation_types = [n.split(".")[0] for n in all_segmentations]
        original_image_paths = [
            f"{input_base_path}/{row.group}/{row.crop}/{f}" for f in all_segmentations
        ]
        original_image_id_sets = [set(np.unique(tifffile.imread(original_image_path))) for original_image_path in original_image_paths]
        for organelle_name, organelle_label in row.organelle_info.items():
            image_paths = []
            segmentation_types = []
            for idx, p in enumerate(original_image_paths):
                use_image = True
                if "ariadne" in p and organelle_name != "mito":
                    use_image = False
                if original_segmentation_types[idx] in ["predictions", "refinements"]:
                    if original_segmentation_types[idx] == "predictions":
                        annotation_type = "pred"
                    else:
                        annotation_type = "seg"
                    # for things with lumen, we use the base class and label everything else, lumen being what is left behind
                    # for things with in, we use the base class and label out, in being what is left behind
                    is_lumen = False
                    is_in = False

                    adjusted_name = organelle_name
                    if "-lum" in adjusted_name:
                        adjusted_name = adjusted_name.split("-lum")[0]
                        is_lumen = True
                    if "-in" in adjusted_name:
                        adjusted_name = adjusted_name.split("-in")[0]
                        is_in = True

                    path = f"{pred_seg_path}/{row.cell_name}/{row.cell_name}.n5/{adjusted_name}"
                    if not os.path.exists(follow_symlinks(f"{path}_{annotation_type}")):
                        use_image = False
                    elif is_lumen and not os.path.exists(
                        follow_symlinks(f"{path}-mem_{annotation_type}")
                    ):
                        # can only calculate lumen if we have a mem
                        use_image = False
                    elif is_in and not os.path.exists(
                        follow_symlinks(f"{path}-out_{annotation_type}")
                    ):
                        # can only calculate inner part if we have an outer part
                        use_image = False

                if use_image:
                    if type(organelle_label) is not list:
                        organelle_label = [organelle_label]
                    organelle_label_set = set(organelle_label)
                    
                    if set(original_image_id_sets[idx]).intersection(organelle_label_set):
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
                "min": float("nan"),
                "max": float("nan"),
                "sorting": result.sorting,
            }
        if result.gt_idx != result.test_idx:
            current_range = score_ranges[score_ranges_tuple]
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    action="ignore", message="All-NaN axis encountered"
                )
                current_range["min"] = np.nanmin([current_range["min"], result.score])
                current_range["max"] = np.nanmax([current_range["max"], result.score])
    return all_to_all, score_ranges


def calculate_all_to_all(
    group: Union[list, str],
    input_path: str,
    output_path: str,
    metrics_to_calculate: Union[list, str] = "all",
    num_workers: int = None,
    crop: Union[list, str] = "all",
) -> Tuple[dict, dict]:
    """Calculates all-to-all matrices for specified metric(s).

    Args:
        group (Union[list, str]): Group(s) to use.
        input_base_path (str): Path to data.
        output_path (str): Path to write out data.
        metrics_to_calculate (Union[list, str], optional): Metric(s) to calculate scores for. Defaults to "all".
        num_workers (int, optional): Number of dask workers. Defaults to None.
        crop (Union[list, str], optional): Crop(s) to calculate scores for. Defaults to "all".

    Returns:
        Tuple[dict, dict]:  Dict containing all-to-all scores and dict containing corresponding information about scores.
    """
    df = create_dataframe(group, crop, input_path)

    # Setup dask client
    with ExitStack() as stack:
        if num_workers:
            dask.config.set({"distributed.comm.timeouts.connect": 100})
            stack.enter_context(Client(n_workers=num_workers, threads_per_worker=1))
            client = Client.current()
            local_ip = socket.gethostname()
            url = client.dashboard_link.replace("127.0.0.1", local_ip)
            display_url(url, "Click here to monitor all-to-all calculation progress")

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
                delayed(plot_all_to_all)(
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


def main():
    from annotator_metrics.src.preprocessing import copy_data
    from annotator_metrics.util.image_util import (
        create_variance_images,
        get_neuroglancer_view,
    )
    import annotator_metrics.util.io_util as io_util
    import annotator_metrics.util.dask_util as dask_util

    # Get information regarding run
    submission_directory = os.getcwd()
    args = io_util.parser_params()
    num_workers = args.num_workers
    required_settings, optional_settings = io_util.read_run_config(args.config_path)

    # Setup config parameters
    output_path = required_settings["output_path"]
    group = required_settings["group"]
    crop = optional_settings["crop"]
    metrics_to_calculate = optional_settings["metrics_to_calculate"]

    group = [group.split(",") if "," in group else group][0]
    crop = [crop.split(",") if "," in crop else crop][0]
    metrics_to_calculate = [
        (
            metrics_to_calculate.split(",")
            if "," in metrics_to_calculate
            else metrics_to_calculate
        )
    ][0]

    # Change execution directory
    execution_directory = dask_util.setup_execution_directory(args.config_path, logger)
    logpath = f"{execution_directory}/output.log"

    # Start processing
    with io_util.tee_streams(logpath):

        try:
            os.chdir(execution_directory)

            # Copy data to output directory
            with io_util.Timing_Messager(f"Copying data for {group}.", logger):
                copy_data(
                    group=group,
                    output_path=f"{output_path}/data/",
                    crop=crop,
                    include_nonannotator_results=False,
                )

            # Start dask
            # Since we set up the dask worker before entering, we don't need num_workers
            for current_group in group:
                with dask_util.start_dask(num_workers, "all-to-all", logger):
                    with io_util.Timing_Messager(
                        f"Calculating all-to-all for {current_group}", logger
                    ):
                        calculate_all_to_all(
                            group=current_group,
                            input_path=f"{output_path}/data/",
                            output_path=f"{output_path}/results/",
                            num_workers=None,
                            metrics_to_calculate=metrics_to_calculate,
                        )

                # Restart dask to clean up cluster before variance image creation
                with dask_util.start_dask(num_workers, "variance", logger):
                    with io_util.Timing_Messager(
                        f"Creating variance images for {current_group}, crop {crop}", logger
                    ):
                        create_variance_images(
                            input_path=f"{output_path}/data/",
                            group=current_group,
                            crop=crop,
                            output_path=f"{output_path}/results/n5s",
                            num_workers=None,
                        )

                # Get a neuroglancer view of the crop. The http-served directory is /groups/cellmap/cellmap
                with io_util.Timing_Messager(
                    f"Generating neuroglancer image for {current_group}.", logger
                ):
                    get_neuroglancer_view(
                        n5s_path=f"{output_path}/results/n5s",
                        group=current_group,
                        crop=crop,
                    )
            print_with_datetime("Calculations completed successfully!", logger)
        except Exception:
            print_with_datetime(traceback.format_exc(), logger)
            print_with_datetime("Calculations failed!", logger)
        finally:
            os.chdir(submission_directory)


if __name__ == "__main__":
    """Run main function"""
    main()
