import os
from typing import Dict, Tuple, Union
import numpy as np
import h5py
import pandas
import zarr
from ..util.doc_util import (
    MaskInformation,
    Row,
    get_prediction_paths_df,
)
from ..util.image_util import Cropper
import tifffile
import glob
import shutil

predictions_labels_dict = {
    "mito": 4,
    "mito-mem": 3,
    "mito-dna": 5,
    "er": 17,
    "er-mem": 16,
    "eres": 19,
    "eres-mem": 18,
    "golgi": 7,
    "golgi-mem": 6,
    "pm": 2,
    "endo": 11,
    "endo-mem": 10,
    "vesicle": 9,
    "vesicle-mem": 8,
    "lyso": 13,
    "lyso-mem": 12,
    "mt": 36,
    "mt-out": 30,
    "ne": 21,
    "ne-mem": 20,
    "np": 23,
    "np-out": 22,
    "hchrom": 24,
    "nhchrom": 25,
    "echrom": 26,
    "nechrom": 27,
    "nucleoplasm": 28,
    "nucleolus": 29,
    "cent": 31,
    "cent-sdapp": 32,
    "subdistal_app": 33,
}
# if lumen, put it first


def update_path(path: str) -> str:

    path = os.path.abspath(path)

    path = path.replace("nrs/cosem/", "nrs/cellmap/")
    path = path.replace("groups/cosem/cosem/", "groups/cellmap/cellmap/")

    if (
        os.path.islink(path)
        and "/nrs/cellmap/cosem/training/v0003.2" in path
        and not os.path.exists(path)
    ):
        path = path.replace(
            "/nrs/cellmap/cosem/training/v0003.2",
            "/nearline/cellmap/cosem/training/v0003.2/FROM_NRS/",
        )
        return path

    if not os.path.islink(path) and not os.path.exists(
        path
    ):  # then it is a broken symlink, or a path that doesnt exist
        if "groups/cellmap/cellmap" in path:
            # maybe moved to nrs or nearline
            path = path.replace("/groups/cellmap/", "/nrs/")
        if not os.path.islink(path) and not os.path.exists(path):
            path = path.replace("/nrs/", "/nearline/")
    return path


def update_symlink(path: str) -> str:
    path = update_path(path)
    if os.path.islink(path):
        symlink_path = os.readlink(path)
        if symlink_path[0] != "/":  # then it is a relative path
            symlink_path = path.rsplit("/", 1)[0] + "/" + symlink_path
        print(f"unlink {path}")
        print(f"ln -s {update_path(symlink_path)} {path}")
        os.system(f"unlink {path}")
        os.system(f"ln -s {update_path(symlink_path)} {path}")
        path = update_symlink(symlink_path)
    return path


def update_symlinks(path_glob: str) -> None:
    for cell_dir in glob.glob(path_glob):
        cell_name = cell_dir.split("/")[-1]
        for dataset in glob.glob(f"{cell_dir}/{cell_name}.n5/*"):
            if os.path.islink(dataset):
                update_symlink(dataset)
        if os.path.islink(f"{cell_dir}/mesh"):
            update_symlink(f"{cell_dir}/mesh")
        for dataset in glob.glob(f"{cell_dir}/mesh/*"):
            if os.path.islink(dataset):
                update_symlink(dataset)


def follow_symlinks(path: str) -> str:
    path = update_path(path)
    if os.path.islink(path):
        symlink_path = os.readlink(path)
        if symlink_path[0] != "/":  # then it is a relative path
            symlink_path = path.rsplit("/", 1)[0] + "/" + symlink_path
        path = follow_symlinks(symlink_path)
    return path


def get_resolution_and_offset_from_zarr(
    zarr_array: zarr.Group,
) -> Tuple[Union[int, float], np.ndarray]:
    attrs = zarr_array.attrs.asdict()
    if "transformation" in attrs:
        resolution = attrs["transformation"]["scale"][0]
    elif "transform" in attrs:
        resolution = attrs["transform"]["scale"][0]
    elif "pixelResolution" in attrs:
        resolution = attrs["pixelResolution"]["dimensions"][0]
    else:
        resolution = attrs["resolution"][0]

    offset = np.zeros((3,), dtype=int)
    if "offset" in attrs:
        offset = np.array(attrs["offset"], dtype=int)

    return resolution, offset


def get_predictions_and_refinements_from_row(
    row: Row,
    prediction_paths_df: pandas.DataFrame,
    base_path: str = "/groups/cellmap/cellmap/ackermand/forDavis/renumbered/",
) -> Dict[str, np.ndarray]:
    # need to figure out labeling
    cell_name = row.cell_name

    full_image_dict = {}
    for result_type in ["pred", "seg", "ariadne"]:
        has_segmentation = False
        for organelle_name, organelle_label in predictions_labels_dict.items():
            if organelle_label in row.organelle_info.values():
                result_path = None
                if result_type == "pred":
                    # is this necessary for predictions? currently not used
                    df_row = prediction_paths_df.loc[
                        (prediction_paths_df["Group"] == f"{row.group}_{row.crop}")
                        & (prediction_paths_df["Dataset"] == cell_name)
                        & (prediction_paths_df["Class"] == organelle_name)
                    ]
                    if not df_row.empty:
                        result_path = df_row["Prediction Pathway"].values[0]
                else:
                    result_path = follow_symlinks(
                        f"{base_path}/{organelle_name}_{result_type}"
                    )
                result_path = follow_symlinks(
                    f"{base_path}/{organelle_name}_{result_type}"
                )
                # result_path = follow_symlinks(
                #     f"{base_path}/{organelle_name}_{result_type}"
                # )
                # HACK: special case for mus liver crops outside of main region
                # if cell_name == "jrc_mus-liver":
                #     result_path = follow_symlinks(
                #         f"{base_path}/{row.group}_{row.crop}.n5/{organelle_name}_{result_type}"
                #     )

                if result_path and os.path.isdir(result_path):
                    n5, dataset = result_path.rsplit(".n5/", 1)
                    zarr_file = zarr.open(f"{n5}.n5", mode="r")
                    resolution, offset = get_resolution_and_offset_from_zarr(
                        zarr_file[dataset]
                    )

                    if result_type == "ariadne":
                        if row.group == "group1" and row.crop == "09":
                            offset = np.array([30984, 30912, 15728], dtype=int)
                        else:
                            break

                    crop_start = row.converted_4nm_coordinates - offset // 4
                    crop_end = crop_start + (
                        row.original_crop_size // (4 // row.correct_resolution)
                    )

                    rescale_factor_for_annotation = 4 // row.correct_resolution

                    if not has_segmentation:
                        combined_image = np.zeros(
                            (crop_end - crop_start)[::-1], dtype=np.uint8
                        )
                        has_segmentation = True

                    # need to account for ariadne, with resolution 8 nm. so the crop will be half the size of the 4 nm one
                    scale = int(resolution) // 4
                    if scale != 1:
                        crop_start_padded = crop_start // scale
                        crop_end_padded = -1 * (-crop_end // scale)

                        im = zarr_file[dataset][
                            crop_start_padded[2] : crop_end_padded[2],
                            crop_start_padded[1] : crop_end_padded[1],
                            crop_start_padded[0] : crop_end_padded[0],
                        ]

                        adjusted_start = crop_start - crop_start_padded * scale
                        adjusted_end = adjusted_start + (crop_end - crop_start)
                        cropper = Cropper(adjusted_start, adjusted_end)
                        im = cropper.crop(im, scale)
                    else:
                        im = zarr_file[dataset][
                            crop_start[2] : crop_end[2],
                            crop_start[1] : crop_end[1],
                            crop_start[0] : crop_end[0],
                        ]

                    combined_image[
                        im >= (127 if result_type == "pred" else 1)
                    ] = organelle_label

        if has_segmentation:
            combined_image = (
                combined_image.repeat(rescale_factor_for_annotation, axis=0)
                .repeat(rescale_factor_for_annotation, axis=1)
                .repeat(rescale_factor_for_annotation, axis=2)
            )
            if result_type == "pred":
                output_name = "predictions"
            elif result_type == "seg":
                output_name = "refinements"
            elif result_type == "ariadne":
                output_name = "ariadne"
            full_image_dict[output_name] = combined_image

    return full_image_dict


def crop_annotations(
    group: str, crop: str, cropper: Cropper, current_output_path: str
) -> None:
    input_base_path = "/groups/cellmap/cellmap/annotation_and_analytics/training"

    upscale_factor = 1
    if group == "group5" and crop in ["04", "05", "06"]:
        # HACK: mus liver for these crops were annotated by everyone at 8nm, not just the original gt
        # so need to upsample to get it at correct
        upscale_factor = 2

    for annotator_name in os.listdir(input_base_path):
        annotator_dir = f"{input_base_path}/{annotator_name}"
        if os.path.isdir(annotator_dir) and annotator_name not in [
            "conversion_scripts",
            "textfile_templates",
        ]:
            group_dir = f"{annotator_dir}/{group}-labels"
            if os.path.exists(group_dir):
                for trial in os.listdir(group_dir):
                    trial_dir = f"{group_dir}/{trial}"
                    if os.path.isdir(trial_dir) and f"_{crop}_" in trial:
                        if (
                            trial[-2] == "_"
                        ):  # HACK: for leaving out the number in the dir/file name
                            trial = trial[:-1] + "1" + trial[-1:]

                        im_file = f"{trial_dir}/{trial}.tif"
                        if os.path.isfile(im_file):
                            try:
                                im = tifffile.imread(im_file)
                                output_name = trial.split("_")[-1][::-1]
                                im_cropped = cropper.crop(
                                    im, upscale_factor=upscale_factor
                                )
                                tifffile.imwrite(
                                    f"{current_output_path}/{output_name}.tif",
                                    im_cropped,
                                )
                            except:
                                pass

    # cellmap annotators are in a different directory
    cellmap_annotator_dir = (
        f"/groups/cellmap/cellmap/annotations/training/{group}-labels/{group}_{crop}/"
    )
    if os.path.isdir(cellmap_annotator_dir):
        annotator_tifs = glob.glob(f"{cellmap_annotator_dir}/*.tif")
        if annotator_tifs:
            for annotator_tif in annotator_tifs:
                # then don't need to check subdirectories
                output_name = annotator_tif.split(".tif")[0][-2::][::-1]
                im = tifffile.imread(annotator_tif)
                im_cropped = cropper.crop(im, upscale_factor=upscale_factor)
                tifffile.imwrite(f"{current_output_path}/{output_name}.tif", im_cropped)
        else:
            for trial in os.listdir(cellmap_annotator_dir):
                output_name = trial.split("_")[-1][::-1]
                if output_name[0] in ["a", "b", "f"]:
                    im_file = f"{cellmap_annotator_dir}/{trial}/{trial}.tif"
                    if os.path.isfile(im_file):
                        im = tifffile.imread(im_file)
                        im_cropped = cropper.crop(im, upscale_factor=upscale_factor)
                        tifffile.imwrite(
                            f"{current_output_path}/{output_name}.tif", im_cropped
                        )


def copy_data(
    group: Union[str, list],
    output_path: str,
    crop: Union[str, list] = "all",
    include_nonannotator_results=False,
) -> None:
    """Copies data from all source locations to specified output location.

    Args:
        group (Union[str, list]): Group(s) to copy.
        output_base_path (str): Path to copy data to.
        crop (Union[str, list], optional): Specific crop to copy. Defaults to "all".
        include_nonannotator_results (bool, optional): Whether or not to include predictions, refinements and ariadne. Defaults to False.
    """
    mask_information = MaskInformation(group, crop)
    prediction_paths_df = get_prediction_paths_df()

    for row in mask_information.rows:
        group = row.group
        crop = row.crop
        cropper = Cropper(row.mins, row.maxs)
        current_output_path = f"{output_path}/{group}/{crop}/"
        if os.path.exists(current_output_path):
            shutil.rmtree(current_output_path)
        os.makedirs(current_output_path)

        # do the croopping for annotators
        crop_annotations(group, crop, cropper, current_output_path)

        if include_nonannotator_results:
            # do predictions and refinements
            full_im_dict = get_predictions_and_refinements_from_row(
                row, prediction_paths_df
            )
            for name, im in full_im_dict.items():
                im_cropped = cropper.crop(im)
                tifffile.imwrite(f"{current_output_path}/{name}.tif", im_cropped)

        # do the cropping for ground truth
        with h5py.File(row.gt_path, "r") as f:
            im = f["volumes"]["labels"]["gt"][:]
            im_cropped = cropper.crop(
                im, upscale_factor=row.gt_resolution // row.correct_resolution
            )
            im_cropped = im_cropped.astype(np.uint8)

            # only save it if it is not copy of existing one
            gt_is_unique = True
            for annotation_name in os.listdir(current_output_path):
                if annotation_name not in [
                    "predictions.tif",
                    "ariadne.tif",
                    "refinements.tif",
                ]:
                    annotation_im = tifffile.imread(
                        f"{current_output_path}/{annotation_name}"
                    )
                    if np.array_equal(annotation_im, im_cropped):
                        gt_is_unique = False
                        break

            if gt_is_unique:
                tifffile.imwrite(
                    f"{current_output_path}/gt.tif", im_cropped.astype(np.uint8)
                )
