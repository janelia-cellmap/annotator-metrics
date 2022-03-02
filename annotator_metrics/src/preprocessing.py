import os
from typing import Union
import numpy as np
import h5py
import zarr
from ..util.doc_io import MaskInformation, Row
from ..util.image_io import Cropper
import tifffile
import glob
import shutil

predictions_and_segmentations = {
    "group1_03": {
        "data_offset": np.array([0, 0, 0]),
        "resolution": 4,
        "rescale_factor_for_annotation": 2,
        "types": {
            "predictions": {
                "mito": {
                    "path": "/nrs/cellmap/cosem/training/v0003.2/setup03/Macrophage_FS80_Cell2_4x4x4nm/Cryo_FS80_Cell2_4x4x4nm_it1100000.n5",
                    "name": "mito",
                },
                "mito_membrane": {
                    "path": "/nrs/cellmap/cosem/training/v0003.2/setup03/Macrophage_FS80_Cell2_4x4x4nm/Cryo_FS80_Cell2_4x4x4nm_it650000.n5",
                    "name": "mito_membrane",
                },
            },
            "refinements": {
                "mito": {
                    "path": "/groups/cellmap/cellmap/ackermand/paperResultsWithFullPaths/collected/renumbered/Macrophage.n5",
                    "name": "mito_cropped",
                },
                "mito_membrane": {
                    "path": "/groups/cellmap/cellmap/ackermand/paperResultsWithFullPaths/collected/renumbered/Macrophage.n5",
                    "name": "mito_membrane",
                },
            },
        },
        "coordinates": {  # macrophage
            "x": np.array([3840, 3840 + 200]),
            "y": np.array([245, 245 + 200]),
            "z": np.array([7426, 7426 + 200]),
        },
    },
    "group1_09": {
        "data_offset": np.array([30984, 30912, 15728]),
        "resolution": 4,
        "rescale_factor_for_annotation": 1,
        "types": {
            "predictions": {
                "mito": {
                    "path": "/nrs/cellmap/pattonw/training/finetuning/jrc_mus-liver/liver_latest_setup04_many_masked_6-1_100000.n5",
                    "name": "mito",
                },
                "mito_membrane": {
                    "path": "/nrs/cellmap/pattonw/training/finetuning/jrc_mus-liver/liver_latest_setup04_many_masked_6-1_100000.n5",
                    "name": "mito_membrane",
                },
            },
            "refinements": {
                "mito": {
                    "path": "/groups/cellmap/cellmap/ackermand/cosem/jrc_mus-liver.n5/watershedAndAgglomeration/mito.n5",
                    "name": "25_0.975_smoothed_renumbered_filled_renumbered_cropped",
                },
                "mito_membrane": {
                    "path": "/groups/cellmap/cellmap/ackermand/cosem/withFullPaths/training/finetuning/jrc_mus-liver/liver_latest_setup04_many_masked_6-1_100000.n5",
                    "name": "mito_membrane_labeledWith_mito",
                },
            },
            "ariadne": {
                "mito": {
                    "path": "/groups/cellmap/cellmap/bennettd/ariadne/jrc_mus-liver.n5",
                    "name": "mito_instance",
                },
                #'mito_membrane':{'path': '/groups/cosem/cosem/bennettd/ariadne/jrc_mus-liver.n5', 'name':'cristae_instance'}
            },
        },
        "coordinates": {
            "x": np.array([11400, 11400 + 400]),
            "y": np.array([14700, 14700 + 400]),
            "z": np.array([8550, 8550 + 400]),
        },
    },
    "group1_08": {
        "data_offset": np.array([0, 0, 0]),
        "resolution": 4,
        "rescale_factor_for_annotation": 1,
        "types": {
            "predictions": {
                "mito": {
                    "path": "/nrs/cellmap/pattonw/training/finetuning/jrc_mus-liver/group1_08_liver_latest_setup04_many_masked_6-1_100000.n5",
                    "name": "mito",
                },
                "mito_membrane": {
                    "path": "/nrs/cellmap/pattonw/training/finetuning/jrc_mus-liver/group1_08_liver_latest_setup04_many_masked_6-1_100000.n5",
                    "name": "mito_membrane",
                },
            },
        },
        "coordinates": {
            "x": np.array([0, 400]),
            "y": np.array([0, 400]),
            "z": np.array([0, 400]),
        },
    },
    "group1_10": {
        "data_offset": np.array([0, 0, 0]),
        "resolution": 4,
        "rescale_factor_for_annotation": 1,
        "types": {
            "predictions": {
                "mito": {
                    "path": "/nrs/cellmap/pattonw/training/finetuning/jrc_mus-liver/group1_10_liver_latest_setup04_many_masked_6-1_100000.n5",
                    "name": "mito",
                },
                "mito_membrane": {
                    "path": "/nrs/cellmap/pattonw/training/finetuning/jrc_mus-liver/group1_10_liver_latest_setup04_many_masked_6-1_100000.n5",
                    "name": "mito_membrane",
                },
            },
        },
        "coordinates": {
            "x": np.array([0, 400]),
            "y": np.array([0, 400]),
            "z": np.array([0, 400]),
        },
    },
}
labels_dict = {"mito": 4, "mito-mem": 3, "mito-dna": 5, "mito_lumen": 4}
labels_dict_by_group = {
    "group1": {"mito": 4, "mito-mem": 3, "mito-dna": 5},
    "group2": {
        "er": 17,
        "er-mem": 16,
        "eres": 19,
        "eres-mem": 18,
        "golgi": 7,
        "golgi-mem": 6,
    },
    "group3": {
        "pm": 2,
        "endo": 11,
        "endo-mem": 10,
        "vesicle": 9,
        "vesicle-mem": 8,
        "lyso": 13,
        "lyso-mem": 12,
    },
}
# if lumen, remove it put it first


def update_path(path):
    path = path.replace("nrs/cosem/", "nrs/cellmap/")
    path = path.replace("groups/cosem/cosem/", "groups/cellmap/cellmap/")
    return path


def follow_symlinks(path: str):
    path = update_path(path)
    if os.path.islink(path):
        symlink_path = os.readlink(path)
        if symlink_path[0] != "/":  # then it is a relative path
            symlink_path = path.rsplit("/", 1)[0] + "/" + symlink_path
        path = follow_symlinks(symlink_path)
    return path


def update_symlink(path: str):
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


def update_symlinks(path_glob: str):
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


def get_resolution_and_offset_from_zarr(zarr_array):
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
    row: Row, base_path: str = "/groups/cellmap/cellmap/ackermand/forDavis/renumbered/"
):
    # need to figure out labeling
    cell_name = row.raw_path.split("/")[-1].split(".n5")[0]
    base_path = f"{base_path}/{cell_name}/{cell_name}.n5/"

    # convert start and stop from x,y,z to z,y,x for reading from zarr
    # HACK for crop 9 mus liver since there is an additional offset compared to raw data
    # offset = np.zeros((3,), dtype=int)
    # if cell_name == "jrc_mus-liver" and row.crop == "09":
    #    offset = np.array([30984, 30912, 15728], dtype=int)

    # get in terms of 4nm voxels, since all preds/refinements are 4nm
    # crop_start = row.converted_4nm_coordinates - offset // 4
    # crop_end = crop_start + (row.original_crop_size // (4 // row.correct_resolution))

    full_image_dict = {}
    for result_type in ["pred", "seg", "ariadne"]:
        has_segmentation = False
        for organelle_name, organelle_label in labels_dict_by_group[row.group].items():
            if organelle_label in row.organelle_info.values():
                result_path = follow_symlinks(
                    f"{base_path}/{organelle_name}_{result_type}"
                )

                # HACK: special case for crops 8 and 10 which were pulled separately
                if (
                    cell_name == "jrc_mus-liver"
                    and row.crop in ["08", "10"]
                    and row.group == "group1"
                ):
                    result_path = follow_symlinks(
                        f"{base_path}/crop{row.crop}.n5/{organelle_name}_{result_type}"
                    )

                if os.path.isdir(result_path) and not (
                    row.group == "group3"
                    and (
                        row.crop == "03" or row.crop == "06"
                    )  # HACK for another ariadne crop outside the valid range
                ):
                    n5, dataset = result_path.rsplit(".n5/", 1)
                    zarr_file = zarr.open(f"{n5}.n5", mode="r")
                    resolution, offset = get_resolution_and_offset_from_zarr(
                        zarr_file[dataset]
                    )

                    # HACK: for group 1 crop 9, and group3 crop 3, because ariadne was not labeled with offset
                    if (
                        cell_name == "jrc_mus-liver"
                        and ((row.group == "group1" and row.crop == "09"))
                        and result_type == "ariadne"
                    ):
                        offset = np.array([30984, 30912, 15728], dtype=int)

                    crop_start = row.converted_4nm_coordinates - offset // 4
                    crop_end = crop_start + (
                        row.original_crop_size // (4 // row.correct_resolution)
                    )

                    rescale_factor_for_annotation = resolution // row.correct_resolution

                    # need to account for ariadne, with resolution 8 nm. so the crop will be half the size of the 4 nm one
                    scale = int(resolution) // 4
                    if not has_segmentation:
                        combined_image = np.zeros(
                            (crop_end - crop_start)[::-1] // scale, dtype=np.uint8
                        )
                    has_segmentation = True

                    im = zarr_file[dataset][
                        (crop_start[2] // scale) : (crop_end[2] // scale),
                        (crop_start[1] // scale) : (crop_end[1] // scale),
                        (crop_start[0] // scale) : (crop_end[0] // scale),
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


def get_predictions_and_refinements(group_crop: str):
    # we will label mito as 4 so that when we then label membrane, lumen+mito_membrane will be mito
    full_image_dict = {}
    if group_crop not in predictions_and_segmentations:
        return full_image_dict

    crop_results = predictions_and_segmentations[group_crop]
    resolution = crop_results["resolution"]
    offset = crop_results["data_offset"] // resolution
    rescale_factor_for_annotation = crop_results["rescale_factor_for_annotation"]
    coordinates = crop_results["coordinates"]
    x = coordinates["x"] - offset[0]
    y = coordinates["y"] - offset[1]
    z = coordinates["z"] - offset[2]

    # labels mito_mem (3) mito_lumen (4) mito_dna (5)
    for current_type, current_property in crop_results["types"].items():
        combined_image = np.zeros(
            (x[1] - x[0], y[1] - y[0], z[1] - z[0]), dtype=np.uint8
        )

        for organelle_name, organelle_properties in current_property.items():
            organelle_result_path = organelle_properties["path"]
            organelle_result_name = organelle_properties["name"]
            zarr_file = zarr.open(organelle_result_path, mode="r")

            if current_type == "ariadne":
                x_rescaled = x // 2
                y_rescaled = y // 2
                z_rescaled = z // 2
                combined_image = np.zeros(
                    (
                        z_rescaled[1] - z_rescaled[0],
                        y_rescaled[1] - y_rescaled[0],
                        x_rescaled[1] - x_rescaled[0],
                    ),
                    dtype=int,
                )

                zarr_file = zarr_file["multiscale"]["labels"][organelle_result_name][
                    "s0"
                ]

                organelle_results = zarr_file[
                    z_rescaled[0] : z_rescaled[1],
                    y_rescaled[0] : y_rescaled[1],
                    x_rescaled[0] : x_rescaled[1],
                ]

                combined_image[organelle_results >= 1] = labels_dict[
                    organelle_name
                ]  # label their mito as lumen

                # since ariadne is at 8 nm but annotations are at 4
                rescale_factor_for_annotation = 2

            elif current_type == "predictions":
                # using os.path.isdir rather than .array_keys because it is not always recognizing "volumes"
                if os.path.isdir(f"{organelle_result_path}/volumes"):
                    zarr_file = zarr_file["volumes"][organelle_result_name]
                    if os.path.isdir(
                        f"{organelle_result_path}/volumes/{organelle_result_name}/s0"
                    ):
                        zarr_file = zarr_file["s0"]
                else:
                    zarr_file = zarr_file[organelle_result_name]

                organelle_results = zarr_file[z[0] : z[1], y[0] : y[1], x[0] : x[1]]
                combined_image[organelle_results >= 127] = labels_dict[organelle_name]
            else:
                zarr_file = zarr_file[organelle_result_name]
                organelle_results = zarr_file[z[0] : z[1], y[0] : y[1], x[0] : x[1]]
                combined_image[organelle_results >= 1] = labels_dict[organelle_name]

        combined_image = (
            combined_image.repeat(rescale_factor_for_annotation, axis=0)
            .repeat(rescale_factor_for_annotation, axis=1)
            .repeat(rescale_factor_for_annotation, axis=2)
        )
        full_image_dict[current_type] = combined_image

    return full_image_dict


def annotator_cropping(group, crop, cropper, current_output_path):
    input_base_path = "/groups/cellmap/cellmap/annotation_and_analytics/training"
    for annotator_name in os.listdir(input_base_path):
        annotator_dir = f"{input_base_path}/{annotator_name}"
        if os.path.isdir(annotator_dir) and annotator_name not in [
            "conversion_scripts",
            "textfile_templates",
        ]:
            group_dir = f"{annotator_dir}/{group}-labels"
            for trial in os.listdir(group_dir):
                trial_dir = f"{group_dir}/{trial}"
                if os.path.isdir(trial_dir) and f"_{crop}_" in trial:
                    if (
                        trial[-2] == "_"
                    ):  # HACK: for leaving out the number in the dir/file name
                        trial = trial[:-1] + "1" + trial[-1:]

                    im_file = f"{trial_dir}/{trial}.tif"
                    if os.path.isfile(im_file):
                        im = tifffile.imread(f"{trial_dir}/{trial}.tif")
                        output_name = trial.split("_")[-1][::-1]
                        im_cropped = cropper.crop(im)
                        tifffile.imwrite(
                            f"{current_output_path}/{output_name}.tif", im_cropped
                        )

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
                im_cropped = cropper.crop(im)
                tifffile.imwrite(f"{current_output_path}/{output_name}.tif", im_cropped)
        else:
            for trial in os.listdir(cellmap_annotator_dir):
                output_name = trial.split("_")[-1][::-1]
                if output_name[0] in ["a", "b", "f"]:
                    im_file = f"{cellmap_annotator_dir}/{trial}/{trial}.tif"
                    if os.path.isfile(im_file):
                        im = tifffile.imread(im_file)
                        im_cropped = cropper.crop(im)
                        tifffile.imwrite(
                            f"{current_output_path}/{output_name}.tif", im_cropped
                        )


def copy_data(group: Union[str, list], output_base_path: str, crop: str = None):

    mask_information = MaskInformation(group, crop)
    for row in mask_information.rows:
        crop = row.crop
        cropper = Cropper(row.mins, row.maxs)
        current_output_path = f"{output_base_path}/{group}/{crop}/"
        if os.path.exists(current_output_path):
            shutil.rmtree(current_output_path)
        os.makedirs(current_output_path)

        # do the croopping for annotators
        annotator_cropping(group, crop, cropper, current_output_path)

        # do predictions and refinements
        full_im_dict = get_predictions_and_refinements_from_row(row)
        for name, im in full_im_dict.items():
            im_cropped = cropper.crop(im)
            tifffile.imwrite(f"{current_output_path}/{name}.tif", im_cropped)

        # do the cropping for ground truth
        with h5py.File(row.gt_path, "r") as f:
            im = f["volumes"]["labels"]["gt"][:]
            im_cropped = cropper.crop(
                im, upscale_factor=row.gt_resolution // row.correct_resolution
            )
            tifffile.imwrite(
                f"{current_output_path}/gt.tif", im_cropped.astype(np.uint8)
            )

