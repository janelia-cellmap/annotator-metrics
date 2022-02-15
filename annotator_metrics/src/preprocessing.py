import os
from typing import Union
import numpy as np
import h5py
import zarr
from ..util.doc_io import MaskInformation
from ..util.image_io import Cropper
import tifffile

predictions_and_segmentations = {
    "group1_03": {
        "data_offset": np.array([0, 0, 0]),
        "resolution": 4,
        "rescale_factor_for_annotation": 2,
        "types": {
            "predictions": {
                "mito": {
                    "path": "/nrs/cosem/cosem/training/v0003.2/setup03/Macrophage_FS80_Cell2_4x4x4nm/Cryo_FS80_Cell2_4x4x4nm_it1100000.n5",
                    "name": "mito",
                },
                "mito_membrane": {
                    "path": "/nrs/cosem/cosem/training/v0003.2/setup03/Macrophage_FS80_Cell2_4x4x4nm/Cryo_FS80_Cell2_4x4x4nm_it650000.n5",
                    "name": "mito_membrane",
                },
            },
            "refinements": {
                "mito": {
                    "path": "/groups/cosem/cosem/ackermand/paperResultsWithFullPaths/collected/renumbered/Macrophage.n5",
                    "name": "mito_cropped",
                },
                "mito_membrane": {
                    "path": "/groups/cosem/cosem/ackermand/paperResultsWithFullPaths/collected/renumbered/Macrophage.n5",
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
                    "path": "/nrs/cosem/pattonw/training/finetuning/jrc_mus-liver/liver_latest_setup04_many_masked_6-1_100000.n5",
                    "name": "mito",
                },
                "mito_membrane": {
                    "path": "/nrs/cosem/pattonw/training/finetuning/jrc_mus-liver/liver_latest_setup04_many_masked_6-1_100000.n5",
                    "name": "mito_membrane",
                },
            },
            "refinements": {
                "mito": {
                    "path": "/groups/cosem/cosem/ackermand/cosem/jrc_mus-liver.n5/watershedAndAgglomeration/mito.n5",
                    "name": "25_0.975_smoothed_renumbered_filled_renumbered_cropped",
                },
                "mito_membrane": {
                    "path": "/groups/cosem/cosem/ackermand/cosem/withFullPaths/training/finetuning/jrc_mus-liver/liver_latest_setup04_many_masked_6-1_100000.n5",
                    "name": "mito_membrane_labeledWith_mito",
                },
            },
            "ariadne": {
                "mito": {
                    "path": "/groups/cosem/cosem/bennettd/ariadne/jrc_mus-liver.n5",
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
                    "path": "/nrs/cosem/pattonw/training/finetuning/jrc_mus-liver/group1_08_liver_latest_setup04_many_masked_6-1_100000.n5",
                    "name": "mito",
                },
                "mito_membrane": {
                    "path": "/nrs/cosem/pattonw/training/finetuning/jrc_mus-liver/group1_08_liver_latest_setup04_many_masked_6-1_100000.n5",
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
                    "path": "/nrs/cosem/pattonw/training/finetuning/jrc_mus-liver/group1_10_liver_latest_setup04_many_masked_6-1_100000.n5",
                    "name": "mito",
                },
                "mito_membrane": {
                    "path": "/nrs/cosem/pattonw/training/finetuning/jrc_mus-liver/group1_10_liver_latest_setup04_many_masked_6-1_100000.n5",
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
labels_dict = {"mito": 4, "mito_membrane": 3, "mito_lumen": 4, "mito_dna": 5}


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

        for organelle_name, orgaenelle_properties in current_property.items():
            organelle_result_path = orgaenelle_properties["path"]
            organelle_result_name = orgaenelle_properties["name"]
            zarr_file = zarr.open(organelle_result_path, mode="r")

            if current_type == "ariadne":
                x_rescaled = x // 2
                y_rescaled = y // 2
                z_rescaled = z // 2
                combined_image = np.zeros(
                    (
                        x_rescaled[1] - x_rescaled[0],
                        y_rescaled[1] - y_rescaled[0],
                        z_rescaled[1] - z_rescaled[0],
                    ),
                    dtype=np.uint8,
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


def copy_data(
    group: Union[str, list],
    output_base_path: str,
    input_base_path: str = "/groups/cosem/cosem/annotations/training/",
):
    mask_information = MaskInformation(group)
    for row in mask_information.rows:
        crop = row.crop
        cropper = Cropper(row.mins, row.maxs)

        current_input_path = f"{input_base_path}/{group}-labels/{group}_{crop}/"
        current_output_path = f"{output_base_path}/{group}/{crop}/"
        os.makedirs(current_output_path, exist_ok=True)

        # do the croopping for annotators
        for annotator_directory in os.listdir(current_input_path):
            im = tifffile.imread(
                f"{current_input_path}/{annotator_directory}/{annotator_directory}.tif"
            )
            im_cropped = cropper.crop(im)
            annotator = annotator_directory.split("_")[-1][::-1]
            tifffile.imwrite(f"{current_output_path}/{annotator}.tif", im_cropped)

        # do predictions and refinements
        full_im_dict = get_predictions_and_refinements(f"{group}_{crop}")
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

