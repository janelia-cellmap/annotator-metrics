import os
from ..util.doc_io import MaskInformation
from typing import Union
import numpy as np
import h5py
from ..util.image_io import Cropper
import tifffile

predictions_and_segmentations = {
    "macrophage": {
        "offset": np.array([0, 0, 0]),
        "resolution": 4,
        "rescale_factor_for_annotation": 2,
        "result_types": {
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
        "crops": {
            "group1_03": {  # macrophage
                "x": np.array(
                    [3840, 3840 + 200]
                ),  # np.array([3875,3973]),#3840+np.array([71,266])//2,
                "y": np.array(
                    [245, 245 + 200]
                ),  # np.array([331,414]),# 245+np.array([172,339])//2,
                "z": np.array(
                    [7426, 7426 + 200]
                ),  # np.array([7426,7506]),# # 7426+np.array([1,160])//2
            }
        },
    },
    "jrc_mus-liver": {
        "offset": np.array([30984, 30912, 15728]),
        "resolution": 4,
        "rescale_factor_for_annotation": 1,
        "result_types": {
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
        "crops": {
            "group1_09": {
                "x": np.array([11400, 11400 + 400]),
                "y": np.array([14700, 14700 + 400]),
                "z": np.array([8550, 8550 + 400]),
            }
        },
    },
    "jrc_mus-liver2": {
        "offset": np.array([0, 0, 0]),
        "resolution": 4,
        "rescale_factor_for_annotation": 1,
        "result_types": {
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
        "crops": {
            "group1_08": {
                "x": np.array([0, 400]),
                "y": np.array([0, 400]),
                "z": np.array([0, 400]),
            }
        },
    },
    "jrc_mus-liver3": {
        "offset": np.array([0, 0, 0]),
        "resolution": 4,
        "rescale_factor_for_annotation": 1,
        "result_types": {
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
        "crops": {
            "group1_10": {
                "x": np.array([0, 400]),
                "y": np.array([0, 400]),
                "z": np.array([0, 400]),
            }
        },
    },
}
labels_dict = {"mito": 4, "mito_membrane": 3, "mito_lumen": 4, "mito_dna": 5}


# organelle_labels = [all_organelle_labels[i] for i,c in enumerate(row[20:]) if c=="X"]

#             if 3 in organelle_labels or 4 in organelle_labels or 5 in organelle_labels:
#                 organelle_labels.append(0) # This will be used to identify when we need to lable entire mitos

# get ground truth


def copy_data(
    mask_information: MaskInformation,
    group: Union[str, list],
    output_base_path: str,
    input_base_path: str = "/groups/cosem/cosem/annotations/training/",
):
    for row in mask_information.rows:
        if row.group in group:
            crop = row.crop
            cropper = Cropper(row.mins, row.maxs)

            current_input_path = f"{input_base_path}/{group}-labels/{group}_{crop}/"
            current_output_path = f"{output_base_path}/{group}-labels/{group}_{crop}/"
            os.makedirs(current_output_path, exist_ok=True)

            # do the croopping for annotators
            for annotator_directory in os.listdir(current_input_path):
                im = tifffile.imread(
                    f"{current_input_path}/{annotator_directory}/{annotator_directory}.tif"
                )
                im_cropped = cropper.crop(im)
                annotator = annotator_directory.split("_")[-1][::-1]
                tifffile.imwrite(f"{current_output_path}/{annotator}.tif", im_cropped)

            # do predictions

            # do refinements

            # do the cropping for ground truth
            with h5py.File(row.gt_path, "r") as f:
                im = f["volumes"]["labels"]["gt"][:]
                im_cropped = cropper.crop(
                    im, upscale_factor=row.gt_resolution // row.correct_resolution
                )
                tifffile.imwrite(f"{current_output_path}/gt.tif", im_cropped)

