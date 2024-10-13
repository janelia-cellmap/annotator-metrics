from contextlib import ExitStack
from typing import Tuple, Union
import h5py
from numcodecs.gzip import GZip
from annotator_metrics.util.doc_util import MaskInformation, Row
import tifffile
import os
import shutil
import numpy as np
import zarr
import socket
from dask.distributed import Client
import dask
import neuroglancer
from annotator_metrics.util.url_util import display_url
import json

class Cropper:
    def __init__(self, mins: np.ndarray, maxs: np.ndarray):
        self.mins = mins
        self.maxs = maxs

    def crop(self, im: np.ndarray, upscale_factor: int = 1) -> np.ndarray:
        if upscale_factor != 1:
            im = (
                im.repeat(upscale_factor, axis=0)
                .repeat(upscale_factor, axis=1)
                .repeat(upscale_factor, axis=2)
            )

        im = im[
            self.mins[2] : self.maxs[2],
            self.mins[1] : self.maxs[1],
            self.mins[0] : self.maxs[0],
        ]
        return im


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


def create_crop_variance_image(
    input_path: str, row: Row, zarr_root: zarr.Group
) -> None:
    image_names = os.listdir(f"{input_path}/{row.group}/{row.crop}")
    images = []

    rescale_factor = row.raw_resolution[0] // row.correct_resolution
    offset_nm = row.correct_resolution * (
        rescale_factor * row.original_coordinates + row.mins
    )
    offset_nm = offset_nm.tolist()

    # segmentations
    for image_name in image_names:
        current_image = tifffile.imread(
            f"{input_path}/{row.group}/{row.crop}/{image_name}"
        )
        if not any(s in image_name for s in ["ariadne", "predictions", "refinements"]):
            images.append(current_image)
        ds = zarr_root.create_dataset(
            name=image_name.split(".")[0],
            data=current_image.astype(np.uint8),
            shape=current_image.shape,
            chunks=64,
            write_empty_chunks=True,
            compressor=GZip(level=6),
        )
        attributes = ds.attrs
        attributes["pixelResolution"] = {
            "dimensions": 3 * [row.correct_resolution],
            "unit": "nm",
        }
        attributes["offset"] = offset_nm

    # variance
    for organelle_name, organelle_label in row.organelle_info.items():
        organelle_images = []
        for image in images:
            if type(organelle_label) == list:
                current_organelle_image = np.zeros(image.shape, dtype=bool)
                for current_organelle_label in organelle_label:
                    current_organelle_image |= image == current_organelle_label
            else:
                current_organelle_image = image == organelle_label
            
            if np.any(current_organelle_image):
                #NOTE: Assume that if something wasnt annotated, it was intentionally left out so don't include it here
                organelle_images.append(current_organelle_image)
        if len(organelle_images) == 0:
            return
        
        if len(organelle_images) == 1:
            organelle_variance = np.zeros(organelle_images[0].shape, dtype=np.float32)
        else:
            organelle_images = np.stack(tuple(organelle_images))
            organelle_variance = np.var(organelle_images, axis=0).astype(np.float32)

        ds = zarr_root.create_dataset(
            name=organelle_name + "_variance",
            data=organelle_variance,
            shape=organelle_variance.shape,
            chunks=64,
            write_empty_chunks=True,
            compressor=GZip(level=6),
        )
        attributes = ds.attrs
        attributes["pixelResolution"] = {
            "dimensions": 3 * [row.correct_resolution],
            "unit": "nm",
        }
        attributes["offset"] = offset_nm


def cleanup_dir(row: Row, output_path: str) -> zarr.Group:
    n5_path = f"{output_path}/{row.group}/{row.crop}.n5"
    shutil.rmtree(f"{n5_path}", ignore_errors=True)
    os.makedirs(f"{n5_path}", exist_ok=True)
    store = zarr.N5Store(n5_path)
    zarr_root = zarr.group(store=store)
    return zarr_root


def get_raw_path_and_dataset_from_row(row: Row) -> Tuple[str, str]:
    raw_split = row.raw_path.split(".n5")
    raw_n5_path = raw_split[0] + ".n5"
    dataset = raw_split[1]
    if os.path.exists(raw_n5_path + "/" + dataset + "/volumes/raw"):
        dataset += "volumes/raw"
    elif os.path.exists(row.raw_path + "/" + dataset + "/s0"):
        dataset += "/s0"
    else:
        dataset += "em/fibsem-uint8/s0"

    return raw_n5_path, dataset


def get_raw_image(row: Row, zarr_root: zarr.Group) -> None:
    raw = None
    try:
        # sometimes this doesn't exist, but also sometimes raw doesn't exist, so may try both
        with h5py.File(row.gt_path) as f:
            raw = f["volumes"]["raw"]
            gt = f["volumes"]["labels"]["gt"]
            # seems like raw and gt resolutions are inaccurate for crops 8-10? so still need to use "correct resolution" in attributes
            raw_resolution = raw.attrs["resolution"][0]
            # gt_resolution = gt.attrs["resolution"][0]

            offset_nm = (
                np.array(gt.attrs["offset"]) + 1
            )  # TODO: Is this right? Add one because otherwise is 1023...

            # in correct resolution voxels
            # scaling_correction = row.gt_resolution // row.correct_resolution
            offset = offset_nm // (raw_resolution // 2)
            # (gt_resolution * scaling_correction)

            # gt_resolution = gt.attrs["offset"][0]
            # raw_resolution = raw.attrs["resolution"][0]

            # training_gt_offset_nm = np.asarray([gt_offset_nm[d]+row.mins[d]*gt_resolution for d in range(3)])
            training_gt_mins = [int(row.mins[d] + offset[d]) for d in range(3)]
            training_gt_maxs = [int(row.maxs[d] + offset[d]) for d in range(3)]

            cropper = Cropper(training_gt_mins, training_gt_maxs)
            # training_gt_dimensions_nm = np.asarray([row.maxs[d]*gt_resolution for d in range(3)])
            # raw_offset = training_gt_offset_nm//raw_resolution

            raw = cropper.crop(raw[:], upscale_factor=2)
    except:
        pass
    
    try:
        raw_n5_path, dataset = get_raw_path_and_dataset_from_row(row)

        zarr_file = zarr.open(raw_n5_path, mode="r")

        crop_start = row.original_coordinates
        crop_end = row.original_coordinates + row.original_crop_size
        raw = zarr_file[dataset][
            crop_start[2] : crop_end[2],
            crop_start[1] : crop_end[1],
            crop_start[0] : crop_end[0],
        ]

        rescale_factor = row.raw_resolution[0] // row.correct_resolution
        if rescale_factor > 1:
            raw = (
                raw.repeat(rescale_factor, axis=0)
                .repeat(rescale_factor, axis=1)
                .repeat(rescale_factor, axis=2)
            )
        cropper = Cropper(row.mins, row.maxs)
        raw = cropper.crop(raw)
    except:
        pass

    if raw is not None:
        ds = zarr_root.create_dataset(
            name="raw",
            data=raw[:],
            shape=raw.shape,
            chunks=64,
            write_empty_chunks=True,
            compressor=GZip(level=6),
        )
        attributes = ds.attrs
        attributes["pixelResolution"] = {
            "dimensions": 3 * [row.correct_resolution],
            "unit": "nm",
        }


def create_variance_images(
    input_path: str,
    group: Union[list, str],
    output_path: str,
    num_workers: int = None,
    crop: Union[list, str] = "all",
) -> None:
    """Create variance images for data.

    Args:
        input_path (str): Path to image data
        group (Union[list,str]): Group to use.
        output_path (str): Path to save images.
        num_workers (int, optional): Number of dask workers. Defaults to None.
        crop (str, optional): Specific crop to use. Defaults to "all".
    """
    mi = MaskInformation(group, crop, input_path)

    # Setup dask client
    with ExitStack() as stack:
        if num_workers:
            stack.enter_context(Client(n_workers=num_workers, threads_per_worker=1))
            client = Client.current()
            local_ip = socket.gethostname()
            url = client.dashboard_link.replace("127.0.0.1", local_ip)
            display_url(url, "Click here to monitor variance image creation progress")

        lazy_results = []
        for row in mi.rows:
            lazy_results.append(dask.delayed(cleanup_dir)(row, output_path))
        zarr_roots = dask.compute(*lazy_results)

        lazy_results = []
        for idx, row in enumerate(mi.rows):
            lazy_results.append(
                dask.delayed(create_crop_variance_image)(
                    input_path, row, zarr_roots[idx]
                )
            )
            lazy_results.append(dask.delayed(get_raw_image)(row, zarr_roots[idx]))

        dask.compute(*lazy_results)


def save_neuroglancer_link(output_path: str, url: str):
    with open(output_path, "w") as f:
        f.write(
            f'<meta http-equiv="refresh" content="0;url={url}">'
            # f'<html><head><meta http-equiv="refresh" content="0; url={url}" /></head><body> </body></html>'
        )


def get_neuroglancer_view(
    n5s_path: str,
    group: Union[list, str],
    crop: Union[list, str] = "all",
    served_directory: str = "/nrs/cellmap/",
    server_url: str = "https://cellmap-vm1.int.janelia.org/nrs/",
) -> None:
    """Provides neuroglancer link to view of data including variance images if available.

    Args:
        n5s_path (str): Path to n5s directory.
        group (Union[list, str]): Group to get view of
        crop (Union[list, str], optional): Crop(s) to get views of. Defaults to "all".
        served_directory (str, optional): Directory being served via http. Defaults to "/groups/cellmap/cellmap/".
        server_url (str, optional): Server url. Defaults to "https://cellmap-vm1.int.janelia.org".
    """

    n5s_path_relative_to_served_directory = "/" + n5s_path.split(served_directory)[-1]
    mi = MaskInformation(
        group, crop, f"{served_directory}/{n5s_path_relative_to_served_directory}"
    )
    for row in mi.rows:
        path = f"n5://{server_url}/{n5s_path_relative_to_served_directory}/{row.group}/{row.crop}.n5"
        dir_list = os.listdir(f"{n5s_path}/{row.group}/{row.crop}.n5")
        dirs = [
            d
            for d in dir_list
            if (
                d
                not in [
                    "gt",
                    "raw",
                    "attributes.json",
                    "predictions",
                    "refinements",
                    "ariadne",
                ]
                and "variance" not in d
            )
        ]
        dirs.sort()
        if "gt" in dir_list:
            dirs.insert(0, "gt")
        for result_type in ["predictions", "refinements", "ariadne"]:
            if result_type in dir_list:
                dirs.append(result_type)

        variance_images = [d for d in dir_list if "variance" in d]
        variance_images.sort()
        dirs += variance_images
        viewer = neuroglancer.Viewer()
        with viewer.txn() as s:

            # raw
            # shaderControls = {
            #     "normalized": {
            #         "range": [np.amin(zarr_root["raw"]), np.amax(zarr_root["raw"])]
            #     }
            # }
            # , shaderControls=shaderControls,)
            raw_resolution = row.raw_resolution[0]
            raw_n5_path, dataset = get_raw_path_and_dataset_from_row(row)
            source = raw_n5_path + "/" + dataset
            source = source.replace(
                "/nrs/cellmap/",
                "n5://https://cellmap-vm1.int.janelia.org/nrs/",
            )
            s.layers[f"raw {dataset}: {row.group}_{row.crop}"] = neuroglancer.ImageLayer(source=source)

            output_dimensions = neuroglancer.CoordinateSpace(
                names=["x", "y", "z"], units="nm", scales=3 * [raw_resolution]
            )

            zarr_root = zarr.open(
                f"{n5s_path}/{row.group}/{row.crop}.n5",
                mode="r",
            )
            for d in dirs:
                _, offset_nm = get_resolution_and_offset_from_zarr(zarr_root[d])
                offset_voxels = offset_nm.astype(np.float) / raw_resolution

                if "variance" not in d:
                    s.layers[d] = neuroglancer.SegmentationLayer(
                        source=[
                            neuroglancer.LayerDataSource(
                                f"{path}/{d}",
                                transform=neuroglancer.CoordinateSpaceTransform(
                                    output_dimensions=output_dimensions,
                                    matrix=[
                                        [1, 0, 0, offset_voxels[0]],
                                        [0, 1, 0, offset_voxels[1]],
                                        [0, 0, 1, offset_voxels[2]],
                                    ],
                                ),
                            )
                        ],
                    )
                else:
                    shader = "#uicontrol invlerp normalized \nvoid main() {\n\temitRGB(vec3(normalized(),0, 0));\n}"
                    shaderControls = {
                        "normalized": {"range": [np.amin(0), np.amax(zarr_root[d])]}
                    }
                    s.layers[d] = neuroglancer.ImageLayer(
                        source=[
                            neuroglancer.LayerDataSource(
                                f"{path}/{d}",
                                transform=neuroglancer.CoordinateSpaceTransform(
                                    output_dimensions=output_dimensions,
                                    matrix=[
                                        [1, 0, 0, offset_voxels[0]],
                                        [0, 1, 0, offset_voxels[1]],
                                        [0, 0, 1, offset_voxels[2]],
                                    ],
                                ),
                            )
                        ],
                        shader=shader,
                        shaderControls=shaderControls,
                    )
                s.layers[d].visible = False
                # print(offset_nm, offset_voxels, row.correct_resolution, raw_resolution)
            s.position = (
                offset_nm + (row.maxs - row.mins) * row.correct_resolution / 2
            ) / raw_resolution
    
        base_path = n5s_path.rsplit("/n5s", 1)[0]
        viewer_state_dir = f"{base_path}/neuroglancer/jsons/{row.group}/"
        os.makedirs(viewer_state_dir, exist_ok=True)
        with open(f"{viewer_state_dir}/{row.crop}.json", "w") as json_file:
            json.dump(viewer.state.to_json(), json_file, indent=4)

        viewer_state_dir = viewer_state_dir.replace("/groups/cellmap/cellmap/", "/dm11/").replace("/nrs/cellmap/", "/nrs/")
        neuroglancer_url = f"https://neuroglancer-demo.appspot.com#!https://cellmap-vm1.int.janelia.org/{viewer_state_dir}/{row.crop}.json"

        display_url(
            neuroglancer_url,
            f"Click here to view data for {row.group} and crop {row.crop} on neuroglancer",
        )
        neuroglancer.stop()

        os.makedirs(f"{base_path}/neuroglancer/links/{row.group}", exist_ok=True)
        save_neuroglancer_link(
            f"{base_path}/neuroglancer/links/{row.group}/{row.crop}.html", neuroglancer_url
        )
