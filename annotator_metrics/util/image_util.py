from contextlib import ExitStack
from typing import Union
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


def create_crop_variance_image(
    input_path: str, row: Row, zarr_root: zarr.Group
) -> None:
    image_names = os.listdir(f"{input_path}/{row.group}/{row.crop}")
    images = []

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
            organelle_images.append(current_organelle_image)

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


def cleanup_dir(row: Row, output_path: str) -> zarr.Group:
    n5_path = f"{output_path}/{row.group}/{row.crop}.n5"
    shutil.rmtree(f"{n5_path}", ignore_errors=True)
    os.makedirs(f"{n5_path}", exist_ok=True)
    store = zarr.N5Store(n5_path)
    zarr_root = zarr.group(store=store)
    return zarr_root


def get_raw_image(row: Row, zarr_root: zarr.Group) -> None:
    raw_split = row.raw_path.split(".n5")
    raw_n5_path = raw_split[0] + ".n5"
    dataset = raw_split[1]
    if os.path.exists(raw_n5_path + "/" + dataset + "/volumes/raw"):
        dataset += "volumes/raw"
    if os.path.exists(row.raw_path + "/" + dataset + "/s0"):
        dataset += "/s0"

    zarr_file = zarr.open(raw_n5_path, mode="r")

    crop_start = row.original_coordinates
    crop_end = row.original_coordinates + row.original_crop_size
    raw = zarr_file[dataset][
        crop_start[2] : crop_end[2],
        crop_start[1] : crop_end[1],
        crop_start[0] : crop_end[0],
    ]

    # # crop based on 4 nm coordinates
    # # raw resolution is always equal or lower res than correct gt
    # crop_start = row.converted_4nm_coordinates
    # crop_end = crop_start + (row.original_crop_size // (4 // row.correct_resolution))

    # scale = row.raw_resolution[0] / 4
    # if scale > 1:
    #     crop_start_padded = crop_start // scale
    #     crop_end_padded = -1 * (-crop_end // scale)

    #     raw = zarr_file[dataset][
    #         crop_start_padded[2] : crop_end_padded[2],
    #         crop_start_padded[1] : crop_end_padded[1],
    #         crop_start_padded[0] : crop_end_padded[0],
    #     ]

    #     adjusted_start = crop_start - crop_start_padded * scale
    #     adjusted_end = adjusted_start + (crop_end - crop_start)
    #     cropper = Cropper(adjusted_start, adjusted_end)
    #     raw = cropper.crop(raw, scale)
    # elif scale <= 1:
    #     if scale < 1:
    #         # assume if raw_resolution < 4, then it is correct resolution
    #         crop_start = np.ndarray.astype(crop_start / scale, np.int)
    #         crop_end = np.ndarray.astype(crop_end / scale, np.int)

    #     raw = zarr_file[dataset][
    #         crop_start[2] : crop_end[2],
    #         crop_start[1] : crop_end[1],
    #         crop_start[0] : crop_end[0],
    #     ]

    # rescale to correct resolution
    rescale_factor = row.raw_resolution[0] // row.correct_resolution
    if rescale_factor > 1:
        raw = (
            raw.repeat(rescale_factor, axis=0)
            .repeat(rescale_factor, axis=1)
            .repeat(rescale_factor, axis=2)
        )
    cropper = Cropper(row.mins, row.maxs)
    raw = cropper.crop(raw)

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
            local_ip = socket.gethostbyname(socket.gethostname())
            url = client.dashboard_link.replace("127.0.0.1", local_ip)
            display_url(url, "Click here to montior variance image creation progress")

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
    served_directory: str = "/groups/cellmap/cellmap/",
    server_url: str = "https://cellmap-vm1.int.janelia.org/dm11/",
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
            zarr_root = zarr.open(
                f"{n5s_path}/{row.group}/{row.crop}.n5",
                mode="r",
            )

            # raw
            shaderControls = {
                "normalized": {
                    "range": [np.amin(zarr_root["raw"]), np.amax(zarr_root["raw"])]
                }
            }
            s.layers["raw"] = neuroglancer.ImageLayer(
                source=f"{path}/raw", shaderControls=shaderControls
            )

            for d in dirs:
                if "variance" not in d:
                    s.layers[d] = neuroglancer.SegmentationLayer(
                        source=f"{path}/{d}",
                    )
                else:
                    shader = "#uicontrol invlerp normalized \nvoid main() {\n\temitRGB(vec3(normalized(),0, 0));\n}"
                    shaderControls = {
                        "normalized": {"range": [np.amin(0), np.amax(zarr_root[d])]}
                    }
                    s.layers[d] = neuroglancer.ImageLayer(
                        source=f"{path}/{d}",
                        shader=shader,
                        shaderControls=shaderControls,
                    )
                s.layers[d].visible = False

        url = neuroglancer.to_url(viewer.state)
        display_url(
            url,
            f"Click here to view data for {row.group} and crop {row.crop} on neuroglancer",
        )
        neuroglancer.stop()

        base_path = n5s_path.rsplit("/n5s", 1)[0]
        os.makedirs(f"{base_path}/neuroglancer_links/{row.group}", exist_ok=True)
        save_neuroglancer_link(
            f"{base_path}/neuroglancer_links/{row.group}/{row.crop}.html", url
        )
