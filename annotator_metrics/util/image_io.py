from typing import Union
import h5py
from numcodecs.gzip import GZip
from annotator_metrics.util.doc_io import MaskInformation, Row
import tifffile
import os
import shutil
import numpy as np
import zarr
import socket
from dask.distributed import Client
import dask
import neuroglancer
from IPython.core.display import display, HTML


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
            chunks=32,
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
            chunks=32,
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
        ds = zarr_root.create_dataset(
            name="raw",
            data=raw[:],
            shape=raw.shape,
            chunks=32,
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
    group: str,
    output_path: str,
    num_workers: int = 10,
    crop: Union[list, str] = None,
) -> None:
    """Create variance images for data.

    Args:
        input_path (str): Path to image data
        group (str): Group to use.
        output_path (str): Path to save images.
        num_workers (int, optional): Number of dask workers. Defaults to 10.
        crop (str, optional): Specific crop to use. Defaults to None, meaning all crops will be used.
    """
    mi = MaskInformation(group, crop)
    with Client(n_workers=num_workers, threads_per_worker=1) as client:
        local_ip = socket.gethostbyname(socket.gethostname())
        url = client.dashboard_link.replace("127.0.0.1", local_ip)
        display(
            HTML(
                f"""<a href="{url}">Click here to montior variance image creation progress.</a>"""
            ),
        )

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
    group: str,
    crop: Union[list, str] = None,
    served_directory: str = "/groups/cellmap/cellmap/",
    server_url: str = "http://10.150.100.248:8080",
) -> None:
    """Provides neuroglancer link to view of data including variance images if available.

    Args:
        n5s_path (str): Path to n5s directory.
        group (str): Group to get view of
        crop (Union[list, str], optional): Crop(s) to get views of. Defaults to None.
        served_directory (str, optional): Directory being served via http. Defaults to "/groups/cellmap/cellmap/".
        server_url (str, optional): Server url. Defaults to "http://10.150.100.248:8080".
    """
    mi = MaskInformation(group, crop)
    for row in mi.rows:
        n5s_path_relative_to_served_directory = (
            "/" + n5s_path.split(served_directory)[-1]
        )
        path = f"n5://{server_url}/{n5s_path_relative_to_served_directory}/{group}/{row.crop}.n5"
        dir_list = os.listdir(f"{n5s_path}/{group}/{row.crop}.n5")
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
        dirs.insert(0, "gt")
        for result_type in ["predictions", "refinements", "ariadne"]:
            if result_type in dir_list:
                dirs.append(result_type)

        variance_images = [d for d in dir_list if "variance" in d]
        variance_images.sort()
        dirs += variance_images
        viewer = neuroglancer.Viewer()
        with viewer.txn() as s:
            zarr_root = zarr.open(f"{n5s_path}/{group}/{row.crop}.n5", mode="r",)

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
                    s.layers[d] = neuroglancer.SegmentationLayer(source=f"{path}/{d}",)
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

        url = neuroglancer.to_url(viewer.state).replace("https://", "http://")
        display(
            HTML(
                f"""<a href="{url}">Click here to view data for {group} and crop {row.crop} on neuroglancer.</a>"""
            ),
        )
        neuroglancer.stop()

        base_path = n5s_path.rsplit("/n5s", 1)[0]
        os.makedirs(f"{base_path}/neuroglancer_links/{group}", exist_ok=True)
        save_neuroglancer_link(
            f"{base_path}/neuroglancer_links/{group}/{row.crop}.html", url
        )

