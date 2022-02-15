import h5py
from annotator_metrics.util.doc_io import MaskInformation
import tifffile
import os
import shutil
import numpy as np
import zarr
import socket
from dask.distributed import Client
import dask


class Cropper:
    def __init__(self, mins, maxs):
        self.mins = tuple(mins)
        self.maxs = tuple(maxs)

    def crop(self, im, upscale_factor=1):
        if upscale_factor != 1:
            im = (
                im.repeat(upscale_factor, axis=0)
                .repeat(upscale_factor, axis=1)
                .repeat(upscale_factor, axis=2)
            )

        im = im[
            self.mins[0] : self.maxs[0],
            self.mins[1] : self.maxs[1],
            self.mins[2] : self.maxs[2],
        ]
        return im


def create_crop_variance_image(input_path, row, zarr_root):
    image_names = os.listdir(f"{input_path}/{row.group}/{row.crop}")
    images = []

    # segmentations
    for image_name in image_names:
        if not any(s in image_name for s in ["ariadne", "predictions", "refinements"]):
            current_image = tifffile.imread(
                f"{input_path}/{row.group}/{row.crop}/{image_name}"
            )
            images.append(current_image)
            ds = zarr_root.create_dataset(
                name=image_name.split(".")[0],
                data=current_image.astype(np.uint8),
                shape=current_image.shape,
                chunks=32,
                write_empty_chunks=True,
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
        )
        attributes = ds.attrs
        attributes["pixelResolution"] = {
            "dimensions": 3 * [row.correct_resolution],
            "unit": "nm",
        }


def cleanup_dir(row, output_path):
    n5_path = f"{output_path}/{row.group}/{row.crop}.n5"
    shutil.rmtree(f"{n5_path}", ignore_errors=True)
    os.makedirs(f"{n5_path}", exist_ok=True)
    store = zarr.N5Store(n5_path)
    zarr_root = zarr.group(store=store)
    return zarr_root


def get_raw_image(row, zarr_root):
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
        )
        attributes = ds.attrs
        attributes["pixelResolution"] = {
            "dimensions": 3 * [row.correct_resolution],
            "unit": "nm",
        }


def create_variance_images(input_path, group, output_path, num_workers=10):
    mi = MaskInformation(group)
    client = Client(n_workers=num_workers, threads_per_worker=1)
    local_ip = socket.gethostbyname(socket.gethostname())
    print(client.dashboard_link.replace("127.0.0.1", local_ip))

    lazy_results = []
    for row in mi.rows:
        lazy_results.append(dask.delayed(cleanup_dir)(row, output_path))
    zarr_roots = dask.compute(*lazy_results)

    lazy_results = []
    for idx, row in enumerate(mi.rows):
        lazy_results.append(
            dask.delayed(create_crop_variance_image)(input_path, row, zarr_roots[idx])
        )
        lazy_results.append(dask.delayed(get_raw_image)(row, zarr_roots[idx]))

    dask.compute(*lazy_results)

