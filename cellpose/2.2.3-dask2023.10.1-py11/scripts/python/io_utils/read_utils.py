import dask.array as da
import nrrd
import numpy as np
import os
import zarr

from tifffile import TiffFile

from . import zarr_utils


def open(container_path, subpath, block_coords=None):
    real_container_path = os.path.realpath(container_path)
    path_comps = os.path.splitext(container_path)

    container_ext = path_comps[1]

    if container_ext == '.nrrd':
        print(f'Open nrrd {container_path} ({real_container_path})', flush=True)
        im = read_nrrd(container_path, block_coords=block_coords)
        return im, {}
    elif container_ext == '.tif' or container_ext == '.tiff':
        print(f'Open tiff {container_path} ({real_container_path})', flush=True)
        im = read_tiff(container_path, block_coords=block_coords)
        return im, {}
    elif container_ext == '.npy':
        im = np.load(container_path)
        return im, {}
    elif container_ext == '.n5':
        print(f'Open n5 {container_path} ({real_container_path}):{subpath}', flush=True)
        return zarr_utils.open(container_path, subpath,
                               block_coords=block_coords)
    else:
        print(f'Cannot handle {container_path} ({real_container_path}): {subpath}', flush=True)
        return None, {}


def read_tiff(input_path, block_coords=None):
    with TiffFile(input_path) as tif:
        tif_store = tif.aszarr()
        tif_array = zarr.open(tif_store)
        if block_coords is None:
            img = tif_array
        else:
            img = tif_array[block_coords]
        return img


def read_nrrd(input_path, block_coords=None):
    im = nrrd.read(input_path)
    return im[block_coords] if block_coords is not None else im
