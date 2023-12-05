import nrrd
import numpy as np
import os

from skimage import io

from . import zarr_utils


def open(container_path, subpath):
    real_container_path = os.path.realpath(container_path)
    path_comps = os.path.splitext(container_path)

    container_ext = path_comps[1]

    if container_ext == '.nrrd':
        print(f'Open nrrd {container_path} ({real_container_path})', flush=True)
        return nrrd.read(container_path)
    elif container_ext == '.tif' or container_ext == '.tiff':
        print(f'Open tiff {container_path} ({real_container_path})', flush=True)
        im = io.imread(container_path)
        return im, {}
    elif container_ext == '.npy':
        im = np.load(container_path)
        return im, {}
    elif container_ext == '.n5':
        print(f'Open n5 {container_path} ({real_container_path}):{subpath}', flush=True)
        return zarr_utils.open(container_path, subpath)
    else:
        print(f'Cannot handle {container_path} ({real_container_path}): {subpath}', flush=True)
        return None, {}
