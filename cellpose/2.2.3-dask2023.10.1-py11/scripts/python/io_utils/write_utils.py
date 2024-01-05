import dask.array as da
import functools
import nrrd
import numpy as np
import os
import tifffile

from . import zarr_utils


def save(data, container_path, subpath,
         blocksize=None,
         resolution=None,
         scale_factors=None,
):
    """
    Persist distributed data - typically a dask array to the specified
    container

    Parameters
    ==========
    data - the dask array that needs 
    """
    real_container_path = os.path.realpath(container_path)
    path_comps = os.path.splitext(container_path)

    container_ext = path_comps[1]
    persist_block = None
    if container_ext == '.nrrd':
        print(f'Persist data as nrrd {container_path} ({real_container_path})',
              flush=True)
        output_dir = os.path.dirname(container_path)
        output_name = os.path.basename(path_comps[0])
        persist_block = functools.partial(_save_block_to_nrrd,
                                          output_dir=output_dir,
                                          output_name=output_name,
                                          ext=container_ext)
    elif container_ext == '.tif' or container_ext == '.tiff':
        print(f'Persist data as tiff {container_path} ({real_container_path})',
              flush=True)
        output_dir = os.path.dirname(container_path)
        output_name = os.path.basename(path_comps[0])
        persist_block = functools.partial(_save_block_to_tiff,
                                          output_dir=output_dir,
                                          output_name=output_name,
                                          resolution=resolution,
                                          ext=container_ext)
    elif container_ext == '.n5' or (container_ext == '' and subpath):
        print(f'Persist data as n5 {container_path} ',
              f'({real_container_path}):{subpath}',
              flush=True)
        attrs = {}
        if resolution is not None:
            attrs['pixelResolution'] = resolution
        if scale_factors is not None:
            attrs['downsamplingFactors'] = scale_factors
        output_data = zarr_utils.create_dataset(
            container_path,
            subpath,
            data.shape,
            blocksize,
            data.dtype,
            data_store_name='n5',
            **attrs,
        )
        persist_block = functools.partial(_save_block_to_zarr, output=output_data)
    elif container_ext == '.zarr':
        print(f'Persist data as zarr {container_path} ',
              f'({real_container_path}):{subpath}',
              flush=True)
        attrs = {}
        if resolution is not None:
            attrs['pixelResolution'] = resolution
        if scale_factors is not None:
            attrs['downsamplingFactors'] = scale_factors
        output_data = zarr_utils.create_dataset(
            container_path,
            subpath,
            data.shape,
            blocksize,
            data.dtype,
            data_store_name='zarr',
            **attrs,
        )
        persist_block = functools.partial(_save_block_to_zarr, output=output_data)

    else:
        print(f'Cannot persist data using {container_path} ',
              f'({real_container_path}): {subpath}',
              flush=True)

    if persist_block is not None:
        return save_blocks(data, persist_block, blocksize=blocksize)
    else:
        return None    


def save_blocks(dimage, persist_block, blocksize=None):
    if blocksize is None:
        chunksize = dimage.chunksize
    else:
        chunksize = tuple([d1 if d1 <= d2 else d2
                           for (d1,d2) in zip(blocksize, dimage.shape)])

    if chunksize == dimage.chunksize:
        rechunked_dimage = dimage
    else:
        # rechunk the image
        print(f'Rechunk the image using {chunksize} before persisting it',
              flush=True)
        rechunked_dimage = da.zeros_like(dimage, chunks=blocksize)
        chunksize = blocksize
        rechunked_dimage = da.map_blocks(_copy_blocks_from,
                                         rechunked_dimage,
                                         dtype=rechunked_dimage.dtype,
                                         chunks=chunksize,
                                         meta=np.array(rechunked_dimage.shape),
                                         src=dimage)

    return da.map_blocks(persist_block, rechunked_dimage,
                         # drop all axis - the result of map_blocks is None
                         drop_axis=tuple(range(rechunked_dimage.ndim)),
                         meta=np.array((np.nan)))


def _copy_blocks_from(block, block_info=None, src=None):
    if block_info is not None and src is not None:
        block_coords = _block_coords_from_block_info(block_info)
        return src[block_coords]
    else:
        return block


def _save_block_to_nrrd(block, output_dir=None, output_name=None,
                        block_info=None,
                        ext='.nrrd'):
    if block_info is not None:
        block_coords = _block_coords_from_block_info(block_info)

        saved_blocks_count = np.prod(block_info[None]['num-chunks'])
        if saved_blocks_count > 1:
            filename = (output_name + '-' +
                        '-'.join(map(str, block_info[0]['chunk-location'])) +
                        ext)
        else:
            filename = output_name + ext

        full_filename = os.path.join(output_dir, filename)
        print(f'Write block {block.shape}',
              f'block_info: {block_info}',
              f'block_coords: {block_coords}',
              flush=True)
        nrrd.write(full_filename, block.transpose(2, 1, 0),
                   compression_level=2)


def _save_block_to_tiff(block, output_dir=None, output_name=None,
                        block_info=None,
                        resolution=None,
                        ext='.tif',
                        ):
    if block_info is not None:
        block_coords = _block_coords_from_block_info(block_info)

        saved_blocks_count = np.prod(block_info[None]['num-chunks'])
        if saved_blocks_count > 1:
            filename = (output_name + '-' +
                        '-'.join(map(str, block_info[0]['chunk-location'])) +
                        ext)
        else:
            filename = output_name + ext

        full_filename = os.path.join(output_dir, filename)
        print(f'Write block {block.shape}',
              f'block_info: {block_info}',
              f'block_coords: {block_coords}',
              f'to {full_filename}',
              flush=True)
        tiff_metadata = {
            'axes': 'ZYX',
        }
        if resolution is not None:
            tiff_metadata['resolution'] = resolution

        tifffile.imwrite(full_filename, block, 
                         metadata=tiff_metadata)


def _save_block_to_zarr(block, output=None, block_info=None):
    if block_info is not None:
        block_coords = _block_coords_from_block_info(block_info)
        print(f'Write block {block.shape}',
              f'block_info: {block_info}',
              f'block_coords: {block_coords}',
              flush=True)
        output[block_coords] = block


def _block_coords_from_block_info(block_info):
    return tuple([slice(c[0],c[1]) for c in block_info[0]['array-location']])
