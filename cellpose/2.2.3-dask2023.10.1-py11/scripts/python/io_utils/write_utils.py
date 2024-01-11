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
        print(f'Persist {data.shape} ({data.dtype}) data ',
              f'as N5 to {container_path} ',
              f'({real_container_path}):{subpath}',
              flush=True)
        attrs = {}
        if resolution is not None:
            attrs['pixelResolution'] = resolution
        if scale_factors is not None:
            attrs['downsamplingFactors'] = scale_factors
        data_store = 'n5'
        zarr_data = zarr_utils.create_dataset(
            container_path,
            subpath,
            data.shape,
            blocksize,
            data.dtype,
            data_store_name='n5',
            **attrs,
        )
        persist_block = functools.partial(_save_block_to_zarr,
                                          zarr_output=zarr_data)
    elif container_ext == '.zarr':
        print(f'Persist data as zarr {container_path} ',
              f'({real_container_path}):{subpath}',
              flush=True)
        attrs = {}
        if resolution is not None:
            attrs['pixelResolution'] = resolution
        if scale_factors is not None:
            attrs['downsamplingFactors'] = scale_factors
        data_store = 'zarr'
        zarr_data = zarr_utils.create_dataset(
            container_path,
            subpath,
            data.shape,
            blocksize,
            data.dtype,
            data_store_name=data_store,
            **attrs,
        )
        persist_block = functools.partial(_save_block_to_zarr,
                                          zarr_output=zarr_data)
    else:
        print(f'Cannot persist data using {container_path} ',
              f'({real_container_path}): {subpath}',
              flush=True)

    if persist_block is not None:
        return save_blocks(data, persist_block, blocksize)
    else:
        return None


def save_blocks(dimage, persist_block, blocksize):
    if blocksize is None:
        chunksize = dimage.chunksize
    else:
        chunksize = blocksize

    if chunksize == dimage.chunksize:
        print(f'No rechunking of {dimage.shape} image',
              f'- using chunksize={chunksize}',
              flush=True)
        rechunked_dimage = dimage
    else:
        # rechunk the image
        print(f'Rechunk {dimage.shape} image from {dimage.chunksize} ',
              f'to {chunksize} before persisting it',
              flush=True)
        rechunked_dimage = da.rechunk(dimage, chunks=chunksize)
    return da.map_blocks(persist_block,
                         rechunked_dimage,
                         chunks=chunksize)


def _save_block_to_nrrd(block, output_dir=None, output_name=None,
                        block_info=None,
                        ext='.nrrd'):
    res_shape = tuple([1 for r in range(0, block.ndim)])
    if block_info is not None:
        output_coords = _block_coords_from_block_info(block_info, 0)
        block_coords = tuple([slice(s.start-s.start, s.stop-s.start)
                              for s in output_coords])

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
              f'output_coords: {output_coords}',
              f'block_coords: {block_coords}',
              flush=True)
        nrrd.write(full_filename, block[block_coords].transpose(2, 1, 0),
                   compression_level=2)
        return np.ones(res_shape, dtype=np.uint32)
    else:
        return np.zeros(res_shape, dtype=np.uint32)


def _save_block_to_tiff(block, output_dir=None, output_name=None,
                        block_info=None,
                        resolution=None,
                        ext='.tif',
                        ):
    res_shape = tuple([1 for r in range(0, block.ndim)])
    if block_info is not None:
        output_coords = _block_coords_from_block_info(block_info, 0)
        block_coords = tuple([slice(s.start-s.start, s.stop-s.start)
                              for s in output_coords])

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
              f'output_coords: {output_coords}',
              f'block_coords: {block_coords}',
              f'to {full_filename}',
              flush=True)
        tiff_metadata = {
            'axes': 'ZYX',
        }
        if resolution is not None:
            tiff_metadata['resolution'] = resolution

        tifffile.imwrite(full_filename, block[block_coords],
                         metadata=tiff_metadata)
        return np.ones(res_shape, dtype=np.uint32)
    else:
        return np.zeros(res_shape, dtype=np.uint32)


def _save_block_to_zarr(block, zarr_output=None, block_info=None):
    res_shape = tuple([1 for r in range(0, block.ndim)])
    if block_info is not None and zarr_output is not None:
        print(f'Save block {block_info}', flush=True)
        output_coords = _block_coords_from_block_info(block_info, 0)
        block_coords = tuple([slice(s.start-s.start, s.stop-s.start)
                              for s in output_coords])
        print(f'Write block {block.shape}',
              f'output_coords: {output_coords}',
              f'block_coords: {block_coords}',
              flush=True)
        zarr_output[output_coords] = block[block_coords]
        return np.ones(res_shape, dtype=np.uint32)
    else:
        return np.zeros(res_shape, dtype=np.uint32)


def _block_coords_from_block_info(block_info, block_index):
    return tuple([slice(c[0],c[1])
                  for c in block_info[block_index]['array-location']])
