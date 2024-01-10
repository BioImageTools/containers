import numpy as np
import zarr


def create_dataset(data_path, data_subpath, shape, chunks, dtype,
                   data=None, data_store_name=None,
                   **kwargs):
    try:
        data_store = _get_data_store(data_path, data_store_name)
        if data_subpath:
            print('Create dataset', data_path, data_subpath)
            n5_root = zarr.open_group(store=data_store, mode='a')
            dataset = n5_root.require_dataset(
                data_subpath,
                shape=shape,
                chunks=chunks,
                dtype=dtype,
                data=data)
            # set additional attributes
            dataset.attrs.update(**kwargs)
            return dataset
        else:
            print('Create root array', data_path)
            return zarr.open(store=data_store,
                             shape=shape,
                             chunks=chunks,
                             dtype=dtype,
                              mode='a')
    except Exception as e:
        print('Error creating a dataset at', data_path, data_subpath, e)
        raise e


def open(data_path, data_subpath, data_store_name=None,
         mode='r',
         block_coords=None):
    try:
        data_container = zarr.open(store=_get_data_store(data_path,
                                                         data_store_name),
                                   mode=mode)
        a = data_container[data_subpath] if data_subpath else data_container
        ba = a[block_coords] if block_coords is not None else a
        return ba, a.attrs.asdict()
    except Exception as e:
        print(f'Error opening {data_path} : {data_subpath}', e, flush=True)
        raise e


def _get_data_store(data_path, data_store_name):
    if data_store_name is None or data_store_name == 'n5':
        return zarr.N5Store(data_path)
    else:
        return data_path


def get_voxel_spacing(attrs):
    if (attrs.get('downsamplingFactors')):
        voxel_spacing = (np.array(attrs['pixelResolution']) * 
                         np.array(attrs['downsamplingFactors']))
    else:
        voxel_spacing = np.array(attrs['pixelResolution']['dimensions'])
    return voxel_spacing[::-1] # put in zyx order
