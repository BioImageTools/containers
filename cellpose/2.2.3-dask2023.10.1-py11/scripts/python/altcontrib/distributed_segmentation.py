import dask
import dask.array as da
import functools
import io_utils.read_utils as read_utils
import io_utils.zarr_utils as zarr_utils
import numpy as np
import scipy
import sys
import time
import traceback

from cellpose.models import get_user_models
from dask.distributed import as_completed, Semaphore
from sklearn import metrics as sk_metrics


def distributed_segmentation(
        image_path,
        image_subpath,
        image_shape,
        image_ndim,
        model_type,
        diameter,
        blocksize,
        output_dir,
        blocksoverlap=(),
        mask=None,
        use_torch=False,
        use_gpu=False,
        device=None,
        anisotropy=None,
        z_axis=0,
        do_3D=True,
        use_net_avg=False,
        eval_channels=None,
        min_size=15,
        resample=True,
        flow_threshold=0.4,
        cellprob_threshold=0,
        stitch_threshold=0,
        max_tasks=-1,
        gpu_batch_size=8,
        iou_depth=1,
        iou_threshold=0.7,
        persist_labeled_blocks=False,
        client=None
):
    start_time = time.time()
    if diameter <= 0:
        # always specify the diameter
        diameter = 30
    blocksoverlap = (blocksoverlap if (blocksoverlap is not None and 
                                       len(blocksoverlap) == image_ndim)
                                   else (diameter * 2,) * image_ndim)

    blockchunks = np.array(blocksize, dtype=int)
    blockoverlaps = np.array(blocksoverlap, dtype=int)

    # extra check in case blocksize and diameter are very close
    for ax in range(len(blockchunks)):
        if blockoverlaps[ax] > blockchunks[ax] / 2:
            blockoverlaps[ax] = int(blockchunks[ax] / 2)

    nblocks = np.ceil(np.array(image_shape) / blockchunks).astype(int)
    print(f'Blocksize:{blockchunks}, '
          f'overlap:{blockoverlaps} => {nblocks} blocks',
          flush=True)

    blocks_info = []
    for (i, j, k) in np.ndindex(*nblocks):
        start = blockchunks * (i, j, k) - blockoverlaps
        stop = start + blockchunks + 2*blockoverlaps
        start = np.maximum(0, start)
        stop = np.minimum(image_shape, stop)
        blockslice = tuple(slice(x, y) for x, y in zip(start, stop))
        if _is_not_masked(mask, image_shape, blockslice):
            blocks_info.append(((i, j, k), blockslice))

    if max_tasks > 0:
        print(f'Limit segmentation tasks to {max_tasks}', flush=True)
        tasks_semaphore = Semaphore(max_leases=max_tasks,
                                    name='CellposeLimiter')
        eval_model_method = _throttle(_eval_model, tasks_semaphore)
    else:
        eval_model_method = _eval_model

    eval_block = functools.partial(
        eval_model_method,
        model_type=model_type,
        eval_channels=eval_channels,
        use_net_avg=use_net_avg,
        do_3D=do_3D,
        z_axis=z_axis,
        diameter=diameter,
        anisotropy=anisotropy,
        min_size=min_size,
        resample=resample,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        stitch_threshold=stitch_threshold,
        use_torch=use_torch,
        use_gpu=use_gpu,
        device=device,
        gpu_batch_size=gpu_batch_size,
    )

    print('Cache cellpose models', model_type, flush=True)
    get_user_models()

    segment_block = functools.partial(
        _segment_block,
        eval_block,
        image_path=image_path,
        image_subpath=image_subpath,
        blocksize=blockchunks,
        blockoverlaps=blockoverlaps,
    )

    print(f'{time.ctime(start_time)} ',
          f'Start segmenting: {len(blocks_info)} blocks',
          flush=True)
    segment_block_res = client.map(
        segment_block,
        blocks_info,
    )

    labeled_blocks, labeled_blocks_info, max_label = _collect_labeled_blocks(
        segment_block_res,
        image_shape,
        blocksize,
        output_dir=output_dir,
        persist_labeled_blocks=persist_labeled_blocks,
    )

    segmentation_complete_time = time.time()
    print(f'{time.ctime(segmentation_complete_time)} ',
          f'Finished segmentation of {len(blocks_info)} blocks',
          f'in {segmentation_complete_time-start_time}s',
          flush=True)

    if np.prod(nblocks) > 1:
        working_labeled_blocks = labeled_blocks
        print(f'Submit link labels for {nblocks} label blocks', flush=True)
        labeled_blocks_coords = [bi[1] for bi in labeled_blocks_info]
        new_labeling = _link_labels(
            working_labeled_blocks,
            labeled_blocks_coords,
            max_label,
            iou_depth,
            iou_threshold,
            client
        )
        # save labels to a temporary file for the relabeling process
        labels_filename = f'{output_dir}/labels.npy'
        saved_labels_filename = dask.delayed(_save_labels)(
            new_labeling,
            labels_filename,
        )
        print(f'Relabel {nblocks} blocks', flush=True)
        relabeled = da.map_blocks(
            _relabel_block,
            labeled_blocks,
            labels_filename=saved_labels_filename,
            dtype=labeled_blocks.dtype,
            chunks=labeled_blocks.chunks)
    else:
        print('There is only one block - no link labels is needed',
              flush=True)
        relabeled = labeled_blocks
    return relabeled


def _is_not_masked(mask, image_shape, blockslice):
    if mask is None:
        return True

    mask_to_image_ratio = np.array(mask.shape) / image_shape
    mask_start = np.floor([s.start for s in blockslice] * 
                           mask_to_image_ratio).astype(int)
    mask_stop = np.ceil([s.stop for s in blockslice] * 
                         mask_to_image_ratio).astype(int)
    mask_crop = mask[tuple(slice(a, b) for a, b in zip(mask_start, mask_stop))]
    if np.any(mask[mask_crop]):
        return True
    else:
        return False


def _read_block_data(block_info, image_path, image_subpath=None):
    block_index, block_coords = block_info
    print(f'{time.ctime(time.time())} '
        f'Get block: {block_index}, from: {block_coords}',
        flush=True)
    block_data, _ = read_utils.open(image_path, image_subpath,
                                    block_coords=block_coords)
    print(f'{time.ctime(time.time())} '
        f'Retrieved block {block_index} of shape {block_data.shape}',
        flush=True)
    return block_data


def _throttle(m, sem):
    def throttled_m(*args, **kwargs):
        with sem:
            print(f'{time.ctime(time.time())} Secured slot to run {m}',
                  flush=True)
            try:
                return m(*args, **kwargs)
            finally:
                print(f'{time.ctime(time.time())} Release slot used for {m}',
                      flush=True)

    return throttled_m


def _segment_block(eval_method,
                   block_info,
                   image_path=None,
                   image_subpath=None,
                   blocksize=None,
                   blockoverlaps=None,
                   preprocessing_steps=[],
):
    block_index, block_coords = block_info
    start_time = time.time()
    print(f'{time.ctime(start_time)} ',
          f'Segment block: {block_index}, {block_coords}',
          flush=True)
    block_shape = tuple([sl.stop-sl.start for sl in block_coords])

    block_data = _read_block_data(block_info, image_path,
                                  image_subpath=image_subpath)
    # preprocess
    for pp_step in preprocessing_steps:
        block_data = pp_step[0](block_data, **pp_step[1])

    labels = eval_method(block_index, block_data)

    max_label = np.max(labels)

    # remove overlaps
    new_block_coords = list(block_coords)
    for axis in range(block_data.ndim):
        # left side
        if block_coords[axis].start != 0:
            slc = [slice(None),]*block_data.ndim
            slc[axis] = slice(blockoverlaps[axis], None)
            labels = labels[tuple(slc)]
            a, b = block_coords[axis].start, block_coords[axis].stop
            new_block_coords[axis] = slice(a + blockoverlaps[axis], b)

        # right side
        if block_shape[axis] > blocksize[axis]:
            slc = [slice(None),]*block_data.ndim
            slc[axis] = slice(None, blocksize[axis])
            labels = labels[tuple(slc)]
            a = new_block_coords[axis].start
            new_block_coords[axis] = slice(a, a + blocksize[axis])

    end_time = time.time()
    print(f'{time.ctime(start_time)} ',
          f'Finished segmenting block {block_index} ',
          f'in {end_time-start_time}s',
          flush=True)
    return block_index, tuple(new_block_coords), max_label, labels


def _eval_model(block_index,
                block_data,
                model_type='cyto',
                eval_channels=None,
                use_net_avg=False,
                do_3D=True,
                z_axis=0,
                diameter=None,
                anisotropy=None,
                min_size=15,
                resample=True,
                flow_threshold=0.4,
                cellprob_threshold=0,
                stitch_threshold=0,
                use_torch=False,
                use_gpu=False,
                device=None,
                gpu_batch_size=8,
):
    from cellpose import models

    print(f'{time.ctime(time.time())}',
          f'Run model eval for block: {block_index}, size: {block_data.shape}',
          f'3-D:{do_3D}, diameter:{diameter}',
          flush=True)

    np.random.seed(block_index)

    segmentation_device, gpu = models.assign_device(use_torch=use_torch,
                                                    gpu=use_gpu,
                                                    device=device)

    model = models.Cellpose(gpu=gpu,
                            model_type=model_type,
                            net_avg=use_net_avg,
                            device=segmentation_device)
    labels, _, _, _ = model.eval(block_data,
                                 channels=eval_channels,
                                 diameter=diameter,
                                 z_axis=z_axis,
                                 do_3D=do_3D,
                                 min_size=min_size,
                                 resample=resample,
                                 net_avg=use_net_avg,
                                 anisotropy=anisotropy,
                                 flow_threshold=flow_threshold,
                                 cellprob_threshold=cellprob_threshold,
                                 stitch_threshold=stitch_threshold,
                                 batch_size=gpu_batch_size,
                                )
    print(f'{time.ctime(time.time())}',
          f'Finished model eval for block: {block_index}',
          flush=True)

    return labels.astype(np.uint32)


def _collect_labeled_blocks(segment_blocks_res, shape, chunksize,
                            output_dir=None,
                            persist_labeled_blocks=False):
    """
    Collect segmentation results.

    Parameters
    ==========
    segment_blocks_res: block segmentation results
    shape: shape of a full image being segmented
    chunksize: result array chunksize

    Returns
    =======
    labels: dask array created from segmentation results

    """
    print(f'{time.ctime(time.time())}',
          'Begin collecting labeled blocks (blocksize={chunksize})',
          flush=True)
    labeled_blocks_info = []
    if output_dir is not None and persist_labeled_blocks:
        # collect labels in a zarr array
        zarr_labels_container = f'{output_dir}/labeled_blocks.zarr'
        print(f'Save labels to temporary zarr {zarr_labels_container}',
              flush=True)
        labels = zarr_utils.create_dataset(
            zarr_labels_container,
            '',
            shape,
            chunksize,
            np.uint32,
            data_store_name='zarr',
        )
        is_zarr = True
    else:
        labels = da.empty(shape, dtype=np.uint32, chunks=chunksize)
        is_zarr = False
    result_index = 0
    max_label = 0
    # collect segmentation results
    completed_segment_blocks = as_completed(segment_blocks_res, with_results=True)
    for f,r in completed_segment_blocks:
        print(f'Process future {f}', file=sys.stderr, flush=True)
        if f.cancelled():
            exc = f.exception()
            print('Segment block future exception:', exc,
                  file=sys.stderr,
                  flush=True)
            tb = f.traceback()
            traceback.print_tb(tb)

        (block_index, block_coords, max_block_label, block_labels) = r
        block_shape = tuple([sl.stop-sl.start for sl in block_coords])

        print(f'{result_index+1}. ',
            f'Write labels {block_index},{block_coords} ',
            f'block shape: {block_shape} ?==? {block_labels.shape}',
            f'max block label: {max_block_label}',
            f'block labels range: {max_label} - {max_label+max_block_label}',
            flush=True)

        block_labels_offsets = np.where(block_labels > 0,
                                        max_label,
                                        np.uint32(0)).astype(np.uint32)
        block_labels += block_labels_offsets
        # set the block in the dask array of labeled blocks
        labels[block_coords] = block_labels
        # block_index, block_coords, labels_range
        labeled_blocks_info.append((block_index,
                                    block_coords,
                                    (max_label, max_label+max_block_label)))
        max_label = max_label + max_block_label
        result_index += 1

    print(f'{time.ctime(time.time())}',
          f'Finished collecting labels in {shape} image',
          flush=True)
    labels_res = da.from_zarr(labels) if is_zarr else labels
    return labels_res, labeled_blocks_info, max_label


def _link_labels(labels, blocks_coords, max_label, face_depth, iou_threshold,
                 client):
    label_groups = _get_adjacent_label_mappings(labels,
                                                blocks_coords,
                                                face_depth,
                                                iou_threshold,
                                                client)
    print(f'{time.ctime(time.time())}',
          'Find connected components for label groups',
          flush=True)
    return dask.delayed(_get_labels_connected_comps)(label_groups, max_label+1)


def _get_adjacent_label_mappings(labels, blocks_coords, block_face_depth,
                                 iou_threshold,
                                 client):
    print(f'{time.ctime(time.time())}',
          'Create adjacency graph for', labels,
          flush=True)
    blocks_faces_and_axes = _get_blocks_faces_info(blocks_coords,
                                                   block_face_depth,
                                                   labels)
    print(f'{time.ctime(time.time())}',
          f'Invoke label mapping for {len(blocks_faces_and_axes)} faces',
          flush=True)
    mapped_labels = client.map(
        _across_block_label_grouping,
        blocks_faces_and_axes,
        iou_threshold=iou_threshold,
        image=labels
    )
    print(f'{time.ctime(time.time())}',
          f'Start collecting label mappings',
          flush=True)
    all_mappings = [ da.empty((2, 0), dtype=labels.dtype, chunks=1) ]
    completed_mapped_labels = as_completed(mapped_labels, with_results=True)
    for _,mapped in completed_mapped_labels:
        print('Append mapping:', mapped, flush=True)
        all_mappings.append(mapped)

    mappings = da.concatenate(all_mappings, axis=1)
    print(f'{time.ctime(time.time())}',
          f'Concatenated {len(all_mappings)} mappings ->',
          f'{mappings.shape}',
          flush=True)
    return mappings


def _get_blocks_faces_info(blocks_coords, face_depth, image):
    ndim = image.ndim
    depth = da.overlap.coerce_depth(ndim, face_depth)
    face_slices_and_axes = []
    for ax in range(ndim):
        for sl in blocks_coords:
            if sl[ax].stop == image.shape[ax]:
                # this is an end block on this axis
                continue
            slice_to_append = list(sl)
            slice_to_append[ax] = slice(
                sl[ax].stop - depth[ax], sl[ax].stop + depth[ax]
            )
            face_slice_coords = tuple(slice_to_append)
            face_slices_and_axes.append((face_slice_coords, ax))
    return face_slices_and_axes


def _across_block_label_grouping(face_info, iou_threshold=0, image=None):
    face_slice, axis = face_info
    face_shape = tuple([s.stop-s.start for s in face_slice])
    print(f'Group labels for face {face_slice} {face_shape} ',
            f'along {axis} axis',
            flush=True)
    face = image[face_slice].compute()
    print(f'Get label grouping accross axis {axis} from {face.shape} image',
          flush=True)
    unique = np.unique(face)
    face0, face1 = np.split(face, 2, axis)

    intersection = sk_metrics.confusion_matrix(face0.reshape(-1), face1.reshape(-1))
    sum0 = intersection.sum(axis=0, keepdims=True)
    sum1 = intersection.sum(axis=1, keepdims=True)

    # Note that sum0 and sum1 broadcast to square matrix size.
    union = sum0 + sum1 - intersection

    # Ignore errors with divide by zero, which the np.where sets to zero.
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(intersection > 0, intersection / union, 0).astype(np.uint32)

    labels0, labels1 = np.nonzero(iou >= iou_threshold)
    print(f'labels0 ({labels0.shape}):', labels0, flush=True)
    print(f'labels1 ({labels1.shape}):', labels1, flush=True)

    labels0_orig = unique[labels0]
    labels1_orig = unique[labels1]
    print(f'unique labels0 ({labels0_orig.shape}):', labels0_orig, flush=True)
    print(f'unique labels1 ({labels1_orig.shape}):', labels1_orig, flush=True)
    grouped = np.stack([labels0_orig, labels1_orig])

    print(f'Current labels ({grouped.shape}):', grouped, flush=True)
    # Discard any mappings with bg pixels
    valid = np.all(grouped != 0, axis=0).astype(np.uint32)
    print(f'Valid labels ({valid.shape}):', valid, flush=True)
    # if there's not more than one label return it as is
    return (grouped[:, valid] if grouped.size > 2 else grouped)


def _get_labels_connected_comps(label_groups, nlabels):
    # reformat label mappings as csr_matrix
    csr_label_groups = _mappings_as_csr(label_groups, nlabels+1)

    connected_comps = scipy.sparse.csgraph.connected_components(
        csr_label_groups,
        directed=False,
    )[1]
    return connected_comps


def _mappings_as_csr(lmapping, n):
    print(f'Generate csr matrix for {lmapping.shape} labels', flush=True)
    l0 = lmapping[0, :]
    l1 = lmapping[0, :]
    v = np.ones_like(l0)
    mat = scipy.sparse.coo_matrix((v, (l0, l1)), shape=(n, n))
    csr_mat = mat.tocsr()
    return csr_mat


def _save_labels(l, lfilename):
    np.save(lfilename, l)
    return lfilename


def _relabel_block(block,
                   labels_filename=None,
                   block_info=None):
    if block_info is not None and labels_filename is not None:
        print(f'Relabeling block {block_info[0]}', flush=True)
        labels = np.load(labels_filename)
        relabeled_block = labels[block]
        return relabeled_block
    else:
        return block
