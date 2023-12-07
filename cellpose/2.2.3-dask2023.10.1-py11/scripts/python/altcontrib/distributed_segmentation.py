import dask
import dask.array as da
import functools
import numpy as np
import scipy
import time

from dask.distributed import Semaphore
from sklearn import metrics as sk_metrics


def distributed_segmentation(
        image,
        model_type,
        diameter,
        blocksize=None,
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
):
    if diameter <= 0:
        # always specify the diameter
        diameter = 30

    blocksize = (blocksize if (blocksize is not None and 
                               len(blocksize) == image.ndim)
                           else (128,) * image.ndim)

    blocksoverlap = (blocksoverlap if (blocksoverlap is not None and 
                                       len(blocksoverlap) == image.ndim)
                                   else (diameter * 2,) * image.ndim)

    blockchunks = np.array(blocksize, dtype=int)
    blockoverlaps = np.array(blocksoverlap, dtype=int)

    # extra check in case blocksize and diameter are very close
    for ax in range(len(blockchunks)):
        if blockoverlaps[ax] > blockchunks[ax] / 2:
            blockoverlaps[ax] = int(blockchunks[ax] / 2)

    nblocks = np.ceil(np.array(image.shape) / blockchunks).astype(int)
    print(f'Blocksize:{blockchunks}, '
          f'overlap:{blockoverlaps} => {nblocks} blocks',
          flush=True)

    blocks_info = []
    for (i, j, k) in np.ndindex(*nblocks):
        start = blockchunks * (i, j, k) - blockoverlaps
        stop = start + blockchunks + 2*blockoverlaps
        start = np.maximum(0, start)
        stop = np.minimum(image.shape, stop)
        blockslice = tuple(slice(x, y) for x, y in zip(start, stop))
        if _is_not_masked(mask, image.shape, blockslice):
            blocks_info.append(((i, j, k), blockslice))

    get_image_block = functools.partial(_get_block_data, image)

    image_blocks = map(
        get_image_block,
        blocks_info,
    )

    if max_tasks > 0:
        print(f'Limit segmentation tasks to {max_tasks}', flush=True)
        tasks_semaphore = Semaphore(max_leases=max_tasks,
                                    name='CellposeLimiter')
        eval_model_method = _throttle(_eval_model, tasks_semaphore)
    else:
        eval_model_method = _eval_model

    eval_block = functools.partial(
        dask.delayed(eval_model_method),
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

    segment_block = functools.partial(
        _segment_block,
        eval_block,
        blocksize=blockchunks,
        blockoverlaps=blockoverlaps,
    )

    segment_block_res = map(
        segment_block,
        image_blocks
    )

    labeled_blocks, labeled_blocks_info, max_label = _collect_labeled_blocks(
        segment_block_res,
        image.shape,
        blocksize,
    )

    if np.prod(nblocks) > 1:
        print(f'Prepare to link labels for {nblocks}', flush=True)
        new_labeling = _link_labels(
            labeled_blocks, labeled_blocks_info, max_label,
            iou_depth,
            iou_threshold
        )
        relabeled = da.map_blocks(
            _relabel_block,
            labeled_blocks,
            labeling=new_labeling,
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


def _get_block_data(image, block_info):
    block_index, block_coords = block_info
    print(f'{time.ctime(time.time())} '
        f'Get block: {block_index}, from: {block_coords}',
        flush=True)
    block_data = da.from_array(image[block_coords])
    return block_index, block_coords, block_data


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
                   block,
                   blocksize=None,
                   blockoverlaps=None,
                   preprocessing_steps=[],
):
    block_index, block_coords, block_data = block
    block_shape = tuple([sl.stop-sl.start for sl in block_coords])
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

    print(f'Prepared invocations for segmenting block {block_index}', flush=True)

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
    from cellpose.io import logger_setup

    print(f'{time.ctime(time.time())} '
          f'Run model eval for block: {block_index}, size: {block_data.shape}',
          f'3-D:{do_3D}, diameter:{diameter}',
          flush=True)

    logger_setup()
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
    print(f'{time.ctime(time.time())} ',
          f'Finished model eval for block: {block_index}',
          flush=True)

    return labels.astype(np.uint32)


def _collect_labeled_blocks(segment_blocks_res, shape, chunksize):
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
    labeled_blocks_info = []
    labels = da.empty(shape, dtype=np.uint32, chunks=chunksize)
    result_index = 0
    max_label = 0
    # collect segmentation results
    for r in segment_blocks_res:
        (block_index, block_coords, dmax_block_label, dblock_labels) = r
        block_shape = tuple([sl.stop-sl.start for sl in block_coords])
        block_labels = da.from_delayed(dblock_labels,
                                       shape=block_shape,
                                       dtype=np.uint32)

        max_block_label = da.from_delayed(dmax_block_label, shape=(), dtype=np.uint32)

        print(f'{result_index+1}. ',
            f'Write labels {block_index}, {block_coords} ',
            f'data type: {block_labels.dtype}, ',
            f'max block label: {max_block_label}, '
            f'label range: {max_label} - {max_label+max_block_label}',
            flush=True)

        block_labels_offsets = da.where(block_labels > 0,
                                        max_label,
                                        np.uint32(0)).astype(np.uint32)
        block_labels += block_labels_offsets
        # block_index, block_coords, labels_range
        labeled_blocks_info.append((block_index,
                                    block_coords,
                                    (max_label, max_label+max_block_label)))
        # set the block in the dask array of labeled blocks
        labels[block_coords] = block_labels[...]
        max_label = max_label + max_block_label
        result_index += 1

    print(f'Finished collecting labels in {shape} image',
          flush=True)
    return labels, labeled_blocks_info, max_label


def _link_labels(labels, blocks_info, nlabels, face_depth, iou_threshold):
    blocks_coords = [bi[1] for bi in blocks_info]
    label_groups = _label_adjacency_graph(labels,
                                          blocks_coords,
                                          nlabels,
                                          face_depth,
                                          iou_threshold)

    print(f'Label groups found: {label_groups}', flush=True)
    connected_comps = dask.delayed(scipy.sparse.csgraph.connected_components, nout=2)
    connected_comps_res = connected_comps(label_groups, directed=False)[1]
    return da.from_delayed(connected_comps_res, shape=(np.nan,), dtype=labels.dtype)


def _label_adjacency_graph(labels, blocks_coords, nlabels, block_face_depth,
                           iou_threshold):
    block_faces_and_axes = _get_face_slices_and_axes(blocks_coords,
                                                     labels.shape,
                                                     block_face_depth)
    all_mappings = [ da.empty((2, 0), dtype=labels.dtype, chunks=1) ]
    for face_slice, axis in block_faces_and_axes:
        face = labels[face_slice]
        mapped = _across_block_label_grouping_delayed(face, axis,
                                                      iou_threshold)
        print(f'Label mapping for face {face_slice} along {axis} axis: {mapped}',
              flush=True)
        all_mappings.append(mapped)

    i, j = da.concatenate(all_mappings, axis=1)

    # reformat as csr_matrix
    return dask.delayed(_mappings_as_csr)(i, j, nlabels+1)


def _get_face_slices_and_axes(blocks_coords, image_shape, face_depth):
    ndim = len(image_shape)
    depth = da.overlap.coerce_depth(ndim, face_depth)
    face_slices_and_axes = []
    for ax in range(ndim):
        for sl in blocks_coords:
            if sl[ax].stop == image_shape[ax]:
                # this is an end block on this axis
                continue
            slice_to_append = list(sl)
            slice_to_append[ax] = slice(
                sl[ax].stop - depth[ax], sl[ax].stop + depth[ax]
            )
            face_slices_and_axes.append((tuple(slice_to_append), ax))
    return face_slices_and_axes


def _across_block_label_grouping_delayed(face, axis, iou_threshold):
    grouped_results = dask.delayed(_across_block_label_grouping)(
        face, axis, iou_threshold
    )
    return da.from_delayed(
        grouped_results,
        shape=(2, np.nan),
        dtype=face.dtype
    )


def _across_block_label_grouping(face, axis, iou_threshold):
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

    labels0_orig = unique[labels0]
    labels1_orig = unique[labels1]
    grouped = np.stack([labels0_orig, labels1_orig])

    # Discard any mappings with bg pixels
    valid = np.all(grouped != 0, axis=0).astype(np.uint32)
    return grouped[:, valid]


def _mappings_as_csr(i, j, n):
    v = np.ones_like(i)
    return scipy.sparse.coo_matrix((v, (i, j)), shape=(n, n)).tocsr()


def _relabel_block(a, labeling=None):
    return labeling[a]
