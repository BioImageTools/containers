import io_utils.read_utils as read_utils
import io_utils.write_utils as write_utils
import sys
import traceback
import yaml

from cellpose import version_str as cellpose_version
from cellpose.cli import get_arg_parser
from dask.distributed import (Client, LocalCluster)
from flatten_json import flatten

from altcontrib.distributed_segmentation import distributed_segmentation

def _inttuple(arg):
    if arg is not None and arg.strip():
        return tuple([int(d) for d in arg.split(',')])
    else:
        return ()


def _define_args():
    args_parser = get_arg_parser()
    args_parser.add_argument('-i','--input',
                             dest='input',
                             type=str,
                             help = "input directory")
    args_parser.add_argument('--input-subpath', '--input_subpath',
                             dest='input_subpath',
                             type=str,
                             help = "input subpath")

    args_parser.add_argument('--mask',
                             dest='mask',
                             type=str,
                             help = "mask directory")
    args_parser.add_argument('--mask-subpath', '--mask_subpath',
                             dest='mask_subpath',
                             type=str,
                             help = "mask subpath")

    args_parser.add_argument('-o','--output',
                             dest='output',
                             type=str,
                             help = "output file")
    args_parser.add_argument('--output-subpath', '--output_subpath',
                             dest='output_subpath',
                             type=str,
                             help = "output subpath")
    args_parser.add_argument('--output-chunk-size', '--output_chunk_size',
                             dest='output_chunk_size',
                             default=128,
                             type=int,
                             help='Output chunk size as a single int')
    args_parser.add_argument('--output-blocksize', '--output_blocksize',
                             dest='output_blocksize',
                             type=_inttuple,
                             help='Output chunk size as a tuple (x,y,z).')
    args_parser.add_argument('--process-blocksize', '--process_blocksize',
                             dest='process_blocksize',
                             type=_inttuple,
                             help='Output chunk size as a tuple (x,y,z).')
    args_parser.add_argument('--blocks-overlaps', '--blocks_overlaps',
                             dest='blocks_overlaps',
                             type=_inttuple,
                             help='Blocks overlaps as a tuple (x,y,z).')
    args_parser.add_argument('--eval-channels', '--eval_channels',
                             dest='eval_channels',
                             type=_inttuple,
                             help='Cellpose channels: 0,0 - gray images')

    distributed_args = args_parser.add_argument_group("Distributed Arguments")
    distributed_args.add_argument('--dask-scheduler', '--dask_scheduler',
                                  dest='dask_scheduler',
                                  type=str, default=None,
                                  help='Run with distributed scheduler')
    distributed_args.add_argument('--dask-config', '--dask_config',
                                  dest='dask_config',
                                  type=str, default=None,
                                  help='Dask configuration yaml file')
    distributed_args.add_argument('--max-cellpose-tasks', '--max_cellpose_tasks',
                                  dest='max_cellpose_tasks',
                                  type=int, default=-1,
                                  help='Max dask cellpose tasks')
    distributed_args.add_argument('--device', required=False, default='0', type=str,
                                  dest='device',
                                  help='which device to use, use an integer for torch, or mps for M1')    
    distributed_args.add_argument('--model', dest='segmentation_model',
                                  type=str,
                                  default='cyto',
                                  help='segmentation model')
    distributed_args.add_argument('--iou-threshold', '--iou_threshold',
                                  dest='iou_threshold',
                                  type=float,
                                  default=0.7,
                                  help='Intersection over union threshold')
    distributed_args.add_argument('--iou-depth', '--iou_depth',
                                  dest='iou_depth',
                                  type=int,
                                  default=1,
                                  help='Intersection over union depth')
    return args_parser


def _run_segmentation(args):
    image_data, image_attrs = read_utils.open(args.input, args.input_subpath)

    if (args.dask_config):
        import dask.config
        print(f'Use dask config: {args.dask_config}', flush=True)
        with open(args.dask_config) as f:
            dask_config = flatten(yaml.safe_load(f))
            dask.config.set(dask_config)

    if args.dask_scheduler:
        dask_client = Client(address=args.dask_scheduler)
    else:
        # use a local asynchronous client
        dask_client = Client(LocalCluster())

    print('Image data shape/dim/dtype:', 
          image_data.shape, image_data.ndim, image_data.dtype,
          flush=True)
    
    if args.output:
        output_subpath = args.output_subpath if args.output_subpath else args.input_subpath

        if (args.output_blocksize is not None and
            len(args.output_blocksize) == image_data.ndim):
            output_blocks = args.output_blocksize[::-1] # make it zyx
        else:
            # default to output_chunk_size
            output_blocks = (args.output_chunk_size,) * image_data.ndim

        if (args.process_blocksize is not None and
            len(args.process_blocksize) == image_data.ndim):
            process_blocksize = args.process_blocksize[::-1] # make it zyx
        else:
            process_blocksize = output_blocks

        if (args.blocks_overlaps is not None and
            len(args.blocks_overlaps) > 0):
            blocks_overlaps = args.blocks_overlaps[::-1] # make it zyx
        else:
            blocks_overlaps = ()

        if args.eval_channels and len(args.eval_channels) > 0:
            eval_channels = list(args.eval_channels)
        else:
            eval_channels = None

        try:
            output_labels = distributed_segmentation(
                image_data,
                args.segmentation_model,
                args.diam_mean,
                blocksize=process_blocksize,
                blocksoverlap=blocks_overlaps,
                anisotropy=args.anisotropy,
                min_size=args.min_size,
                resample=(not (args.no_resample or
                                args.fast_mode)),
                flow_threshold=args.flow_threshold,
                cellprob_threshold=args.cellprob_threshold,
                eval_channels=eval_channels,
                use_net_avg=args.net_avg,
                use_gpu=args.use_gpu,
                device=args.device,
                max_tasks=args.max_cellpose_tasks,
                iou_threshold=args.iou_threshold,
                iou_depth=args.iou_depth,
            )

            persisted_labels = write_utils.save(
                output_labels, args.output, output_subpath,
                blocksize=output_blocks,
                resolution=image_attrs.get('pixelResolution'),
                scale_factors=image_attrs.get('downsamplingFactors'),
            )

            if persisted_labels is not None:
                dask_client.compute(persisted_labels).result()

        except:
            raise


def _print_version_and_exit():
    print(cellpose_version)
    sys.exit(0)


if __name__ == '__main__':
    args_parser = _define_args()
    args = args_parser.parse_args()
    print('Invoked cellpose segmentation:', args, flush=True)
    try:
        if args.version:
            _print_version_and_exit()

        _run_segmentation(args)
        sys.exit(0)
    except Exception as err:
        print('Segmentation error:', err)
        traceback.print_exception(err)
        sys.exit(1)