import os
import sys
import traceback

from cellpose.models import get_user_models
from cellpose import version_str as cellpose_version
from cellpose.cli import get_arg_parser


def _define_args():
    args_parser = get_arg_parser()


    args_parser.add_argument('--models-dir', dest='models_dir',
                             type=str,
                             help='cache cellpose models directory')
    args_parser.add_argument('--model', dest='segmentation_model',
                             type=str,
                             default='cyto',
                             help='segmentation model')
    return args_parser


def _download_cellpose_models(args):

    if args.models_dir is not None:
        os.environ['CELLPOSE_LOCAL_MODELS_PATH'] = args.models_dir

    try:
        print('Cache cellpose models', args.model_type, flush=True)
        get_user_models()
    except:
        raise


def _print_version_and_exit():
    print(cellpose_version)
    sys.exit(0)


if __name__ == '__main__':
    args_parser = _define_args()
    args = args_parser.parse_args()
    print('Get cellpose models:', args, flush=True)
    try:
        if args.version:
            _print_version_and_exit()

        _download_cellpose_models(args)
        sys.exit(0)
    except Exception as err:
        print('Cellpose labeling error:', err)
        traceback.print_exception(err)
        sys.exit(1)
