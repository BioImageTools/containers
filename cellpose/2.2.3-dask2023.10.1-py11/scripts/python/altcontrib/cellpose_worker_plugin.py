import os

from dask.distributed import Worker
from distributed.diagnostics.plugin import WorkerPlugin


class SetEnvWorkerPlugin(WorkerPlugin):

    def __init__(self, models_dir):
        self.cellpose_models_dir = models_dir

    def setup(self, worker: Worker):
        print('Set env.CELLPOSE_LOCAL_MODELS_PATH to', self.cellpose_models_dir, flush=True)
        os.environ['CELLPOSE_LOCAL_MODELS_PATH'] = self.cellpose_models_dir

    def teardown(self, worker: Worker):
        pass

    def transition(self, key: str, start: str, finish: str, **kwargs):
        pass

    def release_key(self, key: str, state: str, cause: str | None, reason: None, report: bool):
        pass
