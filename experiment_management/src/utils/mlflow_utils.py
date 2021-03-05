"""
MLflow Logger
-------------
"""
import re
from argparse import Namespace
from time import time
from typing import Any, Dict, Optional, Union

from pytorch_lightning import _logger as log
from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_experiment
from pytorch_lightning.utilities import _module_available, rank_zero_only, rank_zero_warn

LOCAL_FILE_URI_PREFIX = "file:"

_MLFLOW_AVAILABLE = _module_available("mlflow")
try:
    import mlflow
    from mlflow.tracking import MlflowClient
# todo: there seems to be still some remaining import error with Conda env
except ImportError:
    _MLFLOW_AVAILABLE = False
    mlflow, MlflowClient = None, None


class MLFlowLogger(LightningLoggerBase):

    LOGGER_JOIN_CHAR = '-'

    def __init__(
        self,
        experiment_name: str = 'default',
        tracking_uri: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        save_dir: Optional[str] = './mlruns',
        prefix: str = '',
    ):
        if mlflow is None:
            raise ImportError(
                'You want to use `mlflow` logger which is not installed yet,'
                ' install it with `pip install mlflow`.'
            )
        super().__init__()
        if not tracking_uri:
            tracking_uri = f'{LOCAL_FILE_URI_PREFIX}{save_dir}'

        self._experiment_name = experiment_name
        self._experiment_id = None
        self._tracking_uri = tracking_uri
        self._run_id = None
        self.tags = tags
        self._prefix = prefix
        self._mlflow_client = MlflowClient(tracking_uri)

    @property
    @rank_zero_experiment
    def experiment(self) -> MlflowClient:
        r"""
        Actual MLflow object. To use MLflow features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.

        Example::

            self.logger.experiment.some_mlflow_function()

        """
        if self._experiment_id is None:
            expt = self._mlflow_client.get_experiment_by_name(self._experiment_name)
            if expt is not None:
                self._experiment_id = expt.experiment_id
            else:
                log.warning(f'Experiment with name {self._experiment_name} not found. Creating it.')
                self._experiment_id = self._mlflow_client.create_experiment(name=self._experiment_name)

        if self._run_id is None:
            run = self._mlflow_client.create_run(experiment_id=self._experiment_id, tags=self.tags)
            self._run_id = run.info.run_id
        return self._mlflow_client

    @property
    def run_id(self):
        # create the experiment if it does not exist to get the run id
        _ = self.experiment
        return self._run_id

    @property
    def experiment_id(self):
        # create the experiment if it does not exist to get the experiment id
        _ = self.experiment
        return self._experiment_id

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        params = self._convert_params(params)
        params = self._flatten_dict(params)
        for k, v in params.items():
            if len(str(v)) > 250:
                rank_zero_warn(
                    f"Mlflow only allows parameters with up to 250 characters. Discard {k}={v}", RuntimeWarning
                )
                continue

            self.experiment.log_param(self.run_id, k, v)


    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        assert rank_zero_only.rank == 0, 'experiment tried to log from global_rank != 0'

        metrics = self._add_prefix(metrics)

        timestamp_ms = int(time() * 1000)
        for k, v in metrics.items():
            if isinstance(v, str):
                log.warning(f'Discarding metric with string value {k}={v}.')
                continue

            new_k = re.sub("[^a-zA-Z0-9_/. -]+", "", k)
            if k != new_k:
                rank_zero_warn(
                    "MLFlow only allows '_', '/', '.' and ' ' special characters in metric name."
                    f" Replacing {k} with {new_k}.", RuntimeWarning
                )
                k = new_k

            self.experiment.log_metric(self.run_id, k, v, timestamp_ms, step)


    @rank_zero_only
    def finalize(self, status: str = 'FINISHED') -> None:
        super().finalize(status)
        status = 'FINISHED' if status == 'success' else status
        if self.experiment.get_run(self.run_id):
            self.experiment.set_terminated(self.run_id, status)


    @property
    def save_dir(self) -> Optional[str]:
        """
        The root file directory in which MLflow experiments are saved.

        Return:
            Local path to the root experiment directory if the tracking uri is local.
            Otherwhise returns `None`.
        """
        if self._tracking_uri.startswith(LOCAL_FILE_URI_PREFIX):
            return self._tracking_uri.lstrip(LOCAL_FILE_URI_PREFIX)

    @property
    def name(self) -> str:
        return self.experiment_id

    @property
    def version(self) -> str:
        return self.run_id