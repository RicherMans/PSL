import torch
import ignite.metrics as metrics
from typing import Callable
from functools import partial
from loguru import logger
from ignite.metrics import EpochMetric
from ignite.contrib.metrics import ROC_AUC
from scipy import stats
import numpy as np

def _d_prime(auc):
    return stats.norm().ppf(auc) * np.sqrt(2.0)

class DPrime(ROC_AUC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self):
        auc = super().compute()
        return _d_prime(auc)


def average_precision_compute_fn(y_preds: torch.Tensor,
                                 y_targets: torch.Tensor,
                                 average:str = 'macro',
                                 ) -> float:
    try:
        from sklearn.metrics import average_precision_score
    except ImportError:
        raise RuntimeError(
            "This contrib module requires sklearn to be installed.")

    y_true = y_targets.numpy()
    y_pred = y_preds.numpy()
    return average_precision_score(y_true, y_pred, average=average)

class AveragePrecision(EpochMetric):
    def __init__(self, average='macro', output_transform: Callable = lambda x: x, check_compute_fn: bool = False) -> None:
        super(AveragePrecision,
              self).__init__(partial(average_precision_compute_fn,
                                     average=average),
                             output_transform=output_transform,
                             check_compute_fn=check_compute_fn)

BCELoss = torch.nn.BCELoss
MSELoss = torch.nn.MSELoss
MAELoss = torch.nn.L1Loss
mAP = AveragePrecision
CrossEntropyLoss = torch.nn.CrossEntropyLoss
BCEWithLogitsLoss = torch.nn.BCEWithLogitsLoss



