"""
utils.py
Evaluation metrics for deepfake detection (ACC, AUC, EER, APCER, BPCER, ACER).
"""
import numpy as np
from easydict import EasyDict
from sklearn.metrics import roc_curve, auc, confusion_matrix
from scipy.optimize import brentq
from scipy.interpolate import interp1d


def cal_metrics(y_trues, y_preds, threshold=0.5):
    """
    Compute detection metrics.

    Args:
        y_trues (list[int]): ground-truth labels
        y_preds (list[float]): predicted probabilities for the fake class
        threshold (float | 'auto' | 'best'): decision threshold
            'auto'  -> EER threshold
            'best'  -> threshold minimising ACER
            float   -> fixed threshold (default 0.5)

    Returns:
        EasyDict with keys: AUC, EER, Thre, ACC, APCER, BPCER, ACER
    """
    metrics = EasyDict()
    fpr, tpr, thresholds = roc_curve(y_trues, y_preds)
    metrics.AUC = auc(fpr, tpr)
    metrics.EER = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    metrics.Thre = float(interp1d(fpr, thresholds)(metrics.EER))

    if threshold == 'best':
        _, best_metrics = _find_best_threshold(y_trues, y_preds)
        return best_metrics
    elif threshold == 'auto':
        threshold = metrics.Thre

    prediction = (np.array(y_preds) > threshold).astype(int)
    res = confusion_matrix(y_trues, prediction, labels=[0, 1])
    TP, FN = res[0, :]
    FP, TN = res[1, :]

    metrics.ACC = (TP + TN) / len(y_trues)
    metrics.APCER = float(FP / (TN + FP))
    metrics.BPCER = float(FN / (FN + TP))
    metrics.ACER  = (metrics.APCER + metrics.BPCER) / 2
    return metrics


def _find_best_threshold(y_trues, y_preds):
    best_thre = 0.5
    best_metrics = None
    for thre in np.unique(np.sort(y_preds)):
        metrics = cal_metrics(y_trues, y_preds, threshold=thre)
        if best_metrics is None or metrics.ACER < best_metrics.ACER:
            best_metrics = metrics
            best_thre = thre
    return best_thre, best_metrics
