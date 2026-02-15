from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Sequence
import numpy as np

@dataclass
class SegMetrics:
    overall_acc: float
    mean_acc: float
    mean_iou: float
    mean_dice: float
    fwavacc: float
    class_dice: dict[int, float]
    class_iou: dict[int, float]

# VainF DeepLabv3Plus thing metrics
class StreamSegMetrics:
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())

    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            if k != "Class IoU" and k != "Class Dice":
                string += "%s: %f\n" % (k, v)
        return string

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist
    

    def get_confusion(self, normalise: Literal["true", "pred", "all"]) -> np.ndarray:
        """return normalised confusion matrix

        Args:
            normalise: normalise to 0-1
                - None: raw counts
                - "true": row-normalised (per GT class; rows sum to 1)
                - "pred": col-normalised (per predicted class; cols sum to 1)
                - "all" : global-normalised (sum to 1)

        Returns:
            np.ndarray: normalised 
        """

        cm = self.confusion_matrix.astype(np.float64, copy=True)

        if normalise is None:
            return cm
    
        eps = 1e-12
        if normalise == "true":
            cm /= (cm.sum(axis=1, keepdims=True) + eps)
        elif normalise == "pred":
            cm /= (cm.sum(axis=0, keepdims=True) + eps)
        elif normalise == "all":
            cm /= (cm.sum() + eps)
        else:
            raise ValueError(f"normalise must be one of None/'true'/'pred'/'all', got {normalise!r}")
        
    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
            - dice scores
        """
        hist = self.confusion_matrix

        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        # dice coefficient
        tp = np.diag(hist)
        fp = hist.sum(axis=0) - tp
        fn = hist.sum(axis=1) - tp
        denom = 2 * tp + fp + fn

        dice = 2 * tp / (denom + 1e-10)  # per-class dice

        # only keep classes that have ground truth pixels
        valid = (tp + fn) > 0
        mean_dice = dice[valid].mean() if np.any(valid) else 0.0

        cls_dice = dict(zip(range(self.n_classes), dice))

        # return SegMetrics(
        #     overall_acc=acc,
        #     mean_acc=acc_cls,
        #     fwavacc=fwavacc,
        #     mean_iou=mean_iu,
        #     mean_dice=mean_dice,
        #     class_dice=cls_dice,
        #     class_iou=cls_dice
        # )

        return {
            "overall_acc": acc,
            "mean_acc": acc_cls,
            "fwavcc": fwavacc,
            "mean_iou": mean_iu,
            "mean_dice": mean_dice,
            "class_dice": cls_dice,
            "class_iou": cls_iu,
        }

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
