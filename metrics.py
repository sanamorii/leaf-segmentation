import numpy as np

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

        return {
            "Overall Acc": acc,
            "Mean Acc": acc_cls,
            "FreqW Acc": fwavacc,
            "Mean IoU": mean_iu,
            "Class IoU": cls_iu,
            "Mean Dice": mean_dice,
            "Class Dice": cls_dice,
        }

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
