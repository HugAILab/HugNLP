from typing import List, Dict
from metrics.metric import Metric

class NERMetric(Metric):
    def __init__(self):
        super(NERMetric, self).__init__()

    def _compute(self, label, pred, hit):
        if label == 0:
            recall = 1 if pred == 0 else 0
            precision = 1 if pred == 0 else (hit / pred)
        else:
            recall = hit / label
            precision = 0 if pred == 0 else (hit / pred)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall, precision, f1

    def calc_metric(self, golden: Dict, predictions: Dict) -> float:
        f1 = 0.
        acc = 0.
        for k in golden.keys():
            hit_entities = [e for e in predictions[k] if e in golden[k]]
            _recall, _precision, _f1 = self._compute(
                len(golden[k]),
                len(predictions[k]),
                len(hit_entities)
            )
            f1 += _f1
            acc += _precision
        return {
            "acc": acc/len(golden.keys()),
            "f1": f1/len(golden.keys())
        }
