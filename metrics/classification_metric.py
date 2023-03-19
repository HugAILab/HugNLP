from typing import List, Dict
from metrics.metric import Metric
from sklearn.metrics import matthews_corrcoef, roc_auc_score, classification_report
from sklearn.metrics import f1_score, precision_score, recall_score


class ClassificationMetric(Metric):
    """
    Calculate metrics for classification
    """
    def __init__(self):
        super(ClassificationMetric, self).__init__()

    def calc_metric(self, golden: Dict, predictions: Dict) -> float:
        correctness = 0.
        total = len(golden.keys())
        golden_list, prediction_list = list(), list()
        for golden_k, golden_v in golden.items():
            if golden_k not in predictions.keys():
                continue
            correctness += 1. if golden_v == predictions[golden_k] else 0.
            golden_list.append(golden_v)
            prediction_list.append(predictions[golden_k])
        acc = round((correctness/total), 4)
        f1 = f1_score(y_true=golden_list, y_pred=prediction_list, average="macro")
        # auc = roc_auc_score(y_true=golden_list, y_score=prediction_list, multi_class="ovr")
        # mcc = matthews_corrcoef(y_true=golden_list, y_pred=prediction_list)
        return {
            "acc": acc,
            "f1": f1,
            "acc_and_f1": (acc + f1) / 2,
            # "auc": auc,
            # "mcc": mcc,
        }
