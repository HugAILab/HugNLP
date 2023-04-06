from typing import List, Dict
from metrics.metric import Metric
from sklearn.metrics import matthews_corrcoef, roc_auc_score, classification_report
from sklearn.metrics import f1_score, precision_score, recall_score
from metrics.bleu import compute_bleu


class GenerationMetric(Metric):
    """
    Calculate metrics for classification
    """
    def __init__(self):
        super(GenerationMetric, self).__init__()

    def calc_metric(self, golden: Dict, predictions: Dict) -> float:
        """
        golden: {"idx": "xxxxx", ...}
        predictions: {"idx": ["xxx", "xxxxx", ...], ...}
        """
        correctness = 0.
        total = len(golden.keys())
        golden_list, prediction_list = list(), list()
        for golden_k, golden_v in golden.items():
            if golden_k not in predictions.keys():
                continue
            for prediction in predictions[golden_k]:
                # recall k prediction of each example, if exist one prediction equal to golden
                if golden_v == prediction:
                    correctness += 1.
                    break
            golden_list.append(golden_v)
            prediction_list.append(predictions[golden_k])
        em = round((correctness/total), 4)
        bleu_1 = compute_bleu(prediction_list, golden_list, max_order=1)
        bleu_2 = compute_bleu(prediction_list, golden_list, max_order=2)
        bleu_3 = compute_bleu(prediction_list, golden_list, max_order=3)
        bleu_4 = compute_bleu(prediction_list, golden_list, max_order=4)
        return {
            "em": em,
            "bleu-1": bleu_1[0],
            "bleu-2": bleu_2[0],
            "bleu-3": bleu_3[0],
            "bleu-4": bleu_4[0],
        }
