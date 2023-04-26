from typing import List, Dict
from metrics.metric import Metric

class SummarizationMetric(Metric):
    """
    use Rouge-L as summarization metric
    """
    def __init__(self):
        super(SummarizationMetric, self).__init__()
        from rouge import Rouge
        self.rouge = Rouge()

    def calc_metric(self, golden: Dict, predictions: Dict) -> float:
        labels = []
        preds = []
        for k in golden.keys():
            labels.append(" ".join(list(golden[k])))
            preds.append(" ".join(list(predictions[k])))
        rouge_scores = self.rouge.get_scores(preds, labels, avg=True)
        return {
            "acc": None,
            "f1": None,
            "rouge-l": rouge_scores["rouge-l"]["f"]
        }
