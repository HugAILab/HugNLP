from metrics.classification_metric import ClassificationMetric
from metrics.summarization_metric import SummarizationMetric
from metrics.mrc_metric import MRCMetric
from metrics.token_cls_metric import NERMetric

datatype2metrics = {
    "classification": ClassificationMetric,
    "mrc": MRCMetric,
    "ner": NERMetric,
    "clue_ner": NERMetric,
    "summ": SummarizationMetric,
    "yesno": ClassificationMetric,
    "keys": MRCMetric,
}
