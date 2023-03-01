import logging
from typing import Dict

logger = logging.getLogger(__name__)

try:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import matthews_corrcoef, f1_score
    _has_sklearn = True
except (AttributeError, ImportError) as e:
    logger.warning(
        'To use data.metrics please install scikit-learn. See https://scikit-learn.org/stable/index.html'
    )
    _has_sklearn = False


class PearsonMetric(Metric):
    """use pearson metric."""
    def __init__(self):
        super(PearsonMetric, self).__init__()

    def calc_metric(self, golden: Dict, predictions: Dict) -> float:
        golden_list, prediction_list = list(), list()
        for golden_k, golden_v in golden.items():
            if golden_k not in predictions.keys():
                continue
            golden_list.append(golden_v)
            prediction_list.append(predictions[golden_k])
        pearson_corr = pearsonr(prediction_list, golden_list)[0]
        spearman_corr = spearmanr(prediction_list, golden_list)[0]
        return {
            'pearson': pearson_corr,
            'spearmanr': spearman_corr,
            'corr': (pearson_corr + spearman_corr) / 2,
        }
