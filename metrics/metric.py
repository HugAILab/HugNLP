from typing import List, Dict

class Metric(object):
    def __init__(self):
        super(Metric, self).__init__()

    def calc_metric(self, golden: Dict, predictions: Dict) -> float:
        """
        params:
        - golden: dictionary contains samples, each sample is a key-value indexed by ID
        - predictions: dictionary contains samples, each sample is a key-value indexed by ID
        """
        raise NotImplementedError
