import json

from swclib.metrics.length_metric import LengthMetric
from swclib.metrics.keypoint_metric import KeypointMetric
from swclib.metrics.fiber_metric import FiberMetric
from swclib.utils.json import *

default_metric_params = {
    "length": {
        "radius_threshold": 2,
        "length_threshold": 0.2,
        "scale": (1.0, 1.0, 1.0),
    },
    "keypoints": {
        "keypoint_types": ["branch", "leaf"],
        "threshold_dis": 5,
        "scale": (1.0, 1.0, 1.0),
    },
    "fiber": {
        "iou_threshold": 0.8,
        "dist_threshold": 5,
        "dist_sample": 1.0,
        "scale": (1.0, 1.0, 1.0),
    },
}

METRIC_MAP = {
    "length": LengthMetric,
    "keypoints": KeypointMetric,
    "fiber": FiberMetric,
}


class MetricManager:
    def __init__(self, metric_names=["length", "keypoints", "fiber"], collect_method='micro', scale=(1.0, 1.0, 1.0)):
        self.metric_names = metric_names
        self.collect_method = collect_method
        self.metric_params = {key:default_metric_params[key] for key in metric_names}
        for key in self.metric_params:
            self.metric_params[key]["scale"] = scale
        self.metrics = self.get_metrics()
        self.results = []

    def update_metric_params(self, metric_params: dict):
        for key in metric_params:
            if key in self.metric_params:
                self.metric_params[key].update(metric_params[key])
        self.metrics = self.get_metrics()
    
    def get_metrics(self):
        metrics = {}
        for name in self.metric_names:
            metric_class = METRIC_MAP[name]
            metric = metric_class(**self.metric_params[name])
            metrics[name] = metric
        return metrics

    def add_data(self, swc_gt, swc_pred):
        result = {}
        for name, metric in self.metrics.items():
            result[name] = metric.run(swc_gt, swc_pred)
        self.results.append(result)

    def collect(self, save_path=None):
        summary_result = {"sample_num": len(self.results)}
        for name in self.metric_names:
            if self.collect_method == 'micro':
                TP, FP, FN = 0, 0, 0
                for res in self.results:
                    TP += res[name]['TP']
                    FP += res[name]['FP']
                    FN += res[name]['FN']
                precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
                f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                summary_result[name] = {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1_score,
                    'TP': TP,
                    'FP': FP,
                    'FN': FN,
                }
            else:
                raise NotImplementedError(f"Collect method {self.collect_method} not implemented.")
        summary_result['cubes'] = self.results
        if save_path is not None:
            with open(save_path, 'w') as f:
                json.dump(summary_result, f, indent='\t', cls=NumpyEncoder)
        return summary_result