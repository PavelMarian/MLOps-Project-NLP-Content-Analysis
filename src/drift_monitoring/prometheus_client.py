from prometheus_client import Counter, Gauge, Histogram, Info, generate_latest, REGISTRY
import time
from typing import Dict, Any


class MetricsCollector:
    def __init__(self):
        self.prediction_counter = Counter(
            'toxicity_predictions_total',
            'Total number of predictions',
            ['label']
        )

        self.prediction_latency = Histogram(
            'toxicity_prediction_latency_seconds',
            'Prediction latency in seconds',
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
        )

        self.drift_score = Gauge(
            'toxicity_drift_score',
            'Current drift score',
            ['drift_type']
        )

        self.drift_detected = Gauge(
            'toxicity_drift_detected',
            'Whether drift is detected (1) or not (0)',
            ['drift_type']
        )

        self.model_info = Info(
            'toxicity_model',
            'Information about current model'
        )

        self.model_f1 = Gauge(
            'toxicity_model_f1_score',
            'Current model F1 score'
        )

        self.toxic_rate = Gauge(
            'toxicity_toxic_rate',
            'Current rate of toxic predictions'
        )

        self.feedback_count = Gauge(
            'toxicity_feedback_count',
            'Total number of feedback samples'
        )

        self.prediction_count = Gauge(
            'toxicity_prediction_count',
            'Total number of predictions'
        )

        self.last_drift_check = Gauge(
            'toxicity_last_drift_check_timestamp',
            'Timestamp of last drift check'
        )

        self.data_drift_ks_stat = Gauge('toxicity_data_drift_ks_stat', 'KS statistic for length distribution')
        self.data_drift_ks_pvalue = Gauge('toxicity_data_drift_ks_pvalue', 'KS p-value for length distribution')
        self.data_drift_vocab_jaccard = Gauge('toxicity_data_drift_vocab_jaccard', 'Jaccard similarity of vocabularies')
        self.data_drift_vocab_score = Gauge('toxicity_data_drift_vocab_score', 'Vocab drift score (1 - Jaccard)')

        self.target_drift_ref_rate = Gauge('toxicity_target_drift_reference_rate', 'Positive rate in reference')
        self.target_drift_curr_rate = Gauge('toxicity_target_drift_current_rate', 'Positive rate in current')
        self.target_drift_rel_change = Gauge('toxicity_target_drift_relative_change', 'Relative change in positive rate')

        self.concept_drift_f1_ref = Gauge('toxicity_concept_drift_f1_reference', 'F1 on reference')
        self.concept_drift_f1_curr = Gauge('toxicity_concept_drift_f1_current', 'F1 on current')
        self.concept_drift_rel_drop = Gauge('toxicity_concept_drift_relative_drop', 'Relative F1 drop')


    def record_prediction(self, label: str, latency: float):
        self.prediction_counter.labels(label=label).inc()
        self.prediction_latency.observe(latency)
        self.prediction_count.set(self.prediction_count._value.get() + 1)

    def update_drift_metrics(self, drift_report: Dict[str, Any]):
        if 'data_drift' in drift_report and drift_report['data_drift']:
            data_drift = drift_report['data_drift']
            self.drift_score.labels(drift_type='data').set(data_drift['score'])
            self.drift_detected.labels(drift_type='data').set(1 if data_drift['detected'] else 0)

        if 'target_drift' in drift_report and drift_report['target_drift']:
            target_drift = drift_report['target_drift']
            self.drift_score.labels(drift_type='target').set(target_drift['score'])
            self.drift_detected.labels(drift_type='target').set(1 if target_drift['detected'] else 0)

        if 'concept_drift' in drift_report and drift_report['concept_drift']:
            concept_drift = drift_report['concept_drift']
            self.drift_score.labels(drift_type='concept').set(concept_drift['score'])
            self.drift_detected.labels(drift_type='concept').set(1 if concept_drift['detected'] else 0)

        if 'data_drift' in drift_report and drift_report['data_drift']:
            dd = drift_report['data_drift']
            self.data_drift_ks_stat.set(dd.get('length_ks_stat', 0))
            self.data_drift_ks_pvalue.set(dd.get('length_ks_pvalue', 0))
            self.data_drift_vocab_jaccard.set(dd.get('vocab_jaccard', 0))
            self.data_drift_vocab_score.set(dd.get('vocab_drift_score', 0))

        if 'target_drift' in drift_report and drift_report['target_drift']:
            td = drift_report['target_drift']
            self.target_drift_ref_rate.set(td.get('reference_positive_rate', 0))
            self.target_drift_curr_rate.set(td.get('current_positive_rate', 0))
            self.target_drift_rel_change.set(td.get('relative_change', 0))

        if 'concept_drift' in drift_report and drift_report['concept_drift']:
            cd = drift_report['concept_drift']
            self.concept_drift_f1_ref.set(cd.get('reference_f1', 0))
            self.concept_drift_f1_curr.set(cd.get('current_f1', 0))
            self.concept_drift_rel_drop.set(cd.get('relative_drop', 0))

        self.last_drift_check.set(time.time())

    def update_model_metrics(self, model_info: Dict[str, Any]):
        self.model_info.info({
            'model_path': model_info.get('model_path', 'unknown'),
            'model_type': model_info.get('model_type', 'unknown'),
            'device': model_info.get('device', 'unknown'),
            'threshold': str(model_info.get('threshold', 0.5))
        })
        if 'f1_score' in model_info:
            self.model_f1.set(model_info['f1_score'])

    def update_feedback_count(self, count: int):
        self.feedback_count.set(count)

    def update_toxic_rate(self, rate: float):
        self.toxic_rate.set(rate)


metrics = MetricsCollector()


def get_metrics():
    return generate_latest(REGISTRY)
