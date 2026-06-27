from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Optional, Any

import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import f1_score
import mlflow


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


@dataclass
class DataDriftReport:
    detected: bool
    score: float
    length_ks_stat: float
    length_ks_pvalue: float
    vocab_jaccard: float
    vocab_drift_score: float
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().timestamp()


@dataclass
class TargetDriftReport:
    detected: bool
    score: float
    reference_positive_rate: float
    current_positive_rate: float
    relative_change: float
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().timestamp()


@dataclass
class ConceptDriftReport:
    detected: bool
    score: float
    reference_f1: float
    current_f1: float
    relative_drop: float
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().timestamp()


@dataclass
class DriftReport:
    data_drift: Optional[DataDriftReport]
    target_drift: Optional[TargetDriftReport]
    concept_drift: Optional[ConceptDriftReport]
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().timestamp()

    def to_dict(self) -> Dict[str, Any]:
        result = {'timestamp': self.timestamp}
        if self.data_drift:
            result['data_drift'] = asdict(self.data_drift)
        if self.target_drift:
            result['target_drift'] = asdict(self.target_drift)
        if self.concept_drift:
            result['concept_drift'] = asdict(self.concept_drift)
        return result

    def has_drift(self) -> bool:
        return (
                (self.data_drift and self.data_drift.detected) or
                (self.target_drift and self.target_drift.detected) or
                (self.concept_drift and self.concept_drift.detected)
        )


class DriftDetector:
    def __init__(self, reference_texts: list[str], reference_labels: Optional[np.ndarray] = None):
        self.reference_texts = reference_texts
        self.reference_labels = reference_labels

        self.ref_lengths = [len(tokenize(text)) for text in reference_texts]

        self.ref_vocab: set[str] = set()
        for text in reference_texts:
            self.ref_vocab.update(tokenize(text))

        self.drift_history: List[DriftReport] = []
        self.last_report: Optional[DriftReport] = None

    def detect_data_drift(
            self,
            current_texts: list[str],
            alpha: float = 0.05,
    ) -> DataDriftReport:
        curr_lengths = [len(tokenize(text)) for text in current_texts]
        ks_stat, ks_pvalue = ks_2samp(self.ref_lengths, curr_lengths)

        curr_vocab: set[str] = set()
        for text in current_texts:
            curr_vocab.update(tokenize(text))

        intersection = len(curr_vocab & self.ref_vocab)
        union = len(curr_vocab | self.ref_vocab)
        jaccard = intersection / union if union else 1.0
        vocab_drift_score = 1.0 - jaccard

        drift_score = 0.5 * ks_stat + 0.5 * vocab_drift_score
        detected = ks_pvalue < alpha and drift_score > 0.3

        return DataDriftReport(
            detected=bool(detected),
            score=float(drift_score),
            length_ks_stat=float(ks_stat),
            length_ks_pvalue=float(ks_pvalue),
            vocab_jaccard=float(jaccard),
            vocab_drift_score=float(vocab_drift_score),
        )

    def detect_target_drift(
            self,
            y_reference: np.ndarray,
            y_current: np.ndarray,
            threshold: float = 0.2,
    ) -> TargetDriftReport:
        y_reference = np.asarray(y_reference).reshape(-1)
        y_current = np.asarray(y_current).reshape(-1)

        reference_rate = float(np.mean(y_reference))
        current_rate = float(np.mean(y_current))
        relative_change = abs(current_rate - reference_rate) / (reference_rate + 1e-12)
        detected = relative_change > threshold

        return TargetDriftReport(
            detected=bool(detected),
            score=float(relative_change),
            reference_positive_rate=reference_rate,
            current_positive_rate=current_rate,
            relative_change=float(relative_change),
        )

    def detect_concept_drift(
            self,
            y_true_reference: np.ndarray,
            y_pred_reference: np.ndarray,
            y_true_current: np.ndarray,
            y_pred_current: np.ndarray,
            threshold: float = 0.1,
    ) -> ConceptDriftReport:
        y_true_reference = np.asarray(y_true_reference).reshape(-1)
        y_pred_reference = np.asarray(y_pred_reference).reshape(-1)
        y_true_current = np.asarray(y_true_current).reshape(-1)
        y_pred_current = np.asarray(y_pred_current).reshape(-1)

        reference_f1 = f1_score(y_true_reference, y_pred_reference)
        current_f1 = f1_score(y_true_current, y_pred_current)
        relative_drop = max(0.0, reference_f1 - current_f1) / (reference_f1 + 1e-12)
        detected = relative_drop > threshold

        return ConceptDriftReport(
            detected=bool(detected),
            score=float(relative_drop),
            reference_f1=float(reference_f1),
            current_f1=float(current_f1),
            relative_drop=float(relative_drop),
        )

    def generate_report(
            self,
            current_texts: list[str],
            y_reference: np.ndarray | None = None,
            y_current: np.ndarray | None = None,
            y_true_reference: np.ndarray | None = None,
            y_pred_reference: np.ndarray | None = None,
            y_true_current: np.ndarray | None = None,
            y_pred_current: np.ndarray | None = None,
            log_to_mlflow: bool = True,
    ) -> DriftReport:
        data_report = self.detect_data_drift(current_texts)

        target_report = None
        if y_reference is not None and y_current is not None:
            target_report = self.detect_target_drift(
                y_reference=y_reference,
                y_current=y_current,
            )

        concept_report = None
        if (
                y_true_reference is not None
                and y_pred_reference is not None
                and y_true_current is not None
                and y_pred_current is not None
        ):
            concept_report = self.detect_concept_drift(
                y_true_reference=y_true_reference,
                y_pred_reference=y_pred_reference,
                y_true_current=y_true_current,
                y_pred_current=y_pred_current,
            )

        report = DriftReport(
            data_drift=data_report,
            target_drift=target_report,
            concept_drift=concept_report,
        )

        self.drift_history.append(report)
        self.last_report = report

        if log_to_mlflow:
            self._log_to_mlflow(report)

        return report

    def _log_to_mlflow(self, report: DriftReport):
        try:
            active_run = mlflow.active_run()
            if active_run is None:
                mlflow.start_run(run_name="drift_check", nested=True)
            else:
                pass

            if report.data_drift:
                mlflow.log_params({
                    "data_drift_detected": report.data_drift.detected,
                    "data_drift_score": report.data_drift.score,
                    "length_ks_stat": report.data_drift.length_ks_stat,
                    "length_ks_pvalue": report.data_drift.length_ks_pvalue,
                    "vocab_jaccard": report.data_drift.vocab_jaccard,
                    "vocab_drift_score": report.data_drift.vocab_drift_score,
                })

            if report.target_drift:
                mlflow.log_params({
                    "target_drift_detected": report.target_drift.detected,
                    "target_drift_score": report.target_drift.score,
                    "reference_positive_rate": report.target_drift.reference_positive_rate,
                    "current_positive_rate": report.target_drift.current_positive_rate,
                    "relative_change": report.target_drift.relative_change,
                })

            if report.concept_drift:
                mlflow.log_params({
                    "concept_drift_detected": report.concept_drift.detected,
                    "concept_drift_score": report.concept_drift.score,
                    "reference_f1": report.concept_drift.reference_f1,
                    "current_f1": report.concept_drift.current_f1,
                    "relative_drop": report.concept_drift.relative_drop,
                })

            mlflow.log_metric("drift_detected_any", 1 if report.has_drift() else 0)

        except Exception as e:
            print(f"Error logging to MLflow: {e}")

    def get_drift_summary(self) -> Dict[str, Any]:
        if not self.drift_history:
            return {"status": "no_data", "message": "No drift checks performed"}

        last_report = self.drift_history[-1]
        total_checks = len(self.drift_history)
        drift_count = sum(1 for r in self.drift_history if r.has_drift())

        summary = {
            "status": "ok",
            "total_checks": total_checks,
            "drift_count": drift_count,
            "drift_rate": drift_count / total_checks if total_checks > 0 else 0,
            "last_check": {
                "data_drift": last_report.data_drift is not None,
                "target_drift": last_report.target_drift is not None,
                "concept_drift": last_report.concept_drift is not None,
                "has_drift": last_report.has_drift(),
                "timestamp": last_report.timestamp
            }
        }

        if last_report.data_drift:
            summary["last_check"]["data_drift_score"] = last_report.data_drift.score
        if last_report.target_drift:
            summary["last_check"]["target_drift_score"] = last_report.target_drift.score
        if last_report.concept_drift:
            summary["last_check"]["concept_drift_score"] = last_report.concept_drift.score

        return summary

    def detect(self, current_texts: list[str]) -> Dict[str, Any]:
        report = self.generate_report(current_texts, log_to_mlflow=False)
        return {
            "drift_detected": report.has_drift(),
            "drift_score": report.data_drift.score if report.data_drift else 0.0,
            "data_drift": report.data_drift.detected if report.data_drift else False,
            "target_drift": report.target_drift.detected if report.target_drift else False,
            "concept_drift": report.concept_drift.detected if report.concept_drift else False,
        }
