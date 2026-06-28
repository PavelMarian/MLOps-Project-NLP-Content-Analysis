import numpy as np
import pytest
from src.drift_monitoring.drift_detector import DriftDetector, DataDriftReport, TargetDriftReport, ConceptDriftReport


def test_drift_detector_initialization():
    ref_texts = ["один два", "три четыре", "пять шесть"]
    ref_labels = np.array([0, 1, 0])
    detector = DriftDetector(reference_texts=ref_texts, reference_labels=ref_labels)

    assert len(detector.reference_texts) == 3
    assert detector.reference_labels is not None
    assert len(detector.ref_lengths) == 3
    assert "один" in detector.ref_vocab
    assert "два" in detector.ref_vocab


def test_detect_data_drift():
    ref_texts = ["один два три", "четыре пять", "шесть семь восемь"]
    detector = DriftDetector(reference_texts=ref_texts)

    current_texts = ["один два", "три четыре пять", "шесть"]
    report = detector.detect_data_drift(current_texts)

    assert isinstance(report, DataDriftReport)
    assert report.detected is not None
    assert report.score >= 0.0
    assert 0.0 <= report.length_ks_pvalue <= 1.0
    assert 0.0 <= report.vocab_jaccard <= 1.0


def test_detect_target_drift():
    y_ref = np.array([0, 1, 0, 1, 0])
    y_cur = np.array([1, 1, 1, 0, 1])
    detector = DriftDetector(reference_texts=[], reference_labels=y_ref)

    report = detector.detect_target_drift(y_reference=y_ref, y_current=y_cur, threshold=0.2)

    assert isinstance(report, TargetDriftReport)
    assert report.detected is True
    assert report.reference_positive_rate == 0.4
    assert report.current_positive_rate == 0.8
    assert report.relative_change > 0.2


def test_detect_concept_drift():
    y_true_ref = np.array([0, 1, 1, 0])
    y_pred_ref = np.array([0, 1, 0, 0])  # F1 = 0.5
    y_true_cur = np.array([0, 1, 1, 1])
    y_pred_cur = np.array([0, 0, 1, 0])  # F1 = 0.4
    detector = DriftDetector(reference_texts=[])

    report = detector.detect_concept_drift(
        y_true_reference=y_true_ref,
        y_pred_reference=y_pred_ref,
        y_true_current=y_true_cur,
        y_pred_current=y_pred_cur,
        threshold=0.1,
    )

    assert isinstance(report, ConceptDriftReport)
    assert report.detected is True
    assert report.reference_f1 >= 0.0
    assert report.current_f1 >= 0.0
    assert report.relative_drop > 0.0


def test_generate_report_full():
    ref_texts = ["a b", "c d", "e f"]
    ref_labels = np.array([0, 1, 0])
    current_texts = ["a b", "g h", "i j"]
    current_labels = np.array([1, 1, 0])
    y_pred_ref = np.array([0, 1, 0])
    y_pred_cur = np.array([1, 0, 0])

    detector = DriftDetector(reference_texts=ref_texts, reference_labels=ref_labels)
    report = detector.generate_report(
        current_texts=current_texts,
        y_reference=ref_labels,
        y_current=current_labels,
        y_true_reference=ref_labels,
        y_pred_reference=y_pred_ref,
        y_true_current=current_labels,
        y_pred_current=y_pred_cur,
        log_to_mlflow=False,
    )

    assert report.data_drift is not None
    assert report.target_drift is not None
    assert report.concept_drift is not None
    assert isinstance(report.has_drift(), bool)
    assert isinstance(report.to_dict(), dict)


def test_detect_simple_data_drift_no_labels():
    ref_texts = ["раз два", "три четыре"]
    detector = DriftDetector(reference_texts=ref_texts)

    current_texts = ["пять шесть", "семь восемь"]
    result = detector.detect(current_texts)

    assert "drift_detected" in result
    assert "drift_score" in result
    assert result["data_drift"] is not None
    assert result["target_drift"] is False
    assert result["concept_drift"] is False
